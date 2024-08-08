#include "amp_object_detection/ObjectDetectionNode.hpp"

#include "amp_benchmark/RosBenchmarkFactory.hpp"
#include "amp_common/common.hpp"
#include "amp_common/system_type.hpp"
#include "amp_configuration_lib/ConfigurationProvider.hpp"
#include "amp_gpu_lib/gpu_info.hpp"
#include "amp_infra_lib/RosPublisher.hpp"
#include "amp_msgs/DetectionStatus.h"
#include "amp_msgs/NodeInfo.h"
#include "amp_msgs/Objects.h"
#include "amp_msgs/SetDetectionOverride.h"
#include "amp_notification_lib/amp_object_detection/ConfigurationFailed.hpp"
#include "amp_notification_lib/amp_object_detection/FailedToDetectObjects.hpp"
#include "amp_notification_lib/amp_object_detection/IncompatibleModel.hpp"
#include "amp_notification_lib/amp_object_detection/ModelNotCreated.hpp"
#include "amp_notification_lib/amp_object_detection/NeuralNetDecryptionError.hpp"
#include "amp_notification_lib/amp_object_detection/NoImageData.hpp"
#include "amp_notification_lib/amp_object_detection/PredictedUnknownMaterial.hpp"
#include "amp_notification_lib/amp_object_detection/RuntimeDeprecated.hpp"
#include "amp_object_detection/common_utils.hpp"
#include "amp_object_detection/triton/TritonObjectInference.hpp"
#include "amp_object_detection/types.hpp"

#include <boost/algorithm/clamp.hpp>
#include <boost/chrono.hpp>
#include <boost/thread.hpp>
#include <cv_bridge/cv_bridge.h>
#include "fmt/format.h"
#include <image_transport/image_transport.h>

namespace amp::object_detection {

namespace {
std::vector<amp_msgs::Point2u16> convertMaskToContour(const cv::Mat& mask, const cv::Rect2i& roi)
{
  // Check for a valid mask
  double mask_check = static_cast<double>(cv::sum(mask)[0]);
  if (mask_check == 0.0) {
    std::string error = "Return mask is all zeros which could indicate a GPU failure, restarting node.";
    AMP_ERROR(error);
    throw std::runtime_error(error);
  }

  std::vector<amp_msgs::Point2u16> result;

  // Resize the mask to the roi
  cv::Mat resized_mask;
  cv::resize(mask, resized_mask, roi.size());

  // Threshold and find contours
  resized_mask.convertTo(resized_mask, CV_8UC1, 255);
  // TODO(cooper): Investigate making the threshold configurable
  cv::threshold(resized_mask, resized_mask, 128, 255, CV_THRESH_BINARY);
  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Vec4i> hierarchy;
  cv::findContours(resized_mask, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

  if (contours.empty()) {
    AMP_DEBUG(
        "Unable to find contours from mask, the object is likely very small or something is wrong with the neural "
        "network, mask sum is: "
        << mask_check);
    return result;
  }

  // Find the max area contour
  auto contour = std::max_element(
      std::begin(contours), std::end(contours), [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
        return cv::contourArea(a) < cv::contourArea(b);
      });

  for (const auto& p : *contour) {
    amp_msgs::Point2u16 point;
    point.x = static_cast<uint16_t>(p.x);
    point.y = static_cast<uint16_t>(p.y);
    result.push_back(point);
  }
  return result;
}
}  // namespace

ObjectDetectionNode::ObjectDetectionNode()
  : Node()
  , _notifiers(&Node::getNotificationRunner())
  , _last_image_time(common::Instant::now())
{
  ros::NodeHandle nh;

  _belt_running_subscriber = nh.subscribe<amp_msgs::BoolStamped>(
      common::CONVEYOR_RUNNING_TOPIC, 1, [&](const auto& msg) { _belt_is_running = msg->value; });

  _detection_status_pub = nh.advertise<amp_msgs::DetectionStatus>(common::OBJECT_DETECTION_STATUS_TOPIC, 1);

  _detected_objects_pub = nh.advertise<amp_msgs::Objects>(common::DETECTED_OBJECTS_TOPIC, 1);

  _set_detection_override_service = nh.advertiseService(
      common::SET_OBJECT_DETECTION_OVERRIDE_SERVICE, &ObjectDetectionNode::setDetectionOverrideCb, this);

  benchmark::RosBenchmarkFactory benchmark_factory(nh);
  _bench = benchmark_factory.makeTimeSetBenchmark("object_detection/time", "time");
  _conveyor_occupancy_bench = benchmark_factory.makeCounterBenchmark("conveyor_occupancy", "conveyor_occupancy");

  _notify_bad_label = _notifiers.registerTimedNotifier<notification::PredictedUnknownMaterial>([this] {
    const std::shared_lock lock(_bad_labels_mutex);
    return nlohmann::json{ { "bad_labels", _bad_labels } };
  });

  _notifiers.registerNotifier<notification::NeuralNetDecryptionError>([&] { return _failed_to_decrypt; });

  _notifiers.registerNotifier<notification::NoImageData>([&] {
    return (
        _inference_model != nullptr && _inference_model->isReady() &&
        (_last_image_time.elapsed().as_secs() > LAST_IMAGE_TIMEOUT_S));
  });

  _notifiers.registerNotifier<notification::ModelNotCreated>([&] {
    return (
        _inference_model != nullptr && !_inference_model->isReady() && _model_load_time.has_value() &&
        (_model_load_time->elapsed().as_secs() > CREATE_MODEL_TIMEOUT));
  });

  _notifiers.registerNotifier<notification::FailedToDetectObjects>([&] { return _failed_to_detect_objects; });

  _notifiers.registerNotifier<notification::IncompatibleModel>([&] { return _failed_to_validate_model; });

  _notifiers.registerNotifier<notification::RuntimeDeprecated>([&] { return _deprecation_warning; });

  _notifiers.registerNotifier<notification::ConfigurationFailed>(
      [this] { return _configuration_error_msg.has_value(); },
      [this] {
        return nlohmann::json{ { "msg", _configuration_error_msg.value_or("") } };
      });
}

constexpr char ObjectDetectionNode::DECRYPTION_PATH[];
constexpr double ObjectDetectionNode::CREATE_MODEL_TIMEOUT;

void ObjectDetectionNode::configure(const configuration::NodeConfiguration& config)
{
  const std::unique_lock lock_labelset(_labelset_mutex);
  const std::unique_lock lock_bad_labels(_bad_labels_mutex);
  _active_image_area.reset();
  _configuration_error_msg.reset();

  const auto recipe = configuration::RecipeProvider().tryGetConfiguration(config.prod.recipe);
  _config = config;

  _labelset = configuration::NeuralNetworkMetadata::create(recipe.nnId, gpu::getAllSystemGpuTypes());
  if (!_labelset) {
    // TODO: propagate more specific error message from NeuralNetworkMetadata.
    _configuration_error_msg = "Unable to load neural network metadata. NN may not have a labels_v6; resolve by "
                               "updating to a more recent NN. NN may not be present on disk. NN may not be optimized "
                               "for your GPU.";
    AMP_ERROR(*_configuration_error_msg);
    return;
  }
  _deprecation_warning = false;
  if (const auto o_runtime = _labelset->getRuntime(); o_runtime.has_value()) {
    if (*o_runtime == configuration::NeuralNetworkMetadata::InferenceRuntime::TENSORFLOW) {
      AMP_ERROR("Neural network is using the deprecated tensorflow runtime.");
      _deprecation_warning = true;
      return;
    }
  }

  _camera_region_of_interest = config.prod.camera.regionOfInterest;
  _camera_exclusion_box = config.prod.camera.exclusionBox;
  _vault_decryption_params = _config.dev.objectDetection.vault;

  const auto api_type = _labelset->model().apiType;
  if (ApiTypeMap.find(api_type) == ApiTypeMap.end()) {
    _configuration_error_msg = "Unknown neural network api type provided: " + api_type;
    AMP_ERROR(*_configuration_error_msg);
    return;
  }
  _nn_api_type = ApiTypeMap.at(api_type);

  ros::NodeHandle nh;
  image_transport::ImageTransport it(nh);

  if (_nn_api_type != ApiType::MASKS_NIR) {
    // Do a basic subscription to our rgb image callback
    _rgb_subscriber.emplace(nh, common::COLOR_IMAGE_TOPIC, 3);
    _rgb_subscriber->registerCallback(
        [this](const sensor_msgs::ImageConstPtr& rgb) { this->handleImage(rgb, nullptr); });

    // Clear out any active NIR ROS connections so we're in a clean state if switching between the two
    _camera_synchronizer = std::nullopt;
    _nir_subscriber = std::nullopt;
  }
  else {
    // Otherwise setup the synchronized subscription to rgb and nir data
    _rgb_subscriber.emplace(nh, common::COLOR_IMAGE_TOPIC, 1);
    _nir_subscriber.emplace(nh, nir_common_lib::NIR_SYNC_FRAME_TOPIC, 1);
    _camera_synchronizer.emplace(*_rgb_subscriber, *_nir_subscriber, 10);
    // Note: it appears for some reason time synchronizer is incompatible with lambda and we have to use boost::bind...
    _camera_synchronizer->registerCallback(boost::bind(&ObjectDetectionNode::handleImage, this, _1, _2));
  }

  _bad_labels.clear();

  auto nn_path = _labelset->getNetworkDirPath().string();
  if (nn_path.empty()) {
    _configuration_error_msg = "Neural network path is empty";
    AMP_ERROR(*_configuration_error_msg);
    return;
  }

  // TODO: we're probably going to want to switch from "encrypted" being a dev.json parameter
  // to auto-detecting it off of the file extensions
  if (config.dev.objectDetection.model.encrypted) {
    auto status = decrypt(nn_path + "/network.tar.gz.crypt", DECRYPTION_PATH);
    if (status != 0) {
      _failed_to_decrypt = true;
      AMP_ERROR("Decryption failed with status code (" << status << ").");
      // This request shutdown restarts the object detection node when decryption fails.
      // This is a workaround for network decrypt failures such as this:
      // https://amprobotics.atlassian.net/browse/AXON-3503
      AMP_ERROR("Requesting shutdown after neural network decryption failure.");
      this->requestShutdown();
      return;
    }
    else {
      AMP_INFO("Successfully decrypted neural net!");
    }
  }
  else {
    // Decrypt also moves the resulting files to the right location
    // If the network isn't encrypted we still need to move them to that
    // location. This is especially important for triton where moving
    // them actually spans them across docker images.
    std::string cmd = fmt::format(
        "sudo rm -rf {0}/network/* && mkdir -p {0}/network && cp -r {1}/* {0}/network && mkdir "
        "{0}/network/model_ensemble/1",
        std::string(DECRYPTION_PATH),
        nn_path);
    AMP_INFO("Running command: " << cmd);
    int ret = std::system(cmd.c_str());
    if (ret != 0) {
      AMP_ERROR("Non-Zero exit code from NN copy command encountered: " << ret);
      // We're going to continue anyway to just see if things happen to work
    }
  }

  auto addr = common::SocketAddrV4::create(std::string(config.dev.objectDetection.triton.socketAddr));
  if (!addr) {
    _configuration_error_msg = "Invalid socket address for Triton Inference Server";
    AMP_ERROR(*_configuration_error_msg);
    return;
  }
  AMP_INFO("Creating connection to Triton Inference Server at: " << addr->to_string());
  _inference_model = TritonObjectInference::create(*addr, config.dev.objectDetection.model, _nn_api_type);
  _model_load_time = common::Instant::now();
}

nlohmann::json ObjectDetectionNode::getConfigMask() const
{
  return { {
      _config.prod.camera_varname,
      { { _config.prod.camera.regionOfInterest_varname,
          nlohmann::json::parse(models::to_string(_config.prod.camera.regionOfInterest)) },
        { _config.prod.camera.exclusionBox_varname,
          nlohmann::json::parse(models::to_string(_config.prod.camera.exclusionBox)) } },
  } };
}

void ObjectDetectionNode::runImpl(double current_time)
{
}

size_t ObjectDetectionNode::calculateActiveImageArea(
    const models::regionOfInterest_t& roi, const models::regionOfInterest_t& excl) const
{
  size_t non_roi_area = (roi.lowerRightPixelX - roi.upperLeftPixelX) * (roi.lowerRightPixelY - roi.upperLeftPixelY);

  models::regionOfInterest_t adjusted_excl;
  adjusted_excl.upperLeftPixelX = std::max(roi.upperLeftPixelX, excl.upperLeftPixelX);
  adjusted_excl.lowerRightPixelX = std::min(roi.lowerRightPixelX, excl.lowerRightPixelX);
  adjusted_excl.upperLeftPixelY = std::max(roi.upperLeftPixelY, excl.upperLeftPixelY);
  adjusted_excl.lowerRightPixelY = std::min(roi.lowerRightPixelY, excl.lowerRightPixelY);

  size_t adjusted_excl_area = (adjusted_excl.lowerRightPixelX - adjusted_excl.upperLeftPixelX) *
                              (adjusted_excl.lowerRightPixelY - adjusted_excl.upperLeftPixelY);
  return non_roi_area - adjusted_excl_area;
}

void ObjectDetectionNode::handleImage(
    const sensor_msgs::ImageConstPtr& image_ptr, const sensor_msgs::ImageConstPtr& nir_ptr)
{
  _last_image_time = common::Instant::now();
  if (!_inference_model || !_inference_model->isReady()) {
    return;
  }

  std::optional<cv::Mat> nir_data = std::nullopt;
  if (_nn_api_type == ApiType::MASKS_NIR) {
    // If we're configure for nir, pull the data out of the ROS message and we'll forward to triton
    if (nir_ptr == nullptr) {
      AMP_ERROR("Handle image callback called with no NIR data, while an NIR network is configured");
      return;
    }
    auto nir_image = cv_bridge::toCvShare(nir_ptr, "mono8");
    nir_data = nir_image->image;
  }

  amp_msgs::DetectionStatus detection_status;
  detection_status.on_hold = _object_detection_on_hold;
  detection_status.hold_overridable = _object_detection_hold_overridable;
  _detection_status_pub.publish(detection_status);

  // Don't run object detection on stopped belt
  if (_belt_is_running.has_value()) {
    if (_belt_is_running.value() == false) {
      _object_detection_hold_overridable = true;
      if (!_force_detection_ON) {
        _object_detection_on_hold = true;
        return;
      }
    }
    else {
      _object_detection_hold_overridable = false;
      _force_detection_ON = false;
    }
  }

  {
    std::lock_guard<std::mutex> lock(_bench_mutex);
    _bench.beginNext("decode");
  }
  cv_bridge::CvImageConstPtr frame;
  try {
    // Note: the NN expects pixels in RGB order, this node is explicitly and intentionally different
    // than the Axon standard of MAIN_CAMERA_ENCODING found in common_constants.hpp
    frame = cv_bridge::toCvShare(image_ptr, sensor_msgs::image_encodings::RGB8);
  }
  catch (cv_bridge::Exception& e) {
    AMP_ERROR("Exception while decoding image message: " << e.what());
    return;
  }

  // TODO: why is all of this happening in this fuction?
  // Why are we re-calculating this ever frame?
  if (!_active_image_area.has_value()) {
    models::regionOfInterest_t roi = _camera_region_of_interest;
    if (roi.upperLeftPixelX < 0) {
      roi.upperLeftPixelX = 0;
    }
    if (roi.upperLeftPixelY < 0) {
      roi.upperLeftPixelY = 0;
    }
    if (roi.lowerRightPixelX < 0) {
      roi.lowerRightPixelX = frame->image.cols;
    }
    if (roi.lowerRightPixelY < 0) {
      roi.lowerRightPixelY = frame->image.rows;
    }

    models::regionOfInterest_t exclusion_box = _camera_exclusion_box;
    bool exclusion_box_enabled =
        (exclusion_box.upperLeftPixelX >= 0 && exclusion_box.upperLeftPixelY >= 0 &&
         exclusion_box.lowerRightPixelX >= 0 && exclusion_box.lowerRightPixelY >= 0);
    if (exclusion_box_enabled) {
      if (exclusion_box.lowerRightPixelX > frame->image.cols) {
        exclusion_box.lowerRightPixelX = frame->image.cols;
      }
      if (exclusion_box.lowerRightPixelY > frame->image.rows) {
        exclusion_box.lowerRightPixelY = frame->image.rows;
      }
      _active_image_area = calculateActiveImageArea(roi, exclusion_box);
    }
    else {
      _active_image_area = (roi.lowerRightPixelX - roi.upperLeftPixelX) * (roi.lowerRightPixelY - roi.upperLeftPixelY);
    }
  }

  {
    std::lock_guard<std::mutex> lock(_bench_mutex);
    _bench.beginNext("infer");
  }
  _object_detection_on_hold = false;
  const auto detection_complete_cb =
      [this, cols = frame->image.cols, rows = frame->image.rows, header = image_ptr->header](const auto& results) {
        auto objects = convertResultsToObjects(results, rows, cols);
        amp_msgs::Objects objects_msg;
        objects_msg.header = header;
        objects_msg.frames_processed = 1;
        objects_msg.objects = objects;
        _detected_objects_pub.publish(objects_msg);
        {
          std::lock_guard<std::mutex> lock(_bench_mutex);
          _bench.collect();
        }

        // Calculates the belt occupancy in pixel space
        auto contour_area = calculateContourArea(objects_msg.objects, cv::Size(rows, cols));

        // Add total occupancy ratio to the benchmark
        {
          std::lock_guard<std::mutex> lock(_cob_mutex);
          _conveyor_occupancy_bench.add(contour_area / _active_image_area.value());
          _conveyor_occupancy_bench.collect();
        }

        // Report objects without a corresponding label in the labelset for notification
        const std::shared_lock lock_labelset(_labelset_mutex);
        const std::unique_lock lock_bad_labels(_bad_labels_mutex);
        for (const auto& object : objects_msg.objects) {
          if (!_labelset->has(object.primary_classification.name)) {
            if (std::find(_bad_labels.begin(), _bad_labels.end(), object.primary_classification.name) ==
                _bad_labels.end()) {
              // Avoid duplicate entries in _bad_labels
              _bad_labels.push_back(object.primary_classification.name);
            }
            _notify_bad_label();
            AMP_ERROR("Neural Network Predicted Unknown Material: '" << object.primary_classification.name << "'");
          }
        }
      };

  bool success{ false };
  success = _inference_model->detectObjects(frame->image, detection_complete_cb, nir_data);

  if (!success) {
    AMP_ERROR("Inference failed!");
  }
}

std::vector<amp_msgs::Object>
ObjectDetectionNode::convertResultsToObjects(const ObjectDetectionModelResults& results, int rows, int cols) const
{
  std::vector<amp_msgs::Object> objects;
  for (auto result_idx = 0u; result_idx < results.n_objects; ++result_idx) {
    const auto& roi = results.rois.at(result_idx);
    // Prevent processing 0-area ROI's that Effecientdet can return
    if (roi.area() == 0) {
      continue;
    }

    amp_msgs::Object object;
    object.id = _uuid_rand_gen.getUuid();

    const auto& [primary_classification, attributes] = results.labels.at(result_idx);
    object.primary_classification.name = primary_classification;

    const auto& [primary_score, attribute_scores] = results.scores.at(result_idx);
    object.primary_classification.score = primary_score;

    assert(attribute_scores.size() == attributes.size());
    for (auto attribute_idx = 0u; attribute_idx < attributes.size(); ++attribute_idx) {
      amp_msgs::ObjectAttribute attribute;
      attribute.name = attributes.at(attribute_idx);
      attribute.score = attribute_scores.at(attribute_idx);
      object.attributes.emplace_back(attribute);
    }

    // The ROI will get clipped below to prevent access out of frame and an overflow when cast to unsigned.
    cv::Point2i tl_clamp(
        boost::algorithm::clamp(roi.tl().x, 0, cols - 1), boost::algorithm::clamp(roi.tl().y, 0, rows - 1));
    cv::Point2i br_clamp(
        boost::algorithm::clamp(roi.br().x, 0, cols - 1), boost::algorithm::clamp(roi.br().y, 0, rows - 1));
    cv::Rect2i roi_clamp(tl_clamp, br_clamp);

    if (roi_clamp.width == 0 || roi_clamp.height == 0) {
      AMP_ERROR(
          "Object " << object.id << " clamped to 0 height and/or width. Ignoring object.\n"

                    << "Original ROI: " << roi.size().width << "x" << roi.size().height << " (" << roi.tl().x << ","
                    << roi.tl().y << ") (" << roi.br().x << "," << roi.br().y << ")\n"

                    << "Clamped to: " << roi_clamp.size().width << "x" << roi_clamp.size().height << " ("
                    << roi_clamp.tl().x << "," << roi_clamp.tl().y << ") (" << roi_clamp.br().x << ","
                    << roi_clamp.br().y << ")\n"
                    << "Input image size: " << cols << "x" << rows);
      continue;
    }

    object.roi.top_left.x = static_cast<uint16_t>(roi_clamp.tl().x);
    object.roi.top_left.y = static_cast<uint16_t>(roi_clamp.tl().y);
    object.roi.bottom_right.x = static_cast<uint16_t>(roi_clamp.br().x);
    object.roi.bottom_right.y = static_cast<uint16_t>(roi_clamp.br().y);

    if (_inference_model->hasGripTargetInfo()) {
      object.grip_location.x = results.grip_locations.at(result_idx).first;
      object.grip_location.y = results.grip_locations.at(result_idx).second;

      cv_bridge::CvImage cv_image;
      cv_image.header.stamp = ros::Time::now();
      cv_image.image = results.heatmaps.at(result_idx);
      cv_image.encoding = sensor_msgs::image_encodings::TYPE_32FC1;

      cv_image.toImageMsg(object.heatmap);
    }
    else {
      object.grip_location.x = -1;
      object.grip_location.y = -1;
    }

    object.contour = convertMaskToContour(results.masks[result_idx], roi_clamp);

    if (object.contour.size() > 0) {
      objects.push_back(object);
    }
  }

  return objects;
}

int ObjectDetectionNode::decrypt(const std::string src, const std::string dst)
{
  AMP_INFO("Model is encrypted. Decrypting and unpacking to '" << dst << "'.");

  // Note: Carter I'm annoyed that `sudo` is being used here, it is making the files owned by root
  // which is not ideal.
  // Note: Added `sudo rm -rf` at the front of this to deal with switching models
  // We're having issues where the previous model is still loaded by triton after switching
  // I don't think this helped much, but is slightly cleaner.
  // Triton recommends not using Poll mode:
  // https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_management.html
  std::string cmd = fmt::format(
      "sudo rm -rf {0}/network/* && mkdir -p {0} && "
      "vault-crypt --vault-url {1} --vault-noverify --vault-auth-method gcp "
      "--vault-role-id {2} --gcp-service-account-file {3} decrypt --use-gsm {5} {4} - | "
      "sudo tar -xz -C {0}",
      dst,
      _vault_decryption_params.url,
      _vault_decryption_params.roleId,
      _vault_decryption_params.gcpServiceAccountFile,
      src,
      (_vault_decryption_params.decryptionTool == "GSM"));
  AMP_DEBUG("Running command '" << cmd << "'.");

  return std::system(cmd.c_str());
}

bool ObjectDetectionNode::setDetectionOverrideCb(
    amp_msgs::SetDetectionOverride::Request& req, amp_msgs::SetDetectionOverride::Response& res)
{
  if (req.override == true) {
    this->_force_detection_ON = true;
  }
  else {
    this->_force_detection_ON = false;
  }
  res.success = true;
  return res.success;
}

}  // namespace amp::object_detection
