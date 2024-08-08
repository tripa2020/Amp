#include "amp_json_tf_util_lib/conversions.hpp"

#include "amp_camera_alignment/CameraAlignmentNode.hpp"
#include "amp_common/common.hpp"
#include "amp_common/msg_utils.hpp"
#include "amp_msgs/EditConfigFile.h"
#include "amp_opencv_util_lib/ros_msg_conversions.hpp"

#include <cv_bridge/cv_bridge.h>
#include <std_msgs/Bool.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_eigen/tf2_eigen.h>

namespace amp::alignment {

CameraAlignmentNode::CameraAlignmentNode()
  : Node()
  , _timeout(ros::Time::now())
  , _nh(ros::NodeHandle())
  , _image_transport(_nh)
{
  _camera_info_subscriber = _nh.subscribe(common::CAMERA_INFO_TOPIC, 0, &CameraAlignmentNode::cameraInfoCallback, this);

  _image_subscriber =
      _image_transport.subscribe(common::COLOR_IMAGE_TOPIC, 1, &CameraAlignmentNode::imageCallback, this);

  _image_publisher = _image_transport.advertise(common::CAMERA_ALIGNMENT_VISUALIZATION_TOPIC, 1);

  _start_node_service =
      _nh.advertiseService(common::CAMERA_ALIGNMENT_SERVICE_START, &CameraAlignmentNode::startServiceCallback, this);

  _stop_node_service =
      _nh.advertiseService(common::CAMERA_ALIGNMENT_SERVICE_STOP, &CameraAlignmentNode::stopServiceCallback, this);

  _save_calibration_service = _nh.advertiseService(
      common::CAMERA_ALIGNMENT_SERVICE_SAVE, &CameraAlignmentNode::saveCalibrationServiceCallback, this);

  _save_config_service_client = _nh.serviceClient<amp_msgs::EditConfigFile>(common::CONFIGATRON_EDIT_CONFIG_FILE_TOPIC);

  _is_valid_publisher = _nh.advertise<std_msgs::Bool>(common::CAMERA_ALIGNMENT_VALID_TOPIC, 1);

  _status_publisher = _nh.advertise<std_msgs::Bool>(common::CAMERA_ALIGNMENT_STATUS_TOPIC, 1);

  ros::TimerCallback status_timer_callback = [&](const ros::TimerEvent&) {
    _status_publisher.publish(common::createBoolMsg(_node_active));
  };

  _status_timer = _nh.createTimer(ros::Duration(0.1), status_timer_callback);
}

bool CameraAlignmentNode::shutdown(double current_time)
{
  this->completeShutdown();
  return true;
}

void CameraAlignmentNode::configure(const configuration::NodeConfiguration& config)
{
  _config = config;
}

void CameraAlignmentNode::runImpl(double current_time)
{
}

void CameraAlignmentNode::cameraInfoCallback(const sensor_msgs::CameraInfo& msg)
{
  _camera_info = opencv_util::CameraInfo{ msg };
}

void CameraAlignmentNode::imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
  if (!this->isBoardSet()) {
    return;
  }

  if (_node_active && _camera_info) {
    if (ros::Time::now() < _timeout) {
      auto cv_ptr = cv_bridge::toCvCopy(msg, amp::common::MAIN_CAMERA_ENCODING);
      opencv_util::StampedImage stamped_image{ .image = cv_ptr->image.clone(),
                                               .time = msg->header.stamp.toSec(),
                                               .frame_id = msg->header.frame_id,
                                               .camera_info = *_camera_info };

      _last_calibration_result = this->onNewImage(stamped_image);
    }
    else {
      _node_active = false;
    }
  }
}

CameraAlignmentNode::CalibrationResult CameraAlignmentNode::onNewImage(const opencv_util::StampedImage& stamped_image)
{
  const auto visualization = std::make_shared<opencv_util::StampedImage>(
      opencv_util::StampedImage{ .image = stamped_image.image.clone(),
                                 .time = stamped_image.time,
                                 .frame_id = stamped_image.frame_id,
                                 .camera_info = stamped_image.camera_info });

  transform::Transform3t camera_pose;
  const auto is_aligned = _calibrator->getCameraPose(camera_pose, stamped_image, visualization);

  if (is_aligned) {
    AMP_TRACE("Target is aligned");

    geometry_msgs::TransformStamped pose_copy = tf2::eigenToTransform(camera_pose.transform);
    pose_copy.child_frame_id = "camera_live";
    pose_copy.header.frame_id = common::CONVEYOR_FRAME_ID;
    pose_copy.header.stamp = ros::Time(stamped_image.time);
    _tf_broadcaster.sendTransform(pose_copy);
  }
  else {
    AMP_TRACE("No alignment detected");
  }

  this->publishFeedback(is_aligned);
  _crosshairs.drawVisualization(*visualization, is_aligned);
  _image_publisher.publish(this->toImageMsg(visualization));

  return { is_aligned, camera_pose };
}

void CameraAlignmentNode::publishFeedback(bool is_valid) const
{
  _is_valid_publisher.publish(common::createBoolMsg(is_valid));
}

sensor_msgs::ImagePtr
CameraAlignmentNode::toImageMsg(const std::shared_ptr<opencv_util::StampedImage>& stamped_image) const
{
  auto msg = opencv_util::getRosMsg(stamped_image->image);
  msg->header.stamp = ros::Time(stamped_image->time);
  msg->header.frame_id = stamped_image->frame_id;
  return msg;
}

bool CameraAlignmentNode::startServiceCallback(
    amp_msgs::StartCameraAlignment::Request& req, amp_msgs::StartCameraAlignment::Response& res)
{
  res.success = true;

  auto board = aruco::createArucoBoardConfig(req.board_type.val);
  if (!board) {
    res.success = false;
    res.message = "Failed to create Aruco Board due to unknown board type";
    return true;
  }

  if (_node_active) {
    res.message = "Node already active, will timeout in 30 minutes";
  }
  else {
    res.message = "Node started, will timeout in 30 minutes";
    _node_active = true;
  }

  std::stringstream ss;
  ss << board->width << "x" << board->height << ", Marker Length: " << board->length
     << "m, Marker Separation: " << board->separation << "m";
  AMP_INFO("Board set to: " << ss.str());

  this->resetCalibrator(*board);
  _timeout = ros::Time::now() + ros::Duration(1800);
  return true;
}

bool CameraAlignmentNode::stopServiceCallback(amp_msgs::Trigger::Request& req, amp_msgs::Trigger::Response& res)
{
  if (_node_active) {
    _node_active = false;
    res.success = true;
    res.error_string = "Node stopped";
  }
  else {
    res.success = false;
    res.error_string = "Node already stopped";
  }

  this->resetCalibrator();
  return true;
}

bool CameraAlignmentNode::saveCalibrationServiceCallback(
    amp_msgs::Trigger::Request& req, amp_msgs::Trigger::Response& res)
{
  if (_node_active) {
    if (_last_calibration_result.valid) {
      res.success = this->writeConfig(_last_calibration_result.camera_pose);
      res.error_string = res.success ? "Configuration saved" : "Failed to save configuration";
    }
    else {
      res.success = false;
      res.error_string = "Current calibration is not valid";
    }
  }
  else {
    res.success = false;
    res.error_string = "Node is not active";
  }

  return true;
}

bool CameraAlignmentNode::writeConfig(const transform::Transform3t& camera_pose)
{
  // Note: Eigen matrices are stored in column-major order by default
  // https://eigen.tuxfamily.org/dox/group__TopicStorageOrders.html

  const nlohmann::json patch = {
    { "cameraTransform",
      { { "transform",
          { { "translation", { { "x", camera_pose.x() }, { "y", camera_pose.y() }, { "z", camera_pose.z() } } },
            { "rotation",
              { { "data",
                  { camera_pose.rotation().data()[0],
                    camera_pose.rotation().data()[3],
                    camera_pose.rotation().data()[6],
                    camera_pose.rotation().data()[1],
                    camera_pose.rotation().data()[4],
                    camera_pose.rotation().data()[7],
                    camera_pose.rotation().data()[2],
                    camera_pose.rotation().data()[5],
                    camera_pose.rotation().data()[8] } } } } } },
        { "frameId", camera_pose.frame },
        { "childFrameId", camera_pose.childFrameId } } }
  };

  amp_msgs::EditConfigFile srv;
  srv.request.user_name = "camera_tf_calibration";
  srv.request.patch = patch.dump();

  if (!_save_config_service_client.call(srv)) {
    return false;
  }

  if (!srv.response.success) {
    AMP_ERROR(srv.response.error_string);
    return false;
  }

  return true;
}

bool CameraAlignmentNode::isBoardSet() const
{
  return _calibrator != nullptr;
}

void CameraAlignmentNode::resetCalibrator(const ArucoBoard& board_config)
{
  auto fiducial_detector = std::make_unique<ArucoBoardFiducialDetector>(board_config);
  _calibrator = std::make_unique<CameraAlignmentCalibrator>(std::move(fiducial_detector));
}

void CameraAlignmentNode::resetCalibrator()
{
  _calibrator.reset();
}

}  // namespace amp::alignment
