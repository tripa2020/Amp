#include "amp_calibration_lib/CameraCalibrator.hpp"

#include <numeric>
#include <sstream>

#include "fmt/format.h"
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

namespace amp::camera_calibration {

std::string prettyPrint(const amp_msgs::CalibrationReport& cr)
{
  std::stringstream ss;
  if (cr.calibration_succeeded) {
    ss << "Camera Info: " << cr.camera_info << std::endl;
  }
  else {
    ss << "Calibration failed." << std::endl;
  }
  if (cr.std_dev.empty()) {
    ss << "Standard Deviation empty." << std::endl;
  }
  else {
    static const std::vector<std::string> PARAM_NAMES{ "fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3",
                                                       "k4", "k5", "k6", "s1", "s2", "s3", "s4", "tx", "ty" };
    ss << "Standard deviation:" << std::endl;
    for (std::size_t i = 0; i < std::min(PARAM_NAMES.size(), cr.std_dev.size()); ++i) {
      ss << "\t" << PARAM_NAMES[i] << ": " << cr.std_dev[i] << std::endl;
    }
  }
  ss << "Detection coverage: " << cr.coverage_fraction << std::endl;
  ss << "Number of detections: " << cr.num_detections << std::endl;
  ss << "Issues: ";
  if (cr.issues.empty()) {
    ss << "None.";
  }
  else {
    ss << std::endl;
    for (const auto& i : cr.issues) {
      ss << std::endl << "\t" << i;
    }
  }
  return ss.str();
}

CameraCalibrator::CameraCalibrator(
    int image_width,
    int image_height,
    double minimum_distance_between_detections,
    CharucoDetector::Config detector_config)
  : _image_size(image_width, image_height)
  , _minimum_distance_between_detections(minimum_distance_between_detections)
  , _detector(std::move(detector_config))
{
}

amp_msgs::DetectionReport CameraCalibrator::addImage(
    const cv::Mat& raw_image,
    const opencv_util::CameraInfo& cam_info,
    benchmark::TimeSetBenchmark& benchmark,
    cv::Mat& visualization)
{
  amp_msgs::DetectionReport report;
  report.detection_succeeded = false;
  report.detection_added = false;
  report.coverage_fraction = _coverage_fraction;
  report.new_detections = _new_detections;
  report.total_detections = _detected_corners.size();
  if (!_mu.try_lock()) {
    // Calibration is in progress.
    return report;
  }
  std::lock_guard lock(_mu, std::adopt_lock);

  auto detection = _detector.detect(
      raw_image, cam_info.getRawCameraMatrix(), cam_info.getDistortionParams(), visualization, benchmark);
  if (detection.ids.empty()) {
    return report;
  }
  report.detection_succeeded = true;
  report.blurriness_1 = detection.blurriness_1;
  report.blurriness_2 = detection.blurriness_2;

  if (detection.blurriness_1.measurements.size() > 0 && detection.blurriness_2.measurements.size() > 0) {
    // To create a stable combined measurement that is less affected by outliers than an arithmetic average, calculate
    // the median of each group of measurements, and average the medians.
    auto& b1 = detection.blurriness_1.measurements;
    auto& b2 = detection.blurriness_2.measurements;

    // Note that nth_element rearranges the elements.
    std::nth_element(
        b1.begin(), b1.begin() + b1.size() / 2, b1.end(), [](const auto& a, const auto& b) { return a.px < b.px; });
    std::nth_element(
        b2.begin(), b2.begin() + b2.size() / 2, b2.end(), [](const auto& a, const auto& b) { return a.px < b.px; });
    report.combined_blurriness_px = (b1[b1.size() / 2].px + b2[b2.size() / 2].px) / 2;
  }
  else {
    report.combined_blurriness_px = 0;
  }

  // Detections at a large angle to the camera have higher error. Reject detections at more than 40 degrees.
  constexpr double BAD_ANGLE_THRESHOLD{ 40 };
  // The available eulerAngles() from Eigen does a really bad job of determining minimum yaw and pitch, the rotations we
  // care about, so this custom euler angle function is provided.
  auto eulerAngles = [](const Eigen::Matrix3d& r) {
    if (r(0, 2) < -0.998) {
      return Eigen::Vector3d(-atan2(r(1, 0), r(1, 1)), -M_PI_2, 0);
    }
    if (r(0, 2) > 0.998) {
      return Eigen::Vector3d(atan2(r(1, 0), r(1, 1)), M_PI_2, 0);
    }
    return Eigen::Vector3d(atan2(-r(1, 2), r(2, 2)), asin(r(0, 2)), atan2(-r(0, 1), r(0, 0)));
  };

  // Pre-rotate the transform by 180 degrees about the Y axis, so that a target directly facing the camera has 0 yaw and
  // pitch.
  Eigen::Vector3d euler_angles =
      eulerAngles(Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitY()) * detection.camera_t_target.rotation());
  euler_angles *= 180 / M_PI;
  if (std::abs(euler_angles(0)) > BAD_ANGLE_THRESHOLD || std::abs(euler_angles(1)) > BAD_ANGLE_THRESHOLD) {
    return report;
  }

  // If the detection is substantially close to a previous detection, save the one with the larger number of detected
  // points. This helps avoid adding detections that do not add useful information. For example, if a target is
  // stationary in the camera frame.
  benchmark.beginNext("rejectCloseDetections");
  std::vector<double> detection_diffs;
  constexpr double CONVERSION_FACTOR_1DEG_EQUALS_1CM = 0.01 * 180 / M_PI;
  // Calculate the diffs before doing min_element, because otherwise min_element would be making 2 calls to
  // transform::diff per element. This could be one pipeline with cpp20 ranges.
  for (const auto& camera_t_target : _camera_t_targets) {
    detection_diffs.push_back(
        transform::diff(detection.camera_t_target, camera_t_target, CONVERSION_FACTOR_1DEG_EQUALS_1CM));
  }
  auto closest_detection = std::min_element(detection_diffs.begin(), detection_diffs.end());

  if (closest_detection != detection_diffs.end() && *closest_detection < _minimum_distance_between_detections) {
    size_t i = std::distance(detection_diffs.begin(), closest_detection);
    if (_detected_ids[i].size() >= detection.ids.size()) {
      // Discard the new detection and return.
      return report;
    }
    else {
      // Discard the old detection.
      _detected_corners.erase(_detected_corners.begin() + i);
      _detected_ids.erase(_detected_ids.begin() + i);
      _camera_t_targets.erase(_camera_t_targets.begin() + i);
      _new_detections -= 1;
    }
  }

  // Add the new detection.
  report.detection_added = true;
  _detected_corners.push_back(std::move(detection.corners));
  _detected_ids.push_back(std::move(detection.ids));
  _camera_t_targets.push_back(std::move(detection.camera_t_target));
  _new_detections += 1;
  report.new_detections = _new_detections;
  report.total_detections = _detected_corners.size();
  benchmark.beginNext("detectionCoverage");
  report.coverage_fraction = _coverage_fraction = detectionCoverage();
  return report;
}

amp_msgs::CalibrationReport CameraCalibrator::calibrate(const opencv_util::CameraInfo& camera_info)
{
  std::lock_guard lock(_mu);
  cv::Size image_size(camera_info.width(), camera_info.height());
  amp_msgs::CalibrationReport report;
  report.num_detections = _detected_corners.size();
  report.num_points = std::accumulate(
      _detected_ids.begin(), _detected_ids.end(), 0, [](int a, const std::vector<int>& ids) { return a + ids.size(); });
  constexpr int min_detections{ 10 };
  constexpr double max_reprojection_error{ 0.5 };
  if (report.num_detections < min_detections) {
    report.issues.push_back(fmt::format("Not enough detections: {} < {}", report.num_detections, min_detections));
    report.calibration_succeeded = false;
    return report;
  }
  cv::Mat camera_matrix = cv::Mat::zeros(cv::Size(3, 3), CV_64F);
  std::vector<double> distortion_coeffs;
  double reprojection_error{ 0.0 };
  try {
    reprojection_error = cv::aruco::calibrateCameraCharuco(
        _detected_corners,
        _detected_ids,
        _detector.getBoard(),
        image_size,
        camera_matrix,
        distortion_coeffs,
        cv::noArray(),
        cv::noArray(),
        report.std_dev,
        cv::noArray(),
        cv::noArray(),
        cv::CALIB_FIX_PRINCIPAL_POINT);
  }
  catch (cv::Exception& e) {
    report.issues.push_back(fmt::format("Exception in calibrateCameraCharuco: {}", e.what()));
    report.calibration_succeeded = false;
    _new_detections = 0;
    return report;
  }

  report.average_reprojection_error = reprojection_error / report.num_points;
  if (report.average_reprojection_error > max_reprojection_error) {
    report.issues.push_back(fmt::format(
        "Reprojection error too high: {:.2f} > {:.2f}", report.average_reprojection_error, max_reprojection_error));
  }

  // Assess coefficient of variance (CoV) (mean divided by std dev) to determine acceptability of calibration.
  double fx_cov{ report.std_dev[0] / camera_matrix.at<double>(0, 0) * 100 };
  double fy_cov{ report.std_dev[1] / camera_matrix.at<double>(1, 1) * 100 };
  double cx_cov{ report.std_dev[2] / camera_matrix.at<double>(0, 2) * 100 };
  double cy_cov{ report.std_dev[3] / camera_matrix.at<double>(1, 2) * 100 };
  constexpr double max_cov{ 1.0 };  // percent

  if (fx_cov > max_cov) {
    report.issues.push_back(fmt::format(
        "Variance of focal length fx too high: {:.1f}% > {:.1f}%. Capture more detections of angled targets.",
        fx_cov,
        max_cov));
  }
  if (fy_cov > max_cov) {
    report.issues.push_back(fmt::format(
        "Variance of focal length fy too high: {:.1f}% > {:.1f}%. Capture more detections of angled targets.",
        fy_cov,
        max_cov));
  }
  if (cx_cov > max_cov) {
    report.issues.push_back(fmt::format("Variance of optical center cx too high: {:.1f}% > {:.1f}%", cx_cov, max_cov));
  }
  if (cy_cov > max_cov) {
    report.issues.push_back(fmt::format("Variance of optical center cy too high: {:.1f}% > {:.1f}%", cy_cov, max_cov));
  }

  report.coverage_fraction = detectionCoverage();
  constexpr double min_coverage_fraction{ 0.5 };
  if (report.coverage_fraction < min_coverage_fraction) {
    report.issues.push_back(fmt::format(
        "Need more coverage: {:.0f}% > {:.0f}%", report.coverage_fraction * 100, min_coverage_fraction * 100));
  }

  try {
    cv::Mat rectified_camera_matrix{ cv::getOptimalNewCameraMatrix(camera_matrix, distortion_coeffs, image_size, 0.0) };
    opencv_util::CameraInfo camera_info(image_size, camera_matrix, distortion_coeffs, rectified_camera_matrix);
    report.camera_info = camera_info.getInfo();
    report.calibration_succeeded = true;
  }
  catch (cv::Exception& e) {
    report.issues.push_back(fmt::format("Exception thrown in getOptimalNewCameraMatrix: {}", e.what()));
    report.calibration_succeeded = false;
  }
  _new_detections = 0;
  return report;
}

std::future<amp_msgs::CalibrationReport> CameraCalibrator::calibrateAsync(opencv_util::CameraInfo camera_info)
{
  return std::async(std::launch::async, &CameraCalibrator::calibrate, this, camera_info);
}

double CameraCalibrator::detectionCoverage() const
{
  if (_image_size.area() == 0) {
    return 0;
  }
  cv::Mat coverage{ cv::Mat::zeros(_image_size, CV_8UC1) };
  for (const auto& points : _detected_corners) {
    cv::Mat hull;
    cv::convexHull(points, hull);
    hull.convertTo(hull, CV_32S);
    cv::fillConvexPoly(coverage, hull, cv::Scalar(1));
  }
  return cv::sum(coverage)[0] / _image_size.area();
}

void CameraCalibrator::overlayDetectionCoverage(cv::Mat& input) const
{
  std::lock_guard lock(_mu);
  for (const auto& points : _detected_corners) {
    cv::Mat temp{ cv::Mat::zeros(input.size(), CV_8UC3) };
    cv::Mat hull;
    cv::convexHull(points, hull);
    hull.convertTo(hull, CV_32S);
    cv::fillConvexPoly(temp, hull, cv::Scalar(0, 255, 0));
    cv::addWeighted(temp, .2, input, 1, 0, input);
  }
}

void CameraCalibrator::clearDetections()
{
  std::lock_guard lock(_mu);
  _detected_corners.clear();
  _detected_ids.clear();
  _camera_t_targets.clear();
  _new_detections = 0;
  _coverage_fraction = 0.0;
}

};  // namespace amp::camera_calibration
