#include "amp_camera_alignment/CameraAlignmentCalibrator.hpp"
#include "amp_common/logging/logging_common.hpp"
#include "amp_opencv_util_lib/depth_utils.hpp"

#include <opencv2/imgproc.hpp>

namespace amp::alignment {

CameraAlignmentCalibrator::CameraAlignmentCalibrator(std::unique_ptr<ArucoBoardFiducialDetector> fiducial_detector)
  : _fiducial_detector{ std::move(fiducial_detector) }
  , _marker_frame_rotation{ Eigen::AngleAxisd(M_PI, eigen_util::Point3::UnitZ()) }
{
}

bool CameraAlignmentCalibrator::getCameraPose(
    transform::Transform3t& marker_to_camera_pose,
    const opencv_util::StampedImage& image,
    std::shared_ptr<opencv_util::StampedImage> visualization)
{
  transform::Transform3t marker_pose;
  if (!_fiducial_detector->detect(marker_pose, image, visualization)) {
    AMP_DEBUG("Failed to detect fiducial.");
    return false;
  }

  // Detector provides location of the target relative to the camera's frame.
  // We want to report the camera's position relative to the conveyor frame.
  // So we do that conversion here.
  Eigen::Affine3d correction_tf = Eigen::Affine3d::Identity();
  Eigen::AngleAxisd correction_rotation(M_PI, Eigen::Vector3d::UnitZ());
  correction_tf *= correction_rotation;
  marker_to_camera_pose.transform = correction_tf * marker_pose.transform.inverse();

  marker_to_camera_pose.frame = common::CONVEYOR_FRAME_ID;
  marker_to_camera_pose.childFrameId = image.frame_id;
  marker_to_camera_pose.time = marker_pose.time;
  AMP_INFO("Detected camera position: " << marker_to_camera_pose.translation().transpose());

  if (visualization != nullptr) {
    drawVisualization(*visualization, marker_pose.translation());
  }

  return true;
}

void CameraAlignmentCalibrator::drawVisualization(
    opencv_util::StampedImage& visualization, const eigen_util::Point3& marker_position)
{
  static const cv::Scalar COLOR(255, 100, 100);
  cv::Point2i center;
  if (!opencv_util::PointToPixel(center.x, center.y, marker_position, visualization.camera_info)) {
    AMP_WARN("Failed to convert marker position for visualization.");
  }
  else {
    cv::circle(visualization.image, center, 10, COLOR, -1);
  }
}

}  // namespace amp::alignment
