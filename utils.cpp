#pragma once

#include "amp_common/logging/logging_common.hpp"
#include "amp_msgs/SuppressingObject.h"
#include "amp_synapse_harvesting/types.hpp"

#include <nlohmann/json.hpp>

#include <optional>
#include <string>
#include <vector>

namespace amp::synapse_harvesting {

inline Polygon pointsMsgToPolygon(const std::vector<geometry_msgs::Point>& points)
{
  Polygon result;
  for (const auto& point : points) {
    boost::geometry::append(result, Point(point.x, point.y));
  }

  boost::geometry::append(result, Point(points.at(0).x, points.at(0).y));
  return result;
}

// Adds the points in the outer ring of a Boost polygon to a vector of ROS Point msgs.
inline void polygonToPointsMsg(const Polygon& polygon, std::vector<geometry_msgs::Point>& points)
{
  for (auto point : polygon.outer()) {
    geometry_msgs::Point p;
    p.x = point.get<0>();
    p.y = point.get<1>();
    p.z = 0;
    points.push_back(p);
  }
}

inline BoundingBox boundingBoxFromPolygon(const Polygon& src)
{
  BoundingBox result;
  boost::geometry::envelope(src, result);
  return result;
}

/**
 * @brief Helper that creates a SuppressingObject message.
 *
 * Used in fire planners to characterize what contaminant object caused suppression of a target object.
 *
 * @param id object id of the contaminant object.
 * @param overlap_distance metric describing the overlap of contaminant object and suppressed object.
 * @return SuppressingObject message used as suppression_data for neighbor suppression metrics.
 */
inline amp_msgs::SuppressingObject createSuppressingObject(const std::string& id, const double overlap_distance)
{
  amp_msgs::SuppressingObject tmp_msg;
  tmp_msg.id = id;
  tmp_msg.overlap_distance = overlap_distance;
  return tmp_msg;
}

/**
 * Applies a scale factor to the x and y dimension of a Polygon.
 */
inline std::optional<Polygon>
relativeScaleTransform(const Polygon& input_polygon, const double x_factor, const double y_factor)
{
  // Define transforms that translate target centroid to/from origin
  Point centroid;
  boost::geometry::centroid(input_polygon, centroid);
  boost::geometry::strategy::transform::translate_transformer<double, 2, 2> origin_to_centroid(
      centroid.get<0>(), centroid.get<1>());  // translates from origin to p centroid
  boost::geometry::multiply_value(centroid, -1.0);
  boost::geometry::strategy::transform::translate_transformer<double, 2, 2> centroid_to_origin(
      centroid.get<0>(), centroid.get<1>());  // translates from p centroid to origin

  // Translate target polygon to origin
  Polygon centered;
  boost::geometry::transform(input_polygon, centered, centroid_to_origin);

  // Define a matrix transformation with separate x and y scale factors
  std::vector<std::vector<double>> mat = { { 1.0 + x_factor, 0, 0 }, { 0, 1.0 + y_factor, 0 }, { 0, 0, 1.0 } };
  boost::geometry::strategy::transform::matrix_transformer<double, 2, 2> xy_scale_transformer(
      mat[0][0], mat[0][1], mat[0][2], mat[1][0], mat[1][1], mat[1][2], mat[2][0], mat[2][1], mat[2][2]);

  // Scale the target polygon, then translate back
  Polygon scaled;
  boost::geometry::transform(centered, scaled, xy_scale_transformer);
  Polygon output;
  boost::geometry::transform(scaled, output, origin_to_centroid);

  // If there's an issue with winding/closedness/etc try to correct.
  if (!boost::geometry::is_valid(output)) {
    boost::geometry::correct(output);
    if (!boost::geometry::is_valid(output)) {
      // If we cannot correct, then we could not form a valid fire plan and should not fire valves.
      AMP_ERROR("Unable to correct invalid output");
      return std::nullopt;
    }
    else {
      AMP_INFO("Corrected invalid output");
    }
  }

  return output;
}

/**
 * Stretches a Polygon a given distance along the x and y axes.
 *
 * The distance applied is a radius, so if x_dist = 0.05, then the resulting polygon will be streched 5 cm on the left
 * and the right, resulting in a 10 cm wider polygon.
 */
inline std::optional<Polygon>
absoluteScaleTransform(const Polygon& input_polygon, const double x_dist, const double y_dist)
{
  double dx = x_dist;
  double dy = y_dist;

  // Obtain target polygon's centroid
  Point centroid;
  boost::geometry::centroid(input_polygon, centroid);
  double c_x = centroid.get<0>();
  double c_y = centroid.get<1>();

  // Generate a scaled polygon
  Polygon scaled_polygon;
  for (auto p : input_polygon.outer()) {
    double p_x = p.get<0>();
    double p_y = p.get<1>();

    // The growth/shrink direction of a point depends on its position relative to the centroid
    double new_x = p_x + ((p_x >= c_x) ? dx : -dx);
    double new_y = p_y + ((p_y >= c_y) ? dy : -dy);

    // If a point's new x or y is on the 'other' side of the centroid, we get weird twists in the polygon that are
    // potentially dangerous. For now, we simply leave those points out of the new polygon.
    if (((p_x >= c_x) == (new_x >= c_x)) && ((p_y >= c_y) == (new_y >= c_y))) {
      boost::geometry::append(scaled_polygon, Point{ new_x, new_y });
    }
  }

  // If there's an issue with winding/closedness/etc try to correct.
  if (!boost::geometry::is_valid(scaled_polygon)) {
    boost::geometry::correct(scaled_polygon);
    if (!boost::geometry::is_valid(scaled_polygon)) {
      // If we cannot correct, then we could not form a valid fire plan and should not fire valves.
      AMP_ERROR("Unable to correct invalid scaled_polygon");
      return std::nullopt;
    }
    else {
      AMP_INFO("Corrected invalid scaled_polygon");
    }
  }

  return scaled_polygon;
}

// Translates the additional polygon so it's cenroid is moved to the main polygon's centroid, then takes union of the
// two polygons. Used for "OR"ing the minimum suppession polygon and a contaminant's suppression polygon.
inline std::optional<Polygon> unionOnCentroid(const Polygon& main_polygon, const Polygon& additional_polygon)
{
  Point main_centroid = boost::geometry::return_centroid<Point>(main_polygon);
  Point additional_centroid = boost::geometry::return_centroid<Point>(additional_polygon);

  // Translate the additional polygon to the main polygon
  boost::geometry::strategy::transform::translate_transformer<double, 2, 2> translate_additional_to_main(
      main_centroid.get<0>() - additional_centroid.get<0>(), main_centroid.get<1>() - additional_centroid.get<1>());

  Polygon translated_additional_polygon;
  boost::geometry::transform(additional_polygon, translated_additional_polygon, translate_additional_to_main);

  // Take their union
  std::vector<Polygon> union_results;
  boost::geometry::union_(main_polygon, translated_additional_polygon, union_results);

  // If we don't get exactly 1 resulting polygon, then we do not want to use the results.
  if (union_results.size() != 1) {
    AMP_ERROR(
        "Union of main polygon (" << boost::geometry::wkt(main_polygon) << ") and additional polygon ("
                                  << boost::geometry::wkt(additional_polygon) << ") did not yield a single polygon");
    return std::nullopt;
  }

  Polygon result_polygon = union_results[0];

  // If there's an issue with winding/closedness/etc try to correct.
  if (!boost::geometry::is_valid(result_polygon)) {
    boost::geometry::correct(result_polygon);
    if (!boost::geometry::is_valid(result_polygon)) {
      // If we cannot correct, then the union yield some polygon weirdness (perhaps it has holes, or inner rings).
      AMP_ERROR("Unable to correct invalid result_polygon");
      return std::nullopt;
    }
    else {
      AMP_INFO("Corrected invalid result_polygon");
    }
  }

  return result_polygon;
}

}  // namespace amp::synapse_harvesting
