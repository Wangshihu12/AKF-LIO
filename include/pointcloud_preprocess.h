#ifndef AKF_LIO_POINTCLOUD_PREPROCESS_H
#define AKF_LIO_POINTCLOUD_PREPROCESS_H

#include <livox_ros_driver/CustomMsg.h>
#include <pcl_conversions/pcl_conversions.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "common_lib.h"

namespace velodyne_ros
{
    struct EIGEN_ALIGN16 Point
    {
        PCL_ADD_POINT4D;
        float intensity;
        float time;
        std::uint16_t ring;
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };
} // namespace velodyne_ros

// clang-format off
POINT_CLOUD_REGISTER_POINT_STRUCT(velodyne_ros::Point,
                                  (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)
                                      (float, time, time)(std::uint16_t, ring, ring))
// clang-format on

/*** Hesai_XT32 LiDAR点云数据结构 ***/
namespace xt32_ros
{
  struct EIGEN_ALIGN16 Point
  {
    PCL_ADD_POINT4D;    // PCL库中的4D点
    float intensity;    // 点云强度
    double timestamp;   // 时间戳
    std::uint16_t ring; // 激光雷达的环号
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };
} // namespace xt32_ros
POINT_CLOUD_REGISTER_POINT_STRUCT(xt32_ros::Point,
                                  (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(double, timestamp, timestamp)(std::uint16_t, ring, ring))
/*****************/

namespace ouster_ros
{
    struct EIGEN_ALIGN16 Point
    {
        PCL_ADD_POINT4D;
        float intensity;
        uint32_t t;
        uint16_t reflectivity;
        uint8_t ring;
        uint16_t ambient;
        uint32_t range;
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };
} // namespace ouster_ros

// clang-format off
POINT_CLOUD_REGISTER_POINT_STRUCT(ouster_ros::Point,
                                  (float, x, x)
                                      (float, y, y)
                                      (float, z, z)
                                      (float, intensity, intensity)
                                      // use std::uint32_t to avoid conflicting with pcl::uint32_t
                                  (std::uint32_t, t, t)
                                  (std::uint16_t, reflectivity, reflectivity)
                                  (std::uint8_t, ring, ring)
                                  (std::uint16_t, ambient, ambient)
                                  (std::uint32_t, range, range)
                                  )
// clang-format on

namespace akf_lio
{

    enum class LidarType
    {
        AVIA = 1,
        VELO32,
        OUST64
    };

    /**
     * point cloud preprocess
     * just unify the point format from livox/velodyne to PCL
     */
    class PointCloudPreprocess
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        PointCloudPreprocess() = default;
        ~PointCloudPreprocess() = default;

        /// processors
        void Process(const livox_ros_driver::CustomMsg::ConstPtr &msg, PointCloudType::Ptr &pcl_out);
        void Process(const sensor_msgs::PointCloud2::ConstPtr &msg, PointCloudType::Ptr &pcl_out);
        void Set(LidarType lid_type, double blind, int point_filter_num);

        // accessors
        double &Max_range() { return max_range_; }
        double &Blind() { return blind_; }
        int &NumScans() { return num_scans_; }
        int &PointFilterNum() { return point_filter_num_; }
        float &TimeScale() { return time_scale_; }
        LidarType GetLidarType() const { return lidar_type_; }
        void SetLidarType(LidarType lt) { lidar_type_ = lt; }
        bool deskew_ = true;

        struct MotorData
        {
            double timestamp;
            double angle;
            double angular_speed;
        };

        static int quick_power(int a, int b, int mod)
        {
            int res = 1;
            while (b)
            {
                if (b & 1)
                {
                    res = (long long)res * a % mod;
                }
                a = (long long)a * a % mod;
                b >>= 1;
            }
            return res;
        }

        static int int_dencypt(int c, int d, int n) { return quick_power(c, d, n); }

        double deg2rad(double deg) { return deg * M_PI / 180.0; }
        void rotatePoint(PointType2 &point, const Eigen::Affine3d &transform)
        {
            Eigen::Vector3d point_position(point.x, point.y, point.z + 0.0148);
            Eigen::Vector3d rotated_position = transform * point_position;
            point.x = rotated_position.x() + 0.1202;
            point.y = rotated_position.y();
            point.z = rotated_position.z() + 0.00927;
        }

        void undistortPointByMotor(PointType2 &point, const MotorData &motor_data)
        {
            // 初始化绕Z轴的120度旋转
            Eigen::Affine3d rotation_z = Eigen::Affine3d::Identity();
            rotation_z.rotate(Eigen::AngleAxisd(120.0 * M_PI / 180.0, Eigen::Vector3d::UnitZ()));

            // 计算电机绕X轴的旋转角度
            double rotation_angle = deg2rad(motor_data.angle);
            Eigen::Affine3d rotation_x = Eigen::Affine3d::Identity();
            rotation_x.rotate(Eigen::AngleAxisd(rotation_angle, Eigen::Vector3d::UnitX()));

            // 初始化绕Y轴的-15度旋转
            Eigen::Affine3d rotation_y = Eigen::Affine3d::Identity();
            rotation_y.rotate(Eigen::AngleAxisd(-15.0 * M_PI / 180.0, Eigen::Vector3d::UnitY()));

            // 组合旋转矩阵（应用顺序：Z → X → Y）
            Eigen::Affine3d combined_transform = rotation_y * rotation_x * rotation_z;

            rotatePoint(point, combined_transform);
        }

        void xt32_motor_handler(
            const sensor_msgs::PointCloud2::ConstPtr &msg,
            const std::deque<nav_msgs::Odometry::ConstPtr> &angle_msgs,
            PointCloudType::Ptr &pcl_out
        );

    private:
        void AviaHandler(const livox_ros_driver::CustomMsg::ConstPtr &msg);
        void Oust64Handler(const sensor_msgs::PointCloud2::ConstPtr &msg);
        void VelodyneHandler(const sensor_msgs::PointCloud2::ConstPtr &msg);

        PointCloudType cloud_full_, cloud_out_;

        LidarType lidar_type_ = LidarType::AVIA;
        int point_filter_num_ = 1;
        int num_scans_ = 6;
        double blind_ = 0.01;
        double max_range_ = 500.0;
        float time_scale_ = 1e-3;
        bool given_offset_time_ = false;
    };
} // namespace akf_lio

#endif // AKF_LIO_POINTCLOUD_PREPROCESS_H
