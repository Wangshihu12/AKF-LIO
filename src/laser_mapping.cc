#include <tf/transform_broadcaster.h>
#include <yaml-cpp/yaml.h>
#include <algorithm>
#include <execution>
#include <fstream>
#include <chrono>

#include "laser_mapping.h"
#include "utils.h"

#include <tbb/parallel_for.h>

#include <opencv2/opencv.hpp>

namespace akf_lio
{

    bool LaserMapping::InitROS(ros::NodeHandle &nh)
    {
        LoadParams(nh);
        SubAndPubToROS(nh);

        // localmap init (after LoadParams)
        ivox_ = std::make_shared<IVoxType>(ivox_options_);

        // esekf init
        std::vector<double> epsi(23, 0.001);
        kf_.init_dyn_share(
            get_f, df_dx, df_dw,
            [this](state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data)
            { ObsModel(s, ekfom_data); },
            options::NUM_MAX_ITERATIONS, epsi.data());

        return true;
    }

    bool LaserMapping::LoadParams(ros::NodeHandle &nh)
    {
        // get params from param server
        int lidar_type, ivox_nearby_type;
        double gyr_cov, acc_cov, b_gyr_cov, b_acc_cov;
        common::V3D lidar_T_wrt_IMU;
        common::M3D lidar_R_wrt_IMU;

        nh.param<double>("akf_update_alpha", p_imu_->alpha, 1);

        nh.param<double>("preprocess/blind", preprocess_->Blind(), 0.01);
        nh.param<float>("preprocess/time_scale", preprocess_->TimeScale(), 1e-3);
        nh.param<int>("preprocess/lidar_type", lidar_type, 1);
        nh.param<int>("preprocess/scan_line", preprocess_->NumScans(), 16);

        nh.param<float>("mapping/lidar_cov", options::LIDAR_COV, 0.001);
        nh.param<double>("mapping/gyr_cov", gyr_cov, 0.1);
        nh.param<double>("mapping/acc_cov", acc_cov, 0.1);
        nh.param<double>("mapping/b_gyr_cov", b_gyr_cov, 0.0001);
        nh.param<double>("mapping/b_acc_cov", b_acc_cov, 0.0001);
        nh.param<std::vector<double>>("mapping/extrinsic_T", extrinT_, std::vector<double>());
        nh.param<std::vector<double>>("mapping/extrinsic_R", extrinR_, std::vector<double>());

        nh.param<bool>("publish/path_publish_en", path_pub_en_, true);
        nh.param<bool>("publish/scan_reg_pub_en", scan_reg_pub_en_, false);
        nh.param<bool>("publish/dense_publish_en", dense_pub_en_, false);
        nh.param<bool>("publish/map_publish_en", map_pub_en_, false);
        nh.param<bool>("publish/gaussian_publish_en", gaussian_publish_en_, false);
        nh.param<double>("publish/gaussian_pub_dis", gaussian_pub_dis_, 10);
        nh.param<double>("publish/gaussian_pub_min_cnt", gaussian_pub_min_cnt_, 5);

        nh.param<bool>("adap_voxel_size_en", adap_voxel_size_en_, true);
        nh.param<int>("target_point_size", target_point_size_, 2000);
        nh.param<int>("point_filter_num", preprocess_->PointFilterNum(), 2);
        nh.param<int>("max_iteration", options::NUM_MAX_ITERATIONS, 4);
        nh.param<double>("init_uncertainty", init_uncertainty_, 0.01);
        nh.param<double>("t_ratio_b", t_ratio_b_, 0);
        nh.param<double>("ivox_grid_resolution", ivox_options_.resolution_, 0.5);
        nh.param<int>("ivox_nearby_type", ivox_nearby_type, 26);
        nh.param<float>("t_mal", options::T_MAL, 11.28);
        nh.param<double>("t_stop_pseudo_merge", t_stop_pseudo_merge_, 11.28);
        nh.param<double>("time_to_delete_local_map", options::TIME_TO_DELETE_LOCAL_MAP, 1000);

        nh.param<bool>("runtime_pos_log_enable", runtime_pos_log_, true);

        LOG(INFO) << "lidar_type " << lidar_type;
        if (lidar_type == 1)
        {
            preprocess_->SetLidarType(LidarType::AVIA);
            LOG(INFO) << "Using AVIA Lidar";
        }
        else if (lidar_type == 2)
        {
            preprocess_->SetLidarType(LidarType::VELO32);
            LOG(INFO) << "Using Velodyne 32 Lidar";
        }
        else if (lidar_type == 3)
        {
            preprocess_->SetLidarType(LidarType::OUST64);
            LOG(INFO) << "Using OUST 64 Lidar";
        }
        else
        {
            LOG(WARNING) << "unknown lidar_type";
            return false;
        }

        if (ivox_nearby_type == 0)
        {
            ivox_options_.nearby_type_ = IVoxType::NearbyType::CENTER;
        }
        else if (ivox_nearby_type == 6)
        {
            ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY6;
        }
        else if (ivox_nearby_type == 18)
        {
            ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY18;
        }
        else if (ivox_nearby_type == 26)
        {
            ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY26;
        }
        else
        {
            LOG(WARNING) << "unknown ivox_nearby_type, use NEARBY18";
            ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY18;
        }

        path_.header.stamp = ros::Time::now();
        path_.header.frame_id = "camera_init";
        gt_path_.header.stamp = ros::Time::now();
        gt_path_.header.frame_id = "camera_init";

        fout_pre.open((std::string(std::string(ROOT_DIR) + "Log/" + "mat_pre.txt")), std::ios::out);
        fout_out.open((std::string(std::string(ROOT_DIR) + "Log/" + "mat_out.txt")), std::ios::out);
        if (fout_pre && fout_out)
            std::cout << "~~~~" << ROOT_DIR << " file opened" << std::endl;
        else
            std::cout << "~~~~" << ROOT_DIR << " doesn't exist" << std::endl;

        lidar_T_wrt_IMU = common::VecFromArray<double>(extrinT_);
        lidar_R_wrt_IMU = common::MatFromArray<double>(extrinR_);

        p_imu_->SetExtrinsic(lidar_T_wrt_IMU, lidar_R_wrt_IMU);
        p_imu_->SetGyrCov(common::V3D(gyr_cov, gyr_cov, gyr_cov));
        p_imu_->SetAccCov(common::V3D(acc_cov, acc_cov, acc_cov));
        p_imu_->SetGyrBiasCov(common::V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov));
        p_imu_->SetAccBiasCov(common::V3D(b_acc_cov, b_acc_cov, b_acc_cov));

        return true;
    }

    void LaserMapping::SubAndPubToROS(ros::NodeHandle &nh)
    {
        // ROS subscribe initialization
        std::string lidar_topic, imu_topic;
        nh.param<std::string>("common/lid_topic", lidar_topic, "/livox/lidar");
        nh.param<std::string>("common/imu_topic", imu_topic, "/livox/imu");

        if (preprocess_->GetLidarType() == LidarType::AVIA)
        {
            sub_pcl_ = nh.subscribe<livox_ros_driver::CustomMsg>(
                lidar_topic, 200000, [this](const livox_ros_driver::CustomMsg::ConstPtr &msg)
                { LivoxPCLCallBack(msg); });
        }
        else
        {
            sub_pcl_ = nh.subscribe<sensor_msgs::PointCloud2>(
                lidar_topic, 200000, [this](const sensor_msgs::PointCloud2::ConstPtr &msg)
                { StandardPCLCallBack(msg); });
        }

        sub_imu_ = nh.subscribe<sensor_msgs::Imu>(imu_topic, 200000,
                                                  [this](const sensor_msgs::Imu::ConstPtr &msg)
                                                  { IMUCallBack(msg); });

        // ROS publisher init
        path_.header.stamp = ros::Time::now();
        path_.header.frame_id = "camera_init";

        pub_laser_cloud_world_ = nh.advertise<sensor_msgs::PointCloud2>("/cloud_dense_world", 100000);
        pub_laser_cloud_reg_world_ = nh.advertise<sensor_msgs::PointCloud2>("/cloud_reg_world", 100000);
        pub_odom_aft_mapped_ = nh.advertise<nav_msgs::Odometry>("/Odometry", 100000);
        pub_path_ = nh.advertise<nav_msgs::Path>("/path", 100000);
        gt_pub_path_ = nh.advertise<nav_msgs::Path>("/gt_path", 100000);
        point_cov_pub = nh.advertise<visualization_msgs::MarkerArray>("/scan_gaussian_reg", 10000);
        pub_map = nh.advertise<sensor_msgs::PointCloud2>("/Laser_map", 100000);
        map_cov_pub = nh.advertise<visualization_msgs::MarkerArray>("/map_gaussian", 10000);
    }

    LaserMapping::LaserMapping()
    {
        preprocess_.reset(new PointCloudPreprocess());
        p_imu_.reset(new ImuProcess());
    }

    /**
     * [功能描述]：将两个点合并为一个新点，通过加权平均的方式融合位置、强度、协方差等信息
     * @param pt1：第一个要合并的点
     * @param pt2：第二个要合并的点
     * @return 合并后的新点，包含融合后的所有属性信息
     * 
     * 该函数实现智能点合并，考虑每个点的观测次数作为权重，
     * 同时正确更新协方差矩阵以保持统计一致性
     */
    PointType LaserMapping::Merge2(const PointType &pt1, const PointType &pt2)
    {
        // 以pt2为基础创建合并后的点
        PointType pt3 = pt2;
        
        // 将点的3D坐标转换为Eigen向量格式，便于数学运算
        common::V3D p1(pt1.x, pt1.y, pt1.z), p2(pt2.x, pt2.y, pt2.z), p3;
        
        // 根据两个点的观测次数计算加权比例
        // pt_num表示该点被观测到的次数，次数越多权重越大
        double ratio_1 = (double)pt1.pt_num / (double)(pt1.pt_num + pt2.pt_num);  // pt1的权重比例
        double ratio_2 = (double)pt2.pt_num / (double)(pt1.pt_num + pt2.pt_num);  // pt2的权重比例

        // 计算加权平均位置
        p3 = p1 * ratio_1 + p2 * ratio_2;
        
        // 将计算结果赋值给合并后点的坐标
        pt3.x = p3(0);
        pt3.y = p3(1);
        pt3.z = p3(2);
        
        // 计算加权平均强度
        pt3.intensity = (pt1.intensity * ratio_1 + pt2.intensity * ratio_2);
        
        // 更新协方差矩阵（考虑两个高斯分布的合并）
        // 使用公式：Cov_new = w1*(Cov1 + μ1*μ1^T) + w2*(Cov2 + μ2*μ2^T) - μ_new*μ_new^T
        // 这确保了合并后协方差矩阵的统计正确性
        pt3.cov =
            ratio_1 * (pt1.cov + p1 * p1.transpose()) + ratio_2 * (pt2.cov + p2 * p2.transpose()) - p3 * p3.transpose();

        // 累加观测次数
        pt3.pt_num = pt1.pt_num + pt2.pt_num;
        
        // 使用较新的时间戳（取两个点时间戳的最大值）
        pt3.time = pt1.time > pt2.time ? pt1.time : pt2.time;

        // 限制最大特征数量，防止数值溢出
        if (pt3.pt_num > options::MAX_FEA_NUM)
            pt3.pt_num = options::MAX_FEA_NUM;

        // 处理使用次数和不确定性信息的合并
        int use_sum = pt1.use_num + pt2.use_num;  // 总使用次数
        
        // 只有当总使用次数不为0时才进行不确定性合并
        if (use_sum != 0)
        {
            // 根据使用次数计算新的权重比例
            double ratio_3 = (double)pt1.use_num / (double)(use_sum);  // pt1使用次数权重
            double ratio_4 = (double)pt2.use_num / (double)(use_sum);  // pt2使用次数权重
            
            // 基于使用次数的加权平均位置（用于不确定性计算）
            common::V3D p4 = ratio_3 * p1 + ratio_4 * p2;
            
            // 计算加权平均不确定性
            pt3.uncertainty = (ratio_3 * pt1.uncertainty + ratio_4 * pt2.uncertainty);
            
            // 累加使用次数
            pt3.use_num = use_sum;
            
            // 限制最大使用次数，防止数值溢出
            if (pt3.use_num > options::MAX_FEA_NUM)
                pt3.use_num = options::MAX_FEA_NUM;
        }

        // 返回合并后的点
        return pt3;
    }

    /**
     * [功能描述]：激光雷达建图主处理函数，执行完整的SLAM处理流程
     * 包括IMU处理、卡尔曼滤波预测/更新、点云处理、增量建图等核心功能
     * @return 无返回值
     */
    void LaserMapping::Run()
    {
        // 记录当前帧处理开始时间，用于计算帧处理耗时
        auto frame_start_time = std::chrono::high_resolution_clock::now();
        
        // 同步传感器数据包（IMU和激光雷达数据）
        if (!SyncPackages())
        {
            return; // 数据同步失败，跳过当前帧
        }

        // 初始化第一帧激光雷达时间戳
        if (first_lidar_time_ < 1e-6)
            first_lidar_time_ = measures_.lidar_bag_time_;
        
        // 运行时位置日志输出
        if (runtime_pos_log_)
            LOG(INFO) << "measures_.lidar_bag_time_ - first_lidar_time_ " << measures_.lidar_bag_time_ - first_lidar_time_;
        
        // 初始化欧拉角和状态变量
        common::V3D euler_cur, ext_euler, print_status; // 当前欧拉角、外参欧拉角、打印状态
        ext_euler.setZero();    // 外参欧拉角置零
        print_status.setZero(); // 打印状态置零

        /// ==== IMU处理、卡尔曼滤波预测、点云去畸变 ====
        p_imu_->Process(measures_, kf_, scan_undistort_);
        pred_cov_ = kf_.get_P().block<6, 6>(0, 0); // 提取预测协方差矩阵的位置和姿态部分
        
        // 检查去畸变后的点云是否为空
        if (scan_undistort_->empty() || (scan_undistort_ == nullptr))
        {
            PublishOdometry(pub_odom_aft_mapped_); // 发布里程计信息
            if (path_pub_en_)
                PublishPath(pub_path_); // 如果启用路径发布，发布路径
            LOG(WARNING) << "No point, skip this scan!"; // 警告：无点云数据
            return;
        }

        // 判断EKF是否已初始化（基于时间阈值）
        flg_EKF_inited_ = (measures_.lidar_bag_time_ - first_lidar_time_) >= options::INIT_TIME;

        /// ==== 点云降采样处理 ====
        Timer::Evaluate([&, this]()
                        { VoxelGridDownsample(scan_undistort_, scan_down_reg_); },
                        "Downsample PointCloud");

        /// ==== 第一帧特殊处理 ====
        if (flg_first_scan_)
        {
            PointVector points_to_add; // 要添加到地图的点云

            state_point_ = kf_.get_x();   // 获取当前状态估计
            state_cov_.setZero();         // 状态协方差置零
            int cur_pts = cur_all_points.size(); // 当前点云数量
            points_to_add.reserve(cur_pts);       // 预分配内存
            PointType point_world;        // 世界坐标系下的点

            // 重力校准
            common::V3D ez(0, 0, -1), gz(state_point_.grav);
            Eigen::Quaterniond G_q_I0 = Eigen::Quaterniond::FromTwoVectors(gz, ez);
            common::M3D G_R_I0 = G_q_I0.toRotationMatrix();

            state_point_.pos = G_R_I0 * state_point_.pos;
            state_point_.rot = G_R_I0 * state_point_.rot;
            state_point_.vel = G_R_I0 * state_point_.vel;
            state_point_.grav = G_R_I0 * state_point_.grav;
            
            // 将所有点从机体坐标系转换到世界坐标系
            for (int i = 0; i < cur_pts; i++)
            {
                PointBodyToWorld(&(cur_all_points.at(i)), &point_world);
                points_to_add.emplace_back(point_world);
            }
            
            // 发布第一帧的结果
            PublishOdometry(pub_odom_aft_mapped_); // 发布里程计
            if (path_pub_en_)
                PublishPath(pub_path_);            // 发布路径
            if (dense_pub_en_)
                PublishFrameWorld();               // 发布稠密点云

            ivox_->AddPoints(points_to_add);       // 将点云添加到iVox地图中
            first_lidar_time_ = measures_.lidar_bag_time_; // 更新第一帧时间
            flg_first_scan_ = false;               // 标记第一帧处理完成
            return;
        }

        /// ==== 常规帧处理：点数检查和数据结构初始化 ====
        size_t cur_pts = scan_down_reg_.size(); // 降采样后的点云数量
        if (cur_pts < 5) // 点数太少，跳过处理
        {
            PublishOdometry(pub_odom_aft_mapped_);
            if (path_pub_en_)
                PublishPath(pub_path_);
            LOG(WARNING) << "Too few points, skip this scan!" << scan_undistort_->size() << ", " << scan_down_reg_.size();
            return;
        }
        
        // 初始化各种数据结构的大小
        scan_down_world_.resize(cur_pts);        // 世界坐标系下的降采样点云
        nearest_points_.clear();                  // 清空最近邻点
        nearest_points_.resize(cur_pts);          // 重新分配最近邻点存储空间
        residuals_.resize(cur_pts);               // 残差向量
        residuals_.assign(cur_pts, 0);            // 残差初始化为0
        point_selected_surf_.resize(cur_pts);     // 选中的表面点标志
        plane_coef_.clear();                      // 清空平面系数
        plane_coef_.resize(cur_pts);              // 重新分配平面系数存储空间

        /// ==== 状态更新和日志记录 ====
        state_point_ = kf_.get_x();                    // 获取当前状态估计
        euler_cur = SO3ToEuler(state_point_.rot);      // 将旋转矩阵转换为欧拉角

        // 设置打印状态信息
        print_status(0) = avg_voxel_size_;                              // 平均体素大小
        print_status(1) = double(cur_pts) / double(target_point_size_); // 点云数量比例

        // 提取外参的欧拉角
        ext_euler(0) = kf_.G_cur(3); // roll
        ext_euler(1) = kf_.G_cur(4); // pitch  
        ext_euler(2) = kf_.G_cur(5); // yaw

        // 输出预处理后的状态信息到日志文件
        fout_pre << std::setw(20) << measures_.lidar_bag_time_ - first_lidar_time_ << " " << euler_cur.transpose() << " "
                << state_point_.pos.transpose() << " " << p_imu_->est_noise_.block<3, 3>(0, 0).diagonal().transpose()
                << " " << p_imu_->est_noise_.block<3, 3>(3, 0).diagonal().transpose() << " " << ext_euler.transpose()
                << " " << state_point_.vel.transpose() << " " << state_point_.bg.transpose() << " "
                << state_point_.ba.transpose() << " " << state_point_.grav << " " << print_status.transpose() << std::endl;

        /// ==== 迭代卡尔曼滤波更新（ICP + IEKF） ====
        Timer::Evaluate(
            [&, this]()
            {
                // 迭代状态估计
                double solve_H_time = 0;
                // 更新观测模型，包括最近邻搜索和点到平面残差计算
                kf_.update_iterated_dyn_share_akf(solve_H_time);
                
                // 保存更新后的状态
                state_point_ = kf_.get_x();                                      // 更新后的状态
                state_cov_ = kf_.get_P().block<6, 6>(0, 0);                     // 更新后的协方差矩阵
                euler_cur_ = SO3ToEuler(state_point_.rot);                      // 更新后的欧拉角
                pos_lidar_ = state_point_.pos + state_point_.rot * state_point_.offset_T_L_I; // 激光雷达在世界坐标系下的位置

                // 更新当前欧拉角和外参
                euler_cur = SO3ToEuler(state_point_.rot);
                ext_euler(0) = kf_.G_cur(0); // 更新外参roll
                ext_euler(1) = kf_.G_cur(1); // 更新外参pitch
                ext_euler(2) = kf_.G_cur(2); // 更新外参yaw
            },
            "IEKF Solve and Update");

        /// ==== 增量建图更新 ====
        Timer::Evaluate([&, this]()
                        { MapIncremental(); }, "    Incremental Mapping");

        /// ==== 性能统计和日志输出 ====
        auto frame_end_time = std::chrono::high_resolution_clock::now();
        auto frame_time = std::chrono::duration_cast<std::chrono::duration<double>>(frame_end_time - frame_start_time).count() * 1000;
        LOG(INFO) << "[Frame Time] Frame " << frame_num_ << " processing time: " << frame_time << " ms";

        LOG(INFO) << "[ mapping ]: In num: " << scan_undistort_->points.size() << " downsamp " << cur_pts
                << " Map grid num: " << ivox_->NumValidGrids() << " effect num : " << effect_feat_num_;

        /// ==== 结果发布 ====
        PublishOdometry(pub_odom_aft_mapped_); // 发布建图后的里程计

        if (path_pub_en_)
            PublishPath(pub_path_); // 发布路径轨迹

        if (dense_pub_en_)
            PublishFrameWorld(); // 发布稠密点云

        if (scan_reg_pub_en_)
            PublishFrameRegWorld(pub_laser_cloud_reg_world_); // 发布配准后的点云

        if (map_pub_en_ && frame_num_ % 10 == 0) // 每10帧发布一次地图
        {
            PublishMap();
        }

        // 计算有效特征点比例
        print_status(2) = double(effect_feat_num_) / double(cur_pts);

        // 输出最终处理结果到日志文件
        fout_out << std::setw(20) << measures_.lidar_bag_time_ - first_lidar_time_ << " " << euler_cur.transpose() << " "
                << state_point_.pos.transpose() << " " << p_imu_->est_noise_.block<3, 3>(6, 0).diagonal().transpose()
                << " " << p_imu_->est_noise_.block<3, 3>(9, 0).diagonal().transpose() << " " << ext_euler.transpose()
                << " " << state_point_.vel.transpose() << " " << state_point_.bg.transpose() << " "
                << state_point_.ba.transpose() << " " << state_point_.grav << " " << print_status.transpose() << std::endl;
        
        frame_num_++; // 帧计数器递增
    }

    /**
     * [功能描述]：体素网格降采样函数，将输入点云按体素网格进行降采样以减少点云密度
     * @param cloud_in：输入的原始点云指针
     * @param cloud_down_reg：输出的降采样后点云容器
     * @return 无返回值
     */
    void LaserMapping::VoxelGridDownsample(PointCloudType::Ptr &cloud_in, PointVector &cloud_down_reg)
    {
        size_t cloud_in_size = cloud_in->size(); // 获取输入点云大小

        common::V3D loc_xyz;                      // 点的位置坐标
        int64_t ijk0, ijk1, ijk2, idx;           // 体素网格索引坐标

        // 清空并预分配输出点云容器
        cloud_down_reg.clear();
        cloud_down_reg.reserve(cloud_in_size);

        // 清空并重新分配当前所有点的存储容器
        cur_all_points.clear();
        cur_all_points.resize(cloud_in_size);

        // 创建点索引数组，用于并行处理
        std::vector<size_t> index0(cloud_in_size);

        common::V3D p3d; // 3D点坐标

        // 初始化索引数组
        for (size_t i = 0; i < cloud_in_size; ++i)
        {
            index0[i] = i;
        }
        
        // 并行处理所有输入点，设置点的基本属性
        std::for_each(std::execution::par_unseq, index0.begin(), index0.end(), [&](const size_t &i)
                    {
            PointType &tp = cur_all_points[i];    // 当前处理的点
            common::V3D pt3d;                     // 点的3D坐标
            common::M3D body_cov;                 // 机体坐标系协方差（未使用）
            
            // 复制点的基本坐标信息
            tp.x = cloud_in->points[i].x;
            tp.y = cloud_in->points[i].y;
            tp.z = cloud_in->points[i].z;

            tp.intensity = cloud_in->points[i].intensity;  // 复制强度信息
            tp.pt_num = 1;                                 // 点数量设为1
            tp.use_num = 1;                                // 使用次数设为1
            // 计算点的相对时间戳（相对于第一帧激光雷达时间）
            tp.time = measures_.lidar_bag_time_ - first_lidar_time_ + cloud_in->points[i].curvature / 1000;
            pt3d << cloud_in->points[i].x, cloud_in->points[i].y, cloud_in->points[i].z;
            tp.cov = options::LIDAR_COV * Eigen::Matrix3d::Identity(); // 设置协方差矩阵
            tp.uncertainty = init_uncertainty_;                        // 设置初始不确定性
        });

        /// ==== 自适应体素大小计算 ====
        // 如果启用自适应体素大小且目标点数大于等于输入点数，使用最小体素大小
        if (adap_voxel_size_en_ && target_point_size_ >= cloud_in_size)
        {
            scan_voxel_size_ = min_voxel_size_;
        }
        else
        {
            if (adap_voxel_size_en_) // 启用自适应体素大小
            {
                int max_sample_cnt = 10; // 最大迭代次数

                // 如果存在历史体素大小记录，计算平均值作为初始值
                if (!voxel_size_sliding_.empty())
                {
                    scan_voxel_size_ = 0;
                    for (auto &it : voxel_size_sliding_)
                    {
                        scan_voxel_size_ += it;
                    }
                    scan_voxel_size_ /= voxel_size_sliding_.size(); // 计算平均体素大小
                    scan_voxel_size_ = round(scan_voxel_size_ / 0.001f) * 0.001f; // 四舍五入到0.001精度
                }

                // 迭代优化体素大小，使降采样后的点数接近目标点数
                for (int iter = 0; iter < max_sample_cnt; iter++)
                {
                    cur_iter = iter; // 记录当前迭代次数
                    std::unordered_map<size_t, size_t> sample_volume_hash; // 体素哈希表，记录每个体素的点数
                    sample_volume_hash.clear();
                    sample_volume_hash.reserve(cloud_in_size / 2);

                    // 为每个点计算体素哈希值
                    std::vector<size_t> hash_values(cloud_in_size);
                    std::for_each(std::execution::par_unseq, index0.begin(), index0.end(), [&](const size_t &i)
                                {
                        p3d << cloud_in->points[i].x, cloud_in->points[i].y, cloud_in->points[i].z;
                        // 将点坐标转换为体素网格坐标
                        for (int j = 0; j < 3; j++) {
                            p3d[j] = p3d[j] / scan_voxel_size_;
                        }
                        // 计算体素网格的整数索引
                        ijk0 = int64_t(floor(p3d[0]));
                        ijk1 = int64_t(floor(p3d[1]));
                        ijk2 = int64_t(floor(p3d[2]));
                        hash_values[i] = ComputeVoxelHash(ijk0, ijk1, ijk2); // 计算体素哈希值
                    });

                    // 统计每个体素中的点数
                    for (size_t i = 0; i < cloud_in_size; ++i)
                    {
                        sample_volume_hash[hash_values[i]]++;
                    }

                    // 计算当前体素大小下的采样比例
                    double sample_target_ratio = double(sample_volume_hash.size()) / double(target_point_size_);

                    // 如果采样比例接近1.0（误差小于10%），说明体素大小合适，退出迭代
                    if (fabs(sample_target_ratio - 1.0f) < 0.1f)
                    {
                        break;
                    }
                    else
                    {
                        // 根据采样比例调整体素大小
                        double last_scan_voxel_size_ = scan_voxel_size_;
                        scan_voxel_size_ = (sample_target_ratio / 2.0f + 0.5f) * scan_voxel_size_;
                        scan_voxel_size_ = round(scan_voxel_size_ / 0.001f) * 0.001f; // 四舍五入精度
                        
                        // 如果体素大小变化很小，退出迭代
                        if (fabs(last_scan_voxel_size_ - scan_voxel_size_) < 0.001f)
                            break;
                    }
                }
                // 更新平均体素大小并维护滑动窗口
                avg_voxel_size_ = scan_voxel_size_;
                voxel_size_sliding_.push_back(scan_voxel_size_);
                if (voxel_size_sliding_.size() > 3) // 保持滑动窗口大小为3
                    voxel_size_sliding_.pop_front();
            }
        }

        // 确保体素大小不小于最小值
        if (scan_voxel_size_ < min_voxel_size_)
            scan_voxel_size_ = min_voxel_size_;

        /// ==== 执行体素网格降采样 ====
        std::unordered_map<size_t, PointVector> leaves_; // 体素叶子节点，存储每个体素中的点
        leaves_.clear();
        leaves_.reserve(cloud_in_size / 2);

        // 计算最终的体素哈希值
        std::vector<size_t> final_hash_values(cloud_in_size);
        std::for_each(std::execution::par_unseq, index0.begin(), index0.end(), [&](const size_t &i)
                    {
            loc_xyz << cur_all_points[i].x, cur_all_points[i].y, cur_all_points[i].z;
            // 将点坐标转换为体素网格坐标
            for (int j = 0; j < 3; j++) {
                loc_xyz[j] = loc_xyz[j] / scan_voxel_size_;
            }
            // 计算体素网格的整数索引
            ijk0 = int64_t(floor(loc_xyz[0]));
            ijk1 = int64_t(floor(loc_xyz[1]));
            ijk2 = int64_t(floor(loc_xyz[2]));
            final_hash_values[i] = ComputeVoxelHash(ijk0, ijk1, ijk2); // 计算最终哈希值
        });

        // 将点分配到对应的体素中
        for (size_t cp = 0; cp < cloud_in_size; cp++)
        {
            auto &voxel = leaves_[final_hash_values[cp]];
            if (voxel.empty())
            {
                voxel.reserve(4); // 预分配空间，一般一个体素不会有太多点
            }
            voxel.push_back(cur_all_points[cp]); // 将点添加到对应体素
        }

        // 从每个体素中选择一个点作为代表点（这里选择第一个点）
        cloud_down_reg.reserve(leaves_.size());
        for (const auto &leaf : leaves_)
        {
            cloud_down_reg.push_back(leaf.second.front()); // 每个体素只保留第一个点
        }

        return;
    }

    bool LaserMapping::time_list(const PointType2 &x, const PointType2 &y) { return (x.curvature < y.curvature); };

    void LaserMapping::StandardPCLCallBack(const sensor_msgs::PointCloud2::ConstPtr &msg)
    {
        mtx_buffer_.lock();
        Timer::Evaluate(
            [&, this]()
            {
                scan_count_++;
                if (msg->header.stamp.toSec() < last_timestamp_lidar_)
                {
                    LOG(ERROR) << "lidar loop back, clear buffer";
                    lidar_buffer_.clear();
                }

                PointCloudType::Ptr ptr(new PointCloudType());
                preprocess_->Process(msg, ptr);
                lidar_buffer_.push_back(ptr);
                time_buffer_.push_back(msg->header.stamp.toSec());
                last_timestamp_lidar_ = msg->header.stamp.toSec();
            },
            "Preprocess (Standard)");
        mtx_buffer_.unlock();
    }

    void LaserMapping::LivoxPCLCallBack(const livox_ros_driver::CustomMsg::ConstPtr &msg)
    {
        mtx_buffer_.lock();
        Timer::Evaluate(
            [&, this]()
            {
                scan_count_++;
                if (msg->header.stamp.toSec() < last_timestamp_lidar_)
                {
                    LOG(WARNING) << "lidar loop back, clear buffer";
                    lidar_buffer_.clear();
                }

                last_timestamp_lidar_ = msg->header.stamp.toSec();

                PointCloudType::Ptr ptr(new PointCloudType());
                preprocess_->Process(msg, ptr);
                lidar_buffer_.emplace_back(ptr);
                time_buffer_.emplace_back(last_timestamp_lidar_);
            },
            "Preprocess (Livox)");

        mtx_buffer_.unlock();
    }

    void LaserMapping::IMUCallBack(const sensor_msgs::Imu::ConstPtr &msg_in)
    {
        publish_count_++;
        sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));

        msg->header.stamp = ros::Time().fromSec(msg_in->header.stamp.toSec());

        double timestamp = msg->header.stamp.toSec();

        if (true)
        {
            // imu 角速度 加速度矫正
            msg->angular_velocity.x = msg_in->angular_velocity.y;
            msg->angular_velocity.y = msg_in->angular_velocity.x;
            msg->linear_acceleration.x = msg_in->linear_acceleration.y;
            msg->linear_acceleration.y = msg_in->linear_acceleration.x;
        }

        mtx_buffer_.lock();
        if (timestamp < last_timestamp_imu_)
        {
            LOG(WARNING) << "imu loop back, clear buffer";
            imu_buffer_.clear();
        }

        last_timestamp_imu_ = timestamp;
        imu_buffer_.emplace_back(msg);
        mtx_buffer_.unlock();
    }

    bool LaserMapping::SyncPackages()
    {
        if (lidar_buffer_.empty() || (imu_buffer_.empty()))
        {
            return false;
        }

        /*** push a lidar scan ***/
        if (!lidar_pushed_)
        {
            measures_.lidar_ = lidar_buffer_.front();
            measures_.lidar_bag_time_ = time_buffer_.front();

            if (measures_.lidar_->points.size() <= 1)
            {
                LOG(WARNING) << "Too few input point cloud!";
                lidar_end_time_ = measures_.lidar_bag_time_ + lidar_mean_scantime_;
            }
            else if (measures_.lidar_->points.back().curvature / double(1000) < 0.5 * lidar_mean_scantime_)
            {
                lidar_end_time_ = measures_.lidar_bag_time_ + lidar_mean_scantime_;
            }
            else
            {
                scan_num_++;
                lidar_end_time_ = measures_.lidar_bag_time_ + measures_.lidar_->points.back().curvature / double(1000);
                lidar_mean_scantime_ +=
                    (measures_.lidar_->points.back().curvature / double(1000) - lidar_mean_scantime_) / scan_num_;
            }

            measures_.lidar_end_time_ = lidar_end_time_;
            lidar_pushed_ = true;
        }

        if (last_timestamp_imu_ < lidar_end_time_)
        {
            return false;
        }

        /*** push imu_ data, and pop from imu_ buffer ***/

        double imu_time = imu_buffer_.front()->header.stamp.toSec();
        measures_.imu_.clear();
        while ((!imu_buffer_.empty()) && (imu_time < lidar_end_time_))
        {
            imu_time = imu_buffer_.front()->header.stamp.toSec();
            if (imu_time > lidar_end_time_)
                break;
            measures_.imu_.push_back(imu_buffer_.front());
            imu_buffer_.pop_front();
        }

        lidar_buffer_.pop_front();
        time_buffer_.pop_front();
        lidar_pushed_ = false;
        return true;
    }

    /**
     * [功能描述]：增量建图函数，将当前帧的点云数据添加到全局地图中，并更新相关点的不确定性
     * 包括坐标转换、最近邻点更新、地图点添加和地图维护等功能
     * @return 无返回值
     */
    void LaserMapping::MapIncremental()
    {
        PointVector points_to_add;    // 待添加到地图的新点云
        PointVector points_to_update; // 待更新不确定性的现有地图点

        // 获取当前卡尔曼滤波器估计的状态（位姿等信息）
        state_point_ = kf_.get_x();
        int cur_pts = scan_down_reg_.size(); // 当前降采样后点云的数量
        
        /// ==== 坐标系转换：机体坐标系 -> 世界坐标系 ====
        points_to_add.clear();
        points_to_add.resize(cur_pts); // 预分配空间
        
        // 创建索引数组用于并行处理
        std::vector<size_t> index(cur_pts);
        for (size_t i = 0; i < cur_pts; i++)
        {
            index[i] = i;
        }
        
        // 并行处理所有点的坐标转换
        std::for_each(std::execution::par_unseq, index.begin(), index.end(), [&](const size_t &i)
                    {
            PointType &point_body = scan_down_reg_.at(i);  // 机体坐标系下的点
            PointType &point_world = points_to_add.at(i);  // 转换后的世界坐标系点
            PointBodyToWorld(&point_body, &point_world);   // 执行坐标转换
            point_world.uncertainty = init_uncertainty_;   // 设置初始不确定性
        });

        /// ==== 处理最近邻点的不确定性更新 ====
        points_to_update.clear();
        points_to_update.reserve(cur_pts * options::MIN_NUM_MATCH_POINTS); // 预分配空间

        // 遍历所有降采样后的点
        for (size_t i = 0; i < cur_pts; i++)
        {
            auto &points_near = nearest_points_[i];         // 当前点的最近邻点集合
            PointType &point_world = points_to_add.at(i);  // 当前点在世界坐标系下的位置
            
            // 如果存在最近邻点且该点被选为表面特征点
            if (!points_near.empty() && point_selected_surf_[i])
            {
                int j = points_near.size(); // 最近邻点的数量
                
                // 遍历所有最近邻点，更新它们的属性
                for (int k = 0; k < j; k++)
                {
                    auto &p_update = points_near.at(k);
                    p_update.uncertainty = point_world.uncertainty; // 用新点的不确定性更新最近邻点
                    p_update.pt_num = 0;                            // 重置点数量
                    p_update.use_num = 1;                           // 设置使用次数为1
                    points_to_update.emplace_back(p_update);        // 添加到待更新列表
                }
            }
        }

        /// ==== 地图更新操作 ====
        // 更新现有地图点的不确定性信息
        ivox_->UpdateUncertainty(points_to_update);

        // 将新的点云添加到iVox地图结构中
        ivox_->AddPoints(points_to_add);
        
        // 擦除距离当前位置较远或时间较老的地图点，维护地图大小
        // state_point_.pos: 当前位置
        // lidar_end_time_ - first_lidar_time_: 相对于第一帧的时间差
        ivox_->ErasePoints(state_point_.pos, lidar_end_time_ - first_lidar_time_);
    }

    /**
     * Lidar point cloud registration
     * will be called by the eskf custom observation model
     * compute point-to-plane residual here
     * @param s kf state
     * @param ekfom_data H matrix
     */
    /**
     * [功能描述]：观测模型函数，构建激光雷达点云的观测方程，用于迭代卡尔曼滤波更新
     * 包括最近邻搜索、平面拟合、残差计算和雅可比矩阵构建等核心功能
     * @param s：当前状态估计（位姿、速度等）
     * @param ekfom_data：EKF数据结构，存储观测方程的系数矩阵和残差
     * @return 无返回值
     */
    void LaserMapping::ObsModel(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data)
    {
        int cnt_pts = scan_down_reg_.size(); // 降采样后点云数量
        
        // 创建索引数组用于并行处理
        std::vector<size_t> index(cnt_pts);
        for (size_t i = 0; i < cnt_pts; i++)
        {
            index[i] = i;
        }
        
        // 获取当前状态估计和协方差矩阵
        state_point_ = kf_.get_x();
        state_cov_ = kf_.get_P().block<6, 6>(0, 0);
        
        /// ==== 激光雷达匹配处理 ====
        Timer::Evaluate(
            [&, this]()
            {
                // 计算激光雷达在世界坐标系下的旋转和平移
                auto R_wl = (s.rot * s.offset_R_L_I);           // 世界到激光雷达的旋转矩阵
                auto t_wl = (s.rot * s.offset_T_L_I + s.pos);   // 世界到激光雷达的平移向量

                {
                    // 并行处理每个点的最近邻搜索和平面拟合
                    std::for_each(std::execution::par_unseq, index.begin(), index.end(), [&](const size_t &i)
                                {
                    PointType &point_body = scan_down_reg_.at(i);   // 机体坐标系下的点
                    PointType &point_world = scan_down_world_.at(i); // 世界坐标系下的点

                    /* 转换到世界坐标系 */
                    common::V3D p_body(point_body.x, point_body.y, point_body.z);
                    PointBodyToWorld(&point_body, &point_world);
                    
                    // 在地图中搜索最近邻点
                    {
                        auto &points_near = nearest_points_[i];
                        /** 在地图中寻找最近的表面点 **/
                        ivox_->GetClosestPoint(point_world, points_near, options::NUM_MATCH_POINTS, 3);

                        auto &p_sum = plane_coef_[i]; // 合并后的平面系数点
                        if (!points_near.empty()) {
                            PointType p1;
                            common::V3D incident_normal;    // 入射法向量
                            common::M3D project_2D_cov;    // 2D投影协方差
                            Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 3, 3>> es_2D;

                            p_sum = points_near.at(0); // 初始化为第一个最近邻点

                            // 迭代合并所有最近邻点以构建局部平面
                            for (int j = 1; j < points_near.size(); j++) {
                                // 如果合并点数量不足最小匹配点数，直接合并
                                if (p_sum.pt_num < options::MIN_NUM_MATCH_POINTS) {
                                    p1 = Merge2(points_near.at(j), p_sum);
                                    p_sum = p1;
                                    continue;
                                } else {
                                    // 进行特征值分解以判断平面质量
                                    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 3, 3>> es1(p_sum.cov);
                                    Eigen::Vector3d normal_x = es1.eigenvectors().col(0); // 主方向（法向量）
                                    Eigen::Vector3d normal_y = es1.eigenvectors().col(1); // 次方向
                                    Eigen::Vector3d normal_z = es1.eigenvectors().col(2); // 最小方向
                                    
                                    // 计算当前点到平面的距离
                                    Eigen::Vector3d pq = Eigen::Vector3d(p_sum.x, p_sum.y, p_sum.z) -
                                                        Eigen::Vector3d(point_world.x, point_world.y, point_world.z);
                                    
                                    // 计算平面内的马氏距离
                                    double p2q =
                                        pow(normal_y.transpose() * pq, 2) * (1.0f / es1.eigenvalues().real()(1)) +
                                        pow(normal_z.transpose() * pq, 2) * (1.0f / es1.eigenvalues().real()(2));

                                    // 判断是否继续合并点：特征值过小或距离过大则继续合并
                                    if ((es1.eigenvalues().real()(1) < 0.04f) ||
                                        p2q > t_stop_pseudo_merge_ * t_stop_pseudo_merge_) {
                                        p1 = Merge2(points_near.at(j), p_sum);
                                        p_sum = p1;
                                        continue;
                                    } else {
                                        // 平面质量足够好，停止合并，删除后续点
                                        for (int k = j; k < points_near.size();) points_near.pop_back();
                                    }
                                }
                            }
                            
                            // 最终检查合并结果的质量
                            if (p_sum.pt_num < options::MIN_NUM_MATCH_POINTS) {
                                points_near.clear(); // 点数不足，清空最近邻点
                            } else {
                                // 重新计算最终平面的特征值和特征向量
                                Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 3, 3>> es1(p_sum.cov);
                                Eigen::Vector3d normal_x = es1.eigenvectors().col(0); // 法向量
                                Eigen::Vector3d normal_y = es1.eigenvectors().col(1);
                                Eigen::Vector3d normal_z = es1.eigenvectors().col(2);
                                
                                Eigen::Vector3d pq = Eigen::Vector3d(p_sum.x, p_sum.y, p_sum.z) -
                                                    Eigen::Vector3d(point_world.x, point_world.y, point_world.z);
                                
                                // 重新计算平面内距离
                                double p2q = pow(normal_y.transpose() * pq, 2) * (1.0f / es1.eigenvalues().real()(1)) +
                                            pow(normal_z.transpose() * pq, 2) * (1.0f / es1.eigenvalues().real()(2));

                                // 最终质量检查
                                if ((es1.eigenvalues().real()(1) >= 0.04f) &&
                                    p2q <= t_stop_pseudo_merge_ * t_stop_pseudo_merge_) {
                                    // 计算点到平面的距离（沿法向量方向）
                                    double p2pl = normal_x.transpose() * pq;

                                    // 距离阈值检查：基于点的距离设置自适应阈值
                                    if (points_near.empty() || fabs(p2pl) > 1.0f / 9.0f * sqrt(p_body.norm())) {
                                        points_near.clear(); // 距离过大，清空最近邻点
                                    } else {
                                        // 设置点的不确定性为距离的平方
                                        point_body.uncertainty = p2pl * p2pl;
                                    }
                                } else
                                    points_near.clear(); // 平面质量不足，清空最近邻点
                            }
                        }
                    } });
                }
            },
            "    ObsModel (Lidar Match)");

        /// ==== 统计有效特征点并构建观测方程 ====
        effect_feat_num_ = 0; // 有效特征点计数器

        // 调整对应点和法向量容器大小
        corr_pts_.resize(cnt_pts);
        corr_norm_.resize(cnt_pts);
        
        // 遍历所有点，筛选有效的表面特征点
        for (int i = 0; i < cnt_pts; i++)
        {
            // 判断是否为有效的表面特征点
            if (!nearest_points_[i].empty())
            {
                point_selected_surf_[i] = true;  // 标记为选中的表面点
            }
            else
                point_selected_surf_[i] = false; // 标记为未选中

            // 将有效特征点添加到对应数组中
            if (point_selected_surf_[i])
            {
                point_selected_idx_[effect_feat_num_] = i;                    // 存储点索引
                corr_norm_[effect_feat_num_] = plane_coef_[i];               // 存储对应的平面系数
                corr_pts_[effect_feat_num_] = scan_down_reg_.at(i);          // 存储对应的点
                effect_feat_num_++;                                          // 有效特征点计数加1
            }
            else
                scan_down_reg_.at(i).uncertainty = (init_uncertainty_);      // 设置无效点的不确定性为初始值
        }
        
        // 调整容器大小到实际有效特征点数量
        corr_pts_.resize(effect_feat_num_);
        corr_norm_.resize(effect_feat_num_);

        // 检查是否有足够的有效特征点
        if (effect_feat_num_ < 1)
        {
            ekfom_data.valid = false;           // 标记数据无效
            ROS_WARN("No Effective Points!");  // 输出警告信息
            return;
        }
        
        /// ==== 构建雅可比矩阵和观测方程 ====
        Timer::Evaluate(
            [&, this]()
            {
                // 初始化雅可比矩阵和观测向量
                ekfom_data.h_x = Eigen::MatrixXd::Zero(effect_feat_num_, 12); // 雅可比矩阵 (观测数 × 状态维度)
                ekfom_data.h.resize(effect_feat_num_);                        // 观测残差向量
                ekfom_data.R_inv.resize(effect_feat_num_, 1);                 // 观测噪声逆矩阵
                
                // 创建有效特征点的索引数组
                std::vector<size_t> index2(effect_feat_num_);
                for (size_t i = 0; i < effect_feat_num_; i++)
                {
                    index2[i] = i;
                }
                
                // 提取外参和状态信息
                const common::M3D off_R = s.offset_R_L_I.toRotationMatrix(); // 外参旋转矩阵
                const common::V3D off_t = s.offset_T_L_I;                     // 外参平移向量
                const common::M3D Rt = s.rot.toRotationMatrix().transpose();  // 状态旋转矩阵的转置

                // 并行计算每个有效特征点的雅可比矩阵和残差
                std::for_each(std::execution::par_unseq, index2.begin(), index2.end(), [&](const size_t &i)
                            {
                                // 坐标转换：机体坐标系 -> IMU坐标系 -> 世界坐标系
                                common::V3D point_this_be(corr_pts_[i].x, corr_pts_[i].y, corr_pts_[i].z); // 机体坐标系下的点
                                common::M3D point_be_crossmat = SKEW_SYM_MATRIX(point_this_be);            // 反对称矩阵
                                common::V3D point_this = off_R * point_this_be + off_t;                    // 转换到IMU坐标系
                                common::M3D point_crossmat = SKEW_SYM_MATRIX(point_this);                 // 反对称矩阵
                                common::V3D point_world = s.rot.toRotationMatrix() * point_this + s.pos;  // 转换到世界坐标系

                                // 提取平面法向量（最小特征值对应的特征向量）
                                Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 3, 3>> es1(corr_norm_[i].cov);
                                common::V3D norm_p = es1.eigenvectors().col(0);                           // 平面法向量
                                common::V3D map_q(corr_norm_[i].x, corr_norm_[i].y, corr_norm_[i].z);    // 地图中对应点的位置

                                // 构建雅可比矩阵：对位置和姿态的偏导数
                                ekfom_data.h_x.block<1, 3>(i, 0) << norm_p.transpose();                                        // 对位置的偏导数
                                ekfom_data.h_x.block<1, 3>(i, 3) << -norm_p.transpose() * (s.rot.toRotationMatrix() * point_crossmat); // 对姿态的偏导数
                                
                                // 计算观测残差：点到平面的距离
                                ekfom_data.h(i) = norm_p.transpose() * (map_q - point_world);

                                // 计算观测噪声：基于平面厚度和不确定性
                                double thickness = (norm_p.transpose() * (corr_norm_[i].cov) * norm_p); // 平面厚度
                                ekfom_data.R_inv(i) = 1.0f / (exp(t_ratio_b_ * corr_norm_[i].uncertainty) * thickness); // 观测噪声的倒数
                            });       
            },"    ObsModel (IEKF Build Jacobian)");
    }

    /////////////////////////////////////  debug save / show /////////////////////////////////////////////////////

    void LaserMapping::PublishPath(const ros::Publisher pub_path)
    {
        SetPosestamp(msg_body_pose_);
        msg_body_pose_.header.stamp = ros::Time().fromSec(lidar_end_time_);
        msg_body_pose_.header.frame_id = "camera_init";

        /*** if path is too large, the rviz will crash ***/
        path_.poses.push_back(msg_body_pose_);
        if (run_in_offline_ == false)
        {
            pub_path.publish(path_);
        }
    }

    void LaserMapping::PublishOdometry(const ros::Publisher &pub_odom_aft_mapped)
    {
        odom_aft_mapped_.header.frame_id = "camera_init";
        odom_aft_mapped_.child_frame_id = "body";
        odom_aft_mapped_.header.stamp = ros::Time().fromSec(lidar_end_time_); // ros::Time().fromSec(lidar_end_time_);
        SetPosestamp(odom_aft_mapped_.pose);

        auto P = kf_.get_P();
        for (int i = 0; i < 6; i++)
        {
            odom_aft_mapped_.pose.covariance[i * 6 + 0] = P(i, 0);
            odom_aft_mapped_.pose.covariance[i * 6 + 1] = P(i, 1);
            odom_aft_mapped_.pose.covariance[i * 6 + 2] = P(i, 2);
            odom_aft_mapped_.pose.covariance[i * 6 + 3] = P(i, 3);
            odom_aft_mapped_.pose.covariance[i * 6 + 4] = P(i, 4);
            odom_aft_mapped_.pose.covariance[i * 6 + 5] = P(i, 5);
        }
        pub_odom_aft_mapped.publish(odom_aft_mapped_);
        static tf::TransformBroadcaster br;
        tf::Transform transform;
        tf::Quaternion q;
        transform.setOrigin(tf::Vector3(odom_aft_mapped_.pose.pose.position.x, odom_aft_mapped_.pose.pose.position.y,
                                        odom_aft_mapped_.pose.pose.position.z));
        q.setW(odom_aft_mapped_.pose.pose.orientation.w);
        q.setX(odom_aft_mapped_.pose.pose.orientation.x);
        q.setY(odom_aft_mapped_.pose.pose.orientation.y);
        q.setZ(odom_aft_mapped_.pose.pose.orientation.z);
        transform.setRotation(q);
        br.sendTransform(tf::StampedTransform(transform, odom_aft_mapped_.header.stamp, "camera_init", "body"));
    }

    void LaserMapping::PublishFrameWorld()
    {
        PointCloudType::Ptr laserCloudWorld;

        PointCloudType::Ptr laserCloudFullRes(scan_undistort_);
        int size = laserCloudFullRes->points.size();
        laserCloudWorld.reset(new PointCloudType(size, 1));
        for (int i = 0; i < size; i++)
        {
            common::V3D p_body(laserCloudFullRes->points.at(i).x, laserCloudFullRes->points.at(i).y,
                               laserCloudFullRes->points.at(i).z);
            common::V3D p_global(state_point_.rot * (state_point_.offset_R_L_I * p_body + state_point_.offset_T_L_I) +
                                 state_point_.pos);

            laserCloudWorld->points[i].x = p_global(0);
            laserCloudWorld->points[i].y = p_global(1);
            laserCloudWorld->points[i].z = p_global(2);
            laserCloudWorld->points[i].intensity = laserCloudFullRes->points.at(i).intensity;
            laserCloudWorld->points[i].curvature = laserCloudFullRes->points.at(i).curvature;
            laserCloudWorld->points[i].normal_x = laserCloudFullRes->points.at(i).normal_x;
            laserCloudWorld->points[i].normal_y = laserCloudFullRes->points.at(i).normal_y;
            laserCloudWorld->points[i].normal_z = laserCloudFullRes->points.at(i).normal_z;
        }
        sensor_msgs::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);
        laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time_);
        laserCloudmsg.header.frame_id = "camera_init";
        pub_laser_cloud_world_.publish(laserCloudmsg);
        publish_count_ -= options::PUBFRAME_PERIOD;
    }

    void LaserMapping::GetRainbowColor(float value, common::V3F &color)
    {
        // this is HSV color palette with hue values going only from 0.0 to 0.833333.

        value = std::min(value, 1.0f);
        value = std::max(value, 0.0f);

        float h = value * 5.0f + 1.0f;
        int i = floor(h);
        float f = h - i;
        if (!(i & 1))
            f = 1 - f; // if i is even
        float n = 1 - f;

        if (i <= 1)
            color[0] = n, color[1] = 0, color[2] = 1;
        else if (i == 2)
            color[0] = 0, color[1] = n, color[2] = 1;
        else if (i == 3)
            color[0] = 0, color[1] = 1, color[2] = n;
        else if (i == 4)
            color[0] = n, color[1] = 1, color[2] = 0;
        else if (i >= 5)
            color[0] = 1, color[1] = n, color[2] = 0;
    }

    /**
     * [功能描述]：发布激光雷达地图点云数据，包括普通点云和高斯分布可视化
     * 该函数从ivox体素网格中获取地图点，并将其转换为ROS点云消息进行发布
     * 同时支持高斯分布的椭球可视化，用于显示点的不确定性
     */
    void LaserMapping::PublishMap()
    {
        // 从ivox体素网格中获取所有地图点
        ivox_->GetMapPoints(map_points_);
        int size = map_points_.size();
        
        // 打印地图统计信息：点数、体素数、平均每体素点数
        LOG(INFO) << "map size " << size << " voxel size " << ivox_->NumValidGrids() << " average point each voxel "
                << float(size) / float(ivox_->NumValidGrids());
        
        // 如果地图为空，直接返回
        if (size == 0)
            return;
        
        // 创建ROS点云消息和PCL点云对象
        sensor_msgs::PointCloud2 laserCloudMap;
        PointCloudType::Ptr map_pub(new PointCloudType());
        PointType2 body_p; // 单个点的临时存储
        map_pub->clear();
        
        // 初始化颜色映射所需的变量
        double max_intensity = 0, min_z = 10000.0, max_z = -10000.0;
        min_z = state_point_.pos.z() + 3;  // 设置Z轴最小值为当前位置+3米
        max_z = state_point_.pos.z() + 30; // 设置Z轴最大值为当前位置+30米
        
        {
            // 高斯分布可视化相关变量初始化
            common::M3D world_cov;                    // 协方差矩阵
            visualization_msgs::Marker p_cov;         // 椭球标记消息
            p_cov.type = visualization_msgs::Marker::SPHERE;  // 设置为椭球类型
            p_cov.action = visualization_msgs::Marker::ADD;   // 添加标记
            p_cov.header.frame_id = "camera_init";            // 坐标系
            p_cov.header.stamp = ros::Time().fromSec(lidar_end_time_); // 时间戳
            p_cov.lifetime = ros::Duration(10);               // 生存时间10秒
            pa_cov.markers.clear();                           // 清空之前的标记
            
            // 当前状态位置和距离计算变量
            common::V3D state_pos(state_point_.pos);
            common::V3D map_p;
            double dis = 0;
            int pt_cnt = -1;
            
            // 遍历所有地图点
            for (auto it = map_points_.begin(); it != map_points_.end(); it++)
            {
                // 获取当前点的协方差矩阵
                world_cov = it->cov;
                
                // 对协方差矩阵进行特征值分解，获取主方向和特征值
                Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 3, 3>> es_cov3(world_cov);
                Eigen::Matrix3d sorted_evecs_cov3 = es_cov3.eigenvectors().real().transpose();
                // 确保第三行为前两行的叉积，保持右手坐标系
                sorted_evecs_cov3.row(2) = sorted_evecs_cov3.row(0).cross(sorted_evecs_cov3.row(1));
                common::V3D ev_cov3 = es_cov3.eigenvalues().real(); // 特征值

                // 填充点云数据
                body_p.x = it->x;
                body_p.y = it->y;
                body_p.z = it->z;
                body_p.curvature = it->pt_num;        // 点数量存储在曲率字段
                body_p.intensity = it->intensity;     // 强度值
                body_p.normal_x = (it->uncertainty);  // 不确定性存储在法向量x分量
                body_p.normal_y = it->use_num;        // 使用次数存储在法向量y分量
                body_p.normal_z = sqrt(fabs(ev_cov3(2))); // 最大特征值的平方根存储在法向量z分量
                map_pub->push_back(body_p);

                // 如果启用高斯分布发布
                if (gaussian_publish_en_)
                {
                    // 更新最大强度值
                    if (it->intensity > max_intensity)
                        max_intensity = it->intensity;

                    // 如果点数量小于最小阈值，跳过可视化
                    if (it->pt_num <= gaussian_pub_min_cnt_)
                        continue;
                    
                    // 计算当前点到状态位置的距离
                    map_p << it->x, it->y, it->z;
                    dis = (map_p.x() - state_pos.x()) * (map_p.x() - state_pos.x()) +
                        (map_p.y() - state_pos.y()) * (map_p.y() - state_pos.y()) +
                        (map_p.z() - state_pos.z()) * (map_p.z() - state_pos.z());

                    // 只发布局部高斯混合模型，距离过远的点跳过
                    if (dis > gaussian_pub_dis_ * gaussian_pub_dis_)
                        continue; // pub local gmm only

                    // 设置椭球的尺寸，基于协方差矩阵的特征值（4倍标准差直径）
                    p_cov.scale.x = 4 * sqrt(fabs(ev_cov3(0))); // scale is diameter 2 sigma
                    p_cov.scale.y = 4 * sqrt(fabs(ev_cov3(1)));
                    p_cov.scale.z = 4 * sqrt(fabs(ev_cov3(2)));
                    
                    // 设置椭球的位置
                    p_cov.pose.position.x = it->x;
                    p_cov.pose.position.y = it->y;
                    p_cov.pose.position.z = it->z;
                    p_cov.id = pa_cov.markers.size(); // 设置唯一ID
                    float x = 0;

                    // 根据高度计算彩虹色
                    common::V3F rgb;
                    {
                        x = 1.0 - ((it->z - min_z) / (max_z - min_z)); // 归一化高度值
                        GetRainbowColor(x, rgb); // 获取对应的彩虹色
                    }
                    // 设置椭球颜色
                    p_cov.color.r = rgb[0];
                    p_cov.color.g = rgb[1];
                    p_cov.color.b = rgb[2];
                    p_cov.ns = "gaussian_map"; // 命名空间
                    p_cov.color.a = 0.8;      // 透明度

                    // 根据特征向量设置椭球的方向
                    Eigen::Matrix3d rotation3 = sorted_evecs_cov3.transpose();
                    Eigen::Quaterniond eq3(rotation3);
                    p_cov.pose.orientation.w = eq3.w();
                    p_cov.pose.orientation.x = eq3.x();
                    p_cov.pose.orientation.y = eq3.y();
                    p_cov.pose.orientation.z = eq3.z();
                    
                    // 添加到标记数组
                    pa_cov.markers.push_back(p_cov);
                }
            }
            // 发布高斯分布可视化标记
            map_cov_pub.publish(pa_cov);
        }

        // 将PCL点云转换为ROS消息格式
        pcl::toROSMsg(*map_pub, laserCloudMap);
        laserCloudMap.header.stamp = ros::Time().fromSec(lidar_end_time_); // 设置时间戳
        laserCloudMap.header.frame_id = "camera_init";                     // 设置坐标系
        pub_map.publish(laserCloudMap); // 发布点云地图
    }

    /**
     * [功能描述]：发布当前帧配准后的世界坐标系点云数据，包含点云、协方差可视化和对应关系可视化
     * @param pub_laser_cloud_reg_world：用于发布配准点云的ROS发布器
     * 该函数将当前帧的激光雷达数据转换到世界坐标系并发布，同时提供不确定性和对应关系的可视化
     */
    void LaserMapping::PublishFrameRegWorld(const ros::Publisher &pub_laser_cloud_reg_world)
    {
        // 获取下采样配准点云的大小
        int size = scan_down_reg_.size();
        // 创建指定大小的点云对象
        PointCloudType::Ptr laser_cloud(new PointCloudType(size, 1));

        // 遍历所有下采样配准点，将其从机体坐标系转换到世界坐标系
        for (int i = 0; i < size; i++)
        {
            // 将点从机体坐标系转换到世界坐标系用于发布
            PointBodyToWorldPub(&scan_down_reg_.at(i), &laser_cloud->points[i]);
            // 如果该点被选为表面点，将平面系数的点数量存储在normal_x中
            if (point_selected_surf_[i])
                laser_cloud->points[i].normal_x = plane_coef_[i].pt_num;
            else
                laser_cloud->points[i].normal_x = 0; // 非表面点设为0
        }
        
        // 将PCL点云转换为ROS消息格式并发布
        sensor_msgs::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*laser_cloud, laserCloudmsg);
        laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time_); // 设置时间戳
        laserCloudmsg.header.frame_id = "camera_init";                     // 设置坐标系
        pub_laser_cloud_reg_world.publish(laserCloudmsg);                 // 发布点云
        
        // 减少发布计数器
        publish_count_ -= options::PUBFRAME_PERIOD;
        
        // 如果启用高斯分布发布，创建协方差可视化
        if (gaussian_publish_en_)
        {
            common::M3D world_cov;                        // 世界坐标系协方差矩阵
            visualization_msgs::MarkerArray pa_scan_cov;  // 扫描协方差标记数组
            visualization_msgs::Marker p_cov;             // 单个协方差标记

            // 设置椭球标记的基本属性
            p_cov.type = visualization_msgs::Marker::SPHERE;           // 椭球类型
            p_cov.action = visualization_msgs::Marker::ADD;            // 添加操作
            p_cov.header.frame_id = "camera_init";                     // 坐标系
            p_cov.header.stamp = ros::Time().fromSec(lidar_end_time_); // 时间戳
            p_cov.lifetime = ros::Duration();                          // 永久显示
            pa_scan_cov.markers.clear();                               // 清空标记数组

            // 处理扫描点的协方差可视化
            size = scan_down_reg_.size();
            scan_down_reg_.resize(size);
            for (int i = 0; i < scan_down_reg_.size(); i++)
            {
                // 将点从机体坐标系转换到世界坐标系
                PointBodyToWorld(&scan_down_reg_[i], &scan_down_world_[i]);
                common::V3D p_body(scan_down_world_[i].x, scan_down_world_[i].y, scan_down_world_[i].z);
                
                // 提取3x3协方差矩阵
                world_cov = scan_down_world_.at(i).cov.block<3, 3>(0, 0);
                
                // 设置椭球位置
                p_cov.pose.position.x = scan_down_world_[i].x;
                p_cov.pose.position.y = scan_down_world_[i].y;
                p_cov.pose.position.z = scan_down_world_[i].z;
                p_cov.id = i;                    // 设置唯一ID
                p_cov.ns = "scan_cov";          // 命名空间为扫描协方差
                
                // 设置扫描点椭球为红色
                p_cov.color.r = 1;
                p_cov.color.g = 0;
                p_cov.color.b = 0;

                // 对协方差矩阵进行特征值分解
                Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 3, 3>> es_cov3(world_cov);
                Eigen::Matrix3d sorted_evecs_cov3 = es_cov3.eigenvectors().real().transpose();
                // 确保第三行为前两行的叉积，保持右手坐标系
                sorted_evecs_cov3.row(2) = sorted_evecs_cov3.row(0).cross(sorted_evecs_cov3.row(1));
                common::V3D ev_cov3 = es_cov3.eigenvalues().real(); // 获取特征值
                
                // 设置椭球尺寸（12倍标准差直径，比地图椭球更大以便区分）
                p_cov.scale.x = 12 * sqrt(fabs(ev_cov3(0))); // scale is diameter
                p_cov.scale.y = 12 * sqrt(fabs(ev_cov3(1)));
                p_cov.scale.z = 12 * sqrt(fabs(ev_cov3(2)));
                p_cov.color.a = 1; // 完全不透明
                
                // 根据特征向量设置椭球方向
                Eigen::Matrix3d rotation3 = sorted_evecs_cov3.transpose();
                Eigen::Quaterniond eq3(rotation3);
                p_cov.pose.orientation.w = eq3.w();
                p_cov.pose.orientation.x = eq3.x();
                p_cov.pose.orientation.y = eq3.y();
                p_cov.pose.orientation.z = eq3.z();
                pa_scan_cov.markers.push_back(p_cov);
            }

            // 处理对应关系法线的协方差可视化
            size = corr_norm_.size();
            for (int i = 0; i < size; i++)
            {
                // 获取对应法线点的协方差矩阵
                world_cov = corr_norm_.at(i).cov.block<3, 3>(0, 0);
                
                // 设置椭球位置
                p_cov.pose.position.x = corr_norm_.at(i).x;
                p_cov.pose.position.y = corr_norm_.at(i).y;
                p_cov.pose.position.z = corr_norm_.at(i).z;
                p_cov.id = i;

                // 对协方差矩阵进行特征值分解
                Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 3, 3>> es_cov3(world_cov);
                Eigen::Matrix3d sorted_evecs_cov3 = es_cov3.eigenvectors().real().transpose();
                sorted_evecs_cov3.row(2) = sorted_evecs_cov3.row(0).cross(sorted_evecs_cov3.row(1));
                common::V3D ev_cov3 = es_cov3.eigenvalues().real();

                // 设置椭球方向
                Eigen::Matrix3d rotation3 = sorted_evecs_cov3.transpose();
                Eigen::Quaterniond eq3(rotation3);
                p_cov.pose.orientation.w = eq3.w();
                p_cov.pose.orientation.x = eq3.x();
                p_cov.pose.orientation.y = eq3.y();
                p_cov.pose.orientation.z = eq3.z();

                // 设置对应点椭球尺寸（4倍标准差直径）
                p_cov.scale.x = 4 * sqrt(fabs(ev_cov3(0))); // scale is diameter
                p_cov.scale.y = 4 * sqrt(fabs(ev_cov3(1)));
                p_cov.scale.z = 4 * sqrt(fabs(ev_cov3(2)));
                
                // 设置对应点椭球为蓝色半透明
                p_cov.color.r = 0;
                p_cov.color.g = 0;
                p_cov.color.b = 1;
                p_cov.color.a = 0.3;
                p_cov.ns = "cor_cov";                                    // 对应协方差命名空间
                p_cov.type = visualization_msgs::Marker::SPHERE;         // 椭球类型
                pa_scan_cov.markers.push_back(p_cov);

                // 创建法线方向箭头可视化
                rotation3.setZero();
                rotation3.row(0) = sorted_evecs_cov3.row(0);  // 使用第一个特征向量作为法线方向
                // 确保方向一致性
                if (rotation3.row(0).sum() < 0)
                    rotation3 = -rotation3;
                Eigen::Quaterniond eq(rotation3.transpose());
                eq.normalize();
                
                // 设置箭头方向
                p_cov.pose.orientation.w = eq.w();
                p_cov.pose.orientation.x = eq.x();
                p_cov.pose.orientation.y = eq.y();
                p_cov.pose.orientation.z = eq.z();

                // 设置箭头尺寸（长度1，宽度0.2）
                p_cov.scale.x = 1;
                p_cov.scale.y = 0.2;
                p_cov.scale.z = 0.2;
                
                // 设置箭头颜色为黄色
                p_cov.color.r = 1;
                p_cov.color.g = 1;
                p_cov.color.b = 0;
                p_cov.color.a = 1;
                p_cov.ns = "cor_normal";                          // 对应法线命名空间
                p_cov.type = visualization_msgs::Marker::ARROW;   // 箭头类型
                pa_scan_cov.markers.push_back(p_cov);
            }
            
            // 创建扫描-对应连接线可视化
            p_cov.type = visualization_msgs::Marker::LINE_LIST;   // 线段列表类型
            p_cov.ns = "scan_cor_line";                           // 扫描对应线命名空间
            
            // 重置位置和方向
            p_cov.pose.position.x = 0;
            p_cov.pose.position.y = 0;
            p_cov.pose.position.z = 0;
            p_cov.pose.orientation.w = 1;
            p_cov.pose.orientation.x = 0;
            p_cov.pose.orientation.y = 0;
            p_cov.pose.orientation.z = 0;
            
            // 设置连接线为绿色
            p_cov.color.r = 0;
            p_cov.color.g = 1;
            p_cov.color.b = 0;
            p_cov.color.a = 1;
            p_cov.scale.x = 0.03;  // 线宽

            // 为每个对应关系创建连接线
            for (int i = 0; i < size; i++)
            {
                geometry_msgs::Point p;
                PointType eff_world;
                p_cov.id = i;
                
                // 对应法线点位置
                p.x = corr_norm_[i].x;
                p.y = corr_norm_[i].y;
                p.z = corr_norm_[i].z;
                p_cov.points.push_back(p);
                
                // 对应点转换到世界坐标系
                PointBodyToWorld(&corr_pts_.at(i), &eff_world);
                p.x = eff_world.x;
                p.y = eff_world.y;
                p.z = eff_world.z;
                p_cov.points.push_back(p);
                
                // 添加线段标记并清空点列表准备下一条线
                pa_scan_cov.markers.push_back(p_cov);
                p_cov.points.clear();
            }
            // 发布所有可视化标记
            point_cov_pub.publish(pa_scan_cov);
        }
    }

    void LaserMapping::Savetrajectory(const std::string &traj_file)
    {
        std::ofstream ofs;
        ofs.open(traj_file, std::ios::out);
        if (!ofs.is_open())
        {
            LOG(ERROR) << "Failed to open traj_file: " << traj_file;
            return;
        }

        ofs << "#timestamp x y z q_x q_y q_z q_w" << std::endl;
        for (const auto &p : path_.poses)
        {
            ofs << std::fixed << std::setprecision(6) << p.header.stamp.toSec() << " " << std::setprecision(15)
                << p.pose.position.x << " " << p.pose.position.y << " " << p.pose.position.z << " " << p.pose.orientation.x
                << " " << p.pose.orientation.y << " " << p.pose.orientation.z << " " << p.pose.orientation.w << std::endl;
        }

        ofs.close();

        // ofs kitti
        std::string kitti_file = (std::string(std::string(ROOT_DIR) + "Log/" + "kitti.txt"));
        ofs.open(kitti_file, std::ios::out);

        if (!ofs.is_open())
        {
            LOG(ERROR) << "Failed to open kitti_file: " << kitti_file;
            return;
        }

        for (const auto &p : path_.poses)
        {
            SO3 r_matrix;
            r_matrix.x() = p.pose.orientation.x;
            r_matrix.y() = p.pose.orientation.y;
            r_matrix.z() = p.pose.orientation.z;
            r_matrix.w() = p.pose.orientation.w;
            Eigen::MatrixXd kitti_pose;
            kitti_pose.resize(3, 4);
            kitti_pose.block<3, 3>(0, 0) = r_matrix.toRotationMatrix();
            kitti_pose(0, 3) = p.pose.position.x;
            kitti_pose(1, 3) = p.pose.position.y;
            kitti_pose(2, 3) = p.pose.position.z;
            ofs << std::fixed << std::setprecision(18) << kitti_pose(0, 0) << " " << kitti_pose(0, 1) << " "
                << kitti_pose(0, 2) << " " << kitti_pose(0, 3) << " " << kitti_pose(1, 0) << " " << kitti_pose(1, 1) << " "
                << kitti_pose(1, 2) << " " << kitti_pose(1, 3) << " " << kitti_pose(2, 0) << " " << kitti_pose(2, 1) << " "
                << kitti_pose(2, 2) << " " << kitti_pose(2, 3) << std::endl;
        }
        ofs.close();
    }

    ///////////////////////////  private method /////////////////////////////////////////////////////////////////////
    template <typename T>
    void LaserMapping::SetPosestamp(T &out)
    {
        out.pose.position.x = state_point_.pos(0);
        out.pose.position.y = state_point_.pos(1);
        out.pose.position.z = state_point_.pos(2);
        out.pose.orientation.x = state_point_.rot.coeffs()[0];
        out.pose.orientation.y = state_point_.rot.coeffs()[1];
        out.pose.orientation.z = state_point_.rot.coeffs()[2];
        out.pose.orientation.w = state_point_.rot.coeffs()[3];
    }

    void LaserMapping::PointBodyToWorld(const PointType *pi, PointType *const po)
    {
        common::V3D p_body(pi->x, pi->y, pi->z);
        common::V3D p_this(state_point_.offset_R_L_I * p_body + state_point_.offset_T_L_I);
        common::V3D p_global(state_point_.rot * p_this + state_point_.pos);
        po->x = p_global(0);
        po->y = p_global(1);
        po->z = p_global(2);
        po->intensity = pi->intensity;
        po->uncertainty = pi->uncertainty;
        po->pt_num = pi->pt_num;
        po->use_num = pi->use_num;
        po->time = pi->time;

        common::M3D point_crossmat;
        point_crossmat << SKEW_SYM_MATRIX(p_this);

        po->cov = state_point_.rot * state_point_.offset_R_L_I * pi->cov * state_point_.offset_R_L_I.conjugate() *
                  state_point_.rot.conjugate();
    }

    void LaserMapping::PointBodyToWorldPub(const PointType *pi, PointType2 *const po)
    {
        common::V3D p_body(pi->x, pi->y, pi->z);
        common::V3D p_global(state_point_.rot * (state_point_.offset_R_L_I * p_body + state_point_.offset_T_L_I) +
                             state_point_.pos);

        po->x = p_global(0);
        po->y = p_global(1);
        po->z = p_global(2);
        po->intensity = pi->intensity;
        po->curvature = pi->pt_num;
    }

    void LaserMapping::PointBodyToWorld(const common::V3F &pi, PointType *const po)
    {
        common::V3D p_body(pi.x(), pi.y(), pi.z());
        common::V3D p_global(state_point_.rot * (state_point_.offset_R_L_I * p_body + state_point_.offset_T_L_I) +
                             state_point_.pos);

        po->x = p_global(0);
        po->y = p_global(1);
        po->z = p_global(2);
        po->intensity = std::abs(po->z);
    }

    void LaserMapping::PointBodyLidarToIMU(PointType const *const pi, PointType *const po)
    {
        common::V3D p_body_lidar(pi->x, pi->y, pi->z);
        common::V3D p_body_imu(state_point_.offset_R_L_I * p_body_lidar + state_point_.offset_T_L_I);

        po->x = p_body_imu(0);
        po->y = p_body_imu(1);
        po->z = p_body_imu(2);
        po->intensity = pi->intensity;
    }

} // namespace akf_lio