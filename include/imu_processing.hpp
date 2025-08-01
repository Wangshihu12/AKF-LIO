#ifndef AKF_LIO_IMU_PROCESSING_H
#define AKF_LIO_IMU_PROCESSING_H

#include <glog/logging.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <cmath>
#include <deque>
#include <fstream>

#include "common_lib.h"
#include "so3_math.h"
#include "use-ikfom.hpp"
#include "utils.h"

namespace akf_lio
{

    constexpr int MAX_INI_COUNT = 20;

    bool time_list(const PointType2 &x, const PointType2 &y) { return (x.curvature < y.curvature); };

    /// IMU Process and undistortion
    class ImuProcess
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        ImuProcess();
        ~ImuProcess();

        void Reset();
        void SetExtrinsic(const common::V3D &transl, const common::M3D &rot);
        void SetGyrCov(const common::V3D &scaler);
        void SetAccCov(const common::V3D &scaler);
        void SetGyrBiasCov(const common::V3D &b_g);
        void SetAccBiasCov(const common::V3D &b_a);
        void Process(const common::MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state,
                     PointCloudType::Ptr pcl_un_);

        std::ofstream fout_imu_;
        Eigen::Matrix<double, 12, 12> Q_;
        common::V3D cov_acc_;
        common::V3D cov_gyr_;
        common::V3D cov_acc_scale_;
        common::V3D cov_gyr_scale_;
        common::V3D cov_bias_gyr_;
        common::V3D cov_bias_acc_;

        double alpha = 1;
        Eigen::Matrix<double, 12, 3> est_noise_;

    private:
        void IMUInit(const common::MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, int &N);
        void UndistortPcl(const common::MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state,
                          PointCloudType &pcl_out);
        void OnlyPropagate(const common::MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state,
                           PointCloudType &pcl_out);

        PointCloudType::Ptr cur_pcl_un_;
        sensor_msgs::ImuConstPtr last_imu_;
        std::deque<sensor_msgs::ImuConstPtr> v_imu_;
        std::vector<common::Pose6D> IMUpose_;
        std::vector<common::M3D> v_rot_pcl_;
        common::M3D Lidar_R_wrt_IMU_;
        common::V3D Lidar_T_wrt_IMU_;
        common::V3D mean_acc_;
        common::V3D mean_gyr_;
        common::V3D angvel_last_;
        common::V3D acc_s_last_;
        double last_lidar_end_time_ = 0;
        int init_iter_num_ = 1;
        bool b_first_frame_ = true;
        bool imu_need_init_ = true;
        double last_dt_square_sum = 0;
    };

    ImuProcess::ImuProcess() : b_first_frame_(true), imu_need_init_(true)
    {
        init_iter_num_ = 1;
        Q_ = process_noise_cov();
        cov_acc_ = common::V3D(0.1, 0.1, 0.1);
        cov_gyr_ = common::V3D(0.1, 0.1, 0.1);
        cov_bias_gyr_ = common::V3D(0.0001, 0.0001, 0.0001);
        cov_bias_acc_ = common::V3D(0.0001, 0.0001, 0.0001);
        mean_acc_ = common::V3D(0, 0, -1.0);
        mean_gyr_ = common::V3D(0, 0, 0);
        angvel_last_ = common::Zero3d;
        Lidar_T_wrt_IMU_ = common::Zero3d;
        Lidar_R_wrt_IMU_ = common::Eye3d;
        last_imu_.reset(new sensor_msgs::Imu());
    }

    ImuProcess::~ImuProcess() {}

    void ImuProcess::Reset()
    {
        mean_acc_ = common::V3D(0, 0, -1.0);
        mean_gyr_ = common::V3D(0, 0, 0);
        angvel_last_ = common::Zero3d;
        imu_need_init_ = true;
        init_iter_num_ = 1;
        v_imu_.clear();
        IMUpose_.clear();
        last_imu_.reset(new sensor_msgs::Imu());
        cur_pcl_un_.reset(new PointCloudType());
    }

    void ImuProcess::SetExtrinsic(const common::V3D &transl, const common::M3D &rot)
    {
        Lidar_T_wrt_IMU_ = transl;
        Lidar_R_wrt_IMU_ = rot;
    }

    void ImuProcess::SetGyrCov(const common::V3D &scaler) { cov_gyr_scale_ = scaler; }

    void ImuProcess::SetAccCov(const common::V3D &scaler) { cov_acc_scale_ = scaler; }

    void ImuProcess::SetGyrBiasCov(const common::V3D &b_g) { cov_bias_gyr_ = b_g; }

    void ImuProcess::SetAccBiasCov(const common::V3D &b_a) { cov_bias_acc_ = b_a; }

    /**
     * [功能描述]：执行IMU传感器的初始化和标定过程
     * @param meas：测量组，包含IMU数据队列
     * @param kf_state：扩展卡尔曼滤波器状态，用于存储初始化结果
     * @param N：初始化迭代计数器的引用
     * 
     * 该函数完成以下初始化任务：
     * 1. 估计重力向量和陀螺仪偏置
     * 2. 计算加速度计和陀螺仪的噪声协方差
     * 3. 初始化卡尔曼滤波器的状态和协方差矩阵
     */
    void ImuProcess::IMUInit(const common::MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state,
                            int &N)
    {
        /** 1. 初始化重力向量、陀螺仪偏置、加速度计和陀螺仪协方差
         ** 2. 将加速度测量值归一化到单位重力加速度 **/

        // 当前IMU测量值的临时存储
        common::V3D cur_acc, cur_gyr;

        // 如果是第一帧数据，进行初始化设置
        if (b_first_frame_)
        {
            Reset();               // 重置所有累积变量
            N = 1;                // 初始化计数器
            b_first_frame_ = false; // 标记不再是第一帧

            // 获取第一个IMU测量值作为初始估计
            const auto &imu_acc = meas.imu_.front()->linear_acceleration;
            const auto &gyr_acc = meas.imu_.front()->angular_velocity;
            
            // 初始化加速度和角速度的均值估计
            mean_acc_ << imu_acc.x, imu_acc.y, imu_acc.z;
            mean_gyr_ << gyr_acc.x, gyr_acc.y, gyr_acc.z;
        }

        // 遍历当前测量组中的所有IMU数据，进行统计估计
        for (const auto &imu : meas.imu_)
        {
            // 提取当前IMU测量值
            const auto &imu_acc = imu->linear_acceleration;
            const auto &gyr_acc = imu->angular_velocity;
            cur_acc << imu_acc.x, imu_acc.y, imu_acc.z;
            cur_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;

            // 使用递增式均值计算公式更新均值估计
            // 新均值 = 旧均值 + (新样本 - 旧均值) / 样本数
            mean_acc_ += (cur_acc - mean_acc_) / N;
            mean_gyr_ += (cur_gyr - mean_gyr_) / N;

            // 使用递增式协方差计算公式更新协方差估计
            // 这是Welford's online算法的变体，用于在线计算方差
            cov_acc_ =
                cov_acc_ * (N - 1.0) / N + (cur_acc - mean_acc_).cwiseProduct(cur_acc - mean_acc_) * (N - 1.0) / (N * N);
            cov_gyr_ =
                cov_gyr_ * (N - 1.0) / N + (cur_gyr - mean_gyr_).cwiseProduct(cur_gyr - mean_gyr_) * (N - 1.0) / (N * N);

            // 增加样本计数
            N++;
        }
        
        // 获取当前卡尔曼滤波器状态进行初始化
        state_ikfom init_state = kf_state.get_x();
        
        // 估计重力向量：假设静止时加速度计测量值主要是重力
        // 将平均加速度归一化并乘以标准重力加速度，然后取负值（因为重力方向向下）
        // S2表示将向量投影到单位球面上
        init_state.grav = S2(-mean_acc_ / mean_acc_.norm() * common::G_m_s2);

        // 设置陀螺仪偏置为平均角速度（假设初始化期间载体静止）
        init_state.bg = mean_gyr_;
        
        // 设置激光雷达相对于IMU的外参
        init_state.offset_T_L_I = Lidar_T_wrt_IMU_;  // 平移外参
        init_state.offset_R_L_I = Lidar_R_wrt_IMU_;  // 旋转外参
        
        // 更新卡尔曼滤波器的状态
        kf_state.change_x(init_state);

        // 初始化协方差矩阵
        esekfom::esekf<state_ikfom, 12, input_ikfom>::cov init_P = kf_state.get_P();
        init_P.setIdentity();  // 先设置为单位矩阵
        
        // 设置各状态分量的初始不确定性
        init_P(0, 0) = init_P(1, 1) = init_P(2, 2) = 0.0001;           // 旋转状态的初始方差（较小，假设姿态相对准确）
        init_P(3, 3) = init_P(4, 4) = init_P(5, 5) = 0.000001;         // 速度状态的初始方差（很小，假设初始静止）
        init_P(6, 6) = init_P(7, 7) = init_P(8, 8) = 0.00001;          // 位置状态的初始方差
        init_P(9, 9) = init_P(10, 10) = init_P(11, 11) = 0.00001;      // 加速度计偏置的初始方差
        init_P(15, 15) = init_P(16, 16) = init_P(17, 17) = 0.0001;     // 陀螺仪偏置的初始方差
        init_P(18, 18) = init_P(19, 19) = init_P(20, 20) = 0.001;      // 重力状态的初始方差
        init_P(21, 21) = init_P(22, 22) = 0.00001;                     // 外参的初始方差
        
        // 更新卡尔曼滤波器的协方差矩阵
        kf_state.change_P(init_P);
        
        // 保存最后一个IMU测量值，用于下次处理
        last_imu_ = meas.imu_.back();

        // 初始化噪声估计矩阵，使用预设的标定参数
        est_noise_.block<3, 3>(0, 0) = cov_gyr_scale_.asDiagonal();    // 陀螺仪噪声协方差
        est_noise_.block<3, 3>(3, 0) = cov_acc_scale_.asDiagonal();    // 加速度计噪声协方差
        est_noise_.block<3, 3>(6, 0) = cov_bias_gyr_.asDiagonal();     // 陀螺仪偏置随机游走协方差
        est_noise_.block<3, 3>(9, 0) = cov_bias_acc_.asDiagonal();     // 加速度计偏置随机游走协方差
    }

    /**
     * [功能描述]：使用IMU数据对激光雷达点云进行运动去畸变处理
     * @param meas：测量组，包含IMU数据队列和激光雷达点云
     * @param kf_state：扩展卡尔曼滤波器状态，用于IMU状态预测和更新
     * @param pcl_out：输出参数，存储去畸变后的点云数据
     * 
     * 该函数通过IMU预积分和运动补偿，消除激光雷达扫描期间载体运动造成的点云畸变
     * 核心思想：将不同时刻采集的激光点统一到同一个时刻的坐标系下
     */
    void ImuProcess::UndistortPcl(const common::MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state,
                                PointCloudType &pcl_out)
    {
        /*** 将上一帧的最后一个IMU数据添加到当前帧的开头，确保时间连续性 ***/
        auto v_imu = meas.imu_;
        v_imu.push_front(last_imu_);  // 添加上一帧尾部IMU数据到当前帧头部
        
        // 获取关键时间戳
        const double &imu_beg_time = v_imu.front()->header.stamp.toSec();  // IMU开始时间
        const double &imu_end_time = v_imu.back()->header.stamp.toSec();   // IMU结束时间
        const double &pcl_beg_time = meas.lidar_bag_time_;                 // 点云开始时间
        const double &pcl_end_time = meas.lidar_end_time_;                 // 点云结束时间

        /*** 按时间偏移对点云进行排序 ***/
        pcl_out = *(meas.lidar_);
        // 使用time_list比较函数按时间戳排序，确保点云按采集时间顺序排列
        sort(pcl_out.points.begin(), pcl_out.points.end(), time_list);

        /*** 初始化IMU位姿 ***/
        state_ikfom imu_state = kf_state.get_x();  // 获取当前卡尔曼滤波状态
        IMUpose_.clear();  // 清空IMU位姿历史记录
        
        // 设置初始IMU位姿（时间偏移为0，即点云开始时刻）
        IMUpose_.push_back(common::set_pose6d(0.0, acc_s_last_, angvel_last_, imu_state.vel, imu_state.pos,
                                            imu_state.rot.toRotationMatrix()));

        /*** 在每个IMU测量点进行前向传播 ***/
        common::V3D angvel_avr, acc_avr, acc_imu, vel_imu, pos_imu;  // 平均角速度、加速度等
        common::M3D R_imu;  // IMU旋转矩阵
        double dt = 0;      // 时间间隔

        // 自适应噪声估计部分
        if (last_dt_square_sum > 0)
        {
            // 当前噪声和平均噪声矩阵
            Eigen::Matrix<double, 12, 3> cur_noise, avg_noise;
            auto fw_t = kf_state.last_fw_total;  // 前向传播雅可比矩阵

            // 获取上次状态增量
            Eigen::Matrix<double, 23, 1> dx_m;
            auto dx = kf_state.last_dx;
            dx_m = dx.block<23, 1>(0, 0);

            // 计算状态增量的外积，用于噪声协方差估计
            Eigen::Matrix<double, 23, 23> dxdxt;
            dxdxt = dx_m * dx_m.transpose();
            auto new_AQ = dxdxt;

            // 计算各状态分量的噪声协方差
            // 通过雅可比矩阵的逆变换将状态空间的噪声映射到传感器测量空间
            cur_noise.block<3, 3>(0, 0) = (fw_t.block(3, 0, 3, 3).inverse() * new_AQ.block<3, 3>(3, 3) *
                                        fw_t.block(3, 0, 3, 3).transpose().inverse());       // 旋转噪声
            cur_noise.block<3, 3>(3, 0) = (fw_t.block(12, 3, 3, 3).inverse() * new_AQ.block<3, 3>(12, 12) *
                                        fw_t.block(12, 3, 3, 3).transpose().inverse());      // 速度噪声
            cur_noise.block<3, 3>(6, 0) = (fw_t.block(15, 6, 3, 3).inverse() * new_AQ.block<3, 3>(15, 15) *
                                        fw_t.block(15, 6, 3, 3).transpose().inverse());      // 位置噪声
            cur_noise.block<3, 3>(9, 0) = (fw_t.block(18, 9, 3, 3).inverse() * new_AQ.block<3, 3>(18, 18) *
                                        fw_t.block(18, 9, 3, 3).transpose().inverse());      // 偏置噪声

            // 使用指数平滑进行噪声估计更新
            auto last_noise = est_noise_;
            est_noise_ = (1 - alpha) * cur_noise + alpha * last_noise;
        }
        
        // 更新过程噪声协方差矩阵Q
        Q_.block<3, 3>(0, 0) = est_noise_.block<3, 3>(0, 0);  // 角速度噪声
        Q_.block<3, 3>(3, 3) = est_noise_.block<3, 3>(3, 0);  // 加速度噪声
        Q_.block<3, 3>(6, 6) = est_noise_.block<3, 3>(6, 0);  // 角速度偏置噪声
        Q_.block<3, 3>(9, 9) = est_noise_.block<3, 3>(9, 0);  // 加速度偏置噪声

        // 重置累积变量
        last_dt_square_sum = 0;
        kf_state.last_fw_total.setZero();

        // IMU前向积分过程
        input_ikfom in;  // 卡尔曼滤波输入
        
        // 遍历所有IMU测量对，进行数值积分
        for (auto it_imu = v_imu.begin(); it_imu < (v_imu.end() - 1); it_imu++)
        {
            auto &&head = *(it_imu);      // 当前IMU测量
            auto &&tail = *(it_imu + 1);  // 下一个IMU测量

            // 跳过早于上次激光雷达结束时间的IMU数据
            if (tail->header.stamp.toSec() < last_lidar_end_time_)
            {
                continue;
            }

            // 计算两个IMU测量之间的平均角速度（中点积分法）
            angvel_avr << 0.5 * (head->angular_velocity.x + tail->angular_velocity.x),
                0.5 * (head->angular_velocity.y + tail->angular_velocity.y),
                0.5 * (head->angular_velocity.z + tail->angular_velocity.z);
                
            // 计算两个IMU测量之间的平均线性加速度
            acc_avr << 0.5 * (head->linear_acceleration.x + tail->linear_acceleration.x),
                0.5 * (head->linear_acceleration.y + tail->linear_acceleration.y),
                0.5 * (head->linear_acceleration.z + tail->linear_acceleration.z);

            // 重力加速度标定：将加速度计测量值归一化到标准重力加速度
            acc_avr = acc_avr * common::G_m_s2 / mean_acc_.norm();

            // 计算时间间隔
            if (head->header.stamp.toSec() < last_lidar_end_time_)
            {
                // 如果头部时间早于上次激光雷达结束时间，只积分部分时间
                dt = tail->header.stamp.toSec() - last_lidar_end_time_;
            }
            else
            {
                // 正常情况下的时间间隔
                dt = tail->header.stamp.toSec() - head->header.stamp.toSec();
            }

            // 设置卡尔曼滤波输入
            in.acc = acc_avr;    // 加速度输入
            in.gyro = angvel_avr; // 角速度输入
            
            // 执行卡尔曼滤波预测步骤
            kf_state.predict(dt, Q_, in);

            // 累积时间间隔的平方和，用于噪声估计
            last_dt_square_sum += dt * dt;

            /* 保存每个IMU测量时刻的位姿 */
            imu_state = kf_state.get_x();  // 获取预测后的状态
            
            // 计算去偏置后的角速度和加速度
            angvel_last_ = angvel_avr - imu_state.bg;                     // 去除陀螺仪偏置
            acc_s_last_ = imu_state.rot * (acc_avr - imu_state.ba);      // 去除加速度计偏置并转换到世界坐标系
            
            // 添加重力分量
            for (int i = 0; i < 3; i++)
            {
                acc_s_last_[i] += imu_state.grav[i];
            }

            // 计算相对于点云开始时间的时间偏移
            double &&offs_t = tail->header.stamp.toSec() - pcl_beg_time;
            
            // 保存当前时刻的IMU位姿信息
            IMUpose_.emplace_back(common::set_pose6d(offs_t, acc_s_last_, angvel_last_, imu_state.vel, imu_state.pos,
                                                    imu_state.rot.toRotationMatrix()));
        }

        /*** 计算帧结束时刻的位置和姿态预测 ***/
        // 判断点云结束时间与IMU结束时间的关系，确定预测方向
        double note = pcl_end_time > imu_end_time ? 1.0 : -1.0;
        dt = note * (pcl_end_time - imu_end_time);
        
        // 预测到点云结束时刻的状态
        kf_state.predict(dt, Q_, in);
        last_dt_square_sum += dt * dt;

        // 更新状态变量
        imu_state = kf_state.get_x();
        last_imu_ = meas.imu_.back();
        last_lidar_end_time_ = pcl_end_time;

        /*** 对每个激光雷达点进行去畸变（后向传播） ***/
        if (pcl_out.points.empty())
        {
            return;  // 如果点云为空，直接返回
        }
        
        // 从点云末尾开始，逆时间顺序处理每个点
        auto it_pcl = pcl_out.points.end() - 1;
        
        // 遍历IMU位姿历史，从后向前处理
        for (auto it_kp = IMUpose_.end() - 1; it_kp != IMUpose_.begin(); it_kp--)
        {
            auto head = it_kp - 1;  // 时间段开始的IMU位姿
            auto tail = it_kp;      // 时间段结束的IMU位姿
            
            // 提取IMU状态信息
            R_imu = common::MatFromArray(head->rot);    // 旋转矩阵
            vel_imu = common::VecFromArray(head->vel);  // 速度
            pos_imu = common::VecFromArray(head->pos);  // 位置
            acc_imu = common::VecFromArray(tail->acc);  // 加速度
            angvel_avr = common::VecFromArray(tail->gyr); // 角速度

            // 处理当前时间段内的所有激光点
            for (; it_pcl->curvature / double(1000) > head->offset_time; it_pcl--)
            {
                // 计算点相对于时间段开始的时间偏移
                dt = it_pcl->curvature / double(1000) - head->offset_time;

                /* 将点转换到'end'帧，仅使用旋转
                * 注意：补偿方向与载体运动方向相反
                * 如果要将时刻i的点p_i补偿到帧结束时刻e
                * p_compensate = R_imu_e^T * (R_i * P_i + T_ei)，其中T_ei在全局坐标系中表示 */
                
                // 计算时刻i的旋转矩阵
                common::M3D R_i(R_imu * Exp(angvel_avr, dt));

                // 原始激光点坐标
                common::V3D P_i(it_pcl->x, it_pcl->y, it_pcl->z);
                
                // 计算从时刻i到帧结束时刻的平移向量
                common::V3D T_ei(pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt - imu_state.pos);
                
                // 执行运动补偿变换
                // 1. 将激光点从激光雷达坐标系转换到IMU坐标系
                // 2. 应用时刻i的旋转和平移
                // 3. 减去时刻i到帧结束的运动
                // 4. 转换回激光雷达坐标系
                common::V3D p_compensate =
                    imu_state.offset_R_L_I.conjugate() *
                    (imu_state.rot.conjugate() * (R_i * (imu_state.offset_R_L_I * P_i + imu_state.offset_T_L_I) + T_ei) -
                    imu_state.offset_T_L_I);

                // 保存去畸变后的点坐标
                it_pcl->x = p_compensate(0);
                it_pcl->y = p_compensate(1);
                it_pcl->z = p_compensate(2);

                // 如果已处理到点云开始，退出循环
                if (it_pcl == pcl_out.points.begin())
                {
                    break;
                }
            }
        }
    }

    /**
     * [功能描述]：处理包含IMU和激光雷达数据的测量组，执行IMU初始化或点云去畸变
     * @param meas：测量组，包含IMU数据队列和激光雷达点云数据
     * @param kf_state：扩展卡尔曼滤波器状态，用于状态估计和预测
     * @param cur_pcl_un_：输出参数，存储去畸变后的当前帧点云数据
     * 
     * 该函数是IMU-激光雷达融合SLAM系统的核心处理函数，负责：
     * 1. IMU的初始化和标定
     * 2. 基于IMU运动补偿的激光雷达点云去畸变
     */
    void ImuProcess::Process(const common::MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state,
                            PointCloudType::Ptr cur_pcl_un_)
    {
        // 检查IMU数据是否为空，如果为空则无法进行处理
        if (meas.imu_.empty())
        {
            return;
        }

        // 断言确保激光雷达数据不为空指针
        // 在调试模式下如果激光雷达数据为空会触发程序终止
        ROS_ASSERT(meas.lidar_ != nullptr);

        // 判断是否需要进行IMU初始化
        if (imu_need_init_)
        {
            /// 处理第一帧激光雷达数据时的IMU初始化
            // 调用IMU初始化函数，传入测量数据、滤波器状态和初始化迭代次数
            IMUInit(meas, kf_state, init_iter_num_);

            // 保持初始化标志为true，直到初始化完成
            imu_need_init_ = true;

            // 保存最后一个IMU测量值，用于下一次处理时的运动预测
            last_imu_ = meas.imu_.back();

            // 获取当前卡尔曼滤波器的状态估计
            state_ikfom imu_state = kf_state.get_x();
            
            // 检查初始化迭代次数是否超过最大限制
            if (init_iter_num_ > MAX_INI_COUNT)
            {
                // 根据重力加速度校正加速度计协方差
                // 使用实际重力加速度与测量平均加速度的比值进行校正
                cov_acc_ *= pow(common::G_m_s2 / mean_acc_.norm(), 2);
                
                // 标记IMU初始化完成
                imu_need_init_ = false;

                // 恢复加速度计和陀螺仪的标准协方差值
                cov_acc_ = cov_acc_scale_;
                cov_gyr_ = cov_gyr_scale_;
                
                // 输出初始化完成信息
                LOG(INFO) << "IMU Initial Done";
                
                // 打开IMU调试输出文件，用于记录IMU处理过程的调试信息
                fout_imu_.open(common::DEBUG_FILE_DIR("imu_.txt"), std::ios::out);
            }

            // IMU初始化阶段直接返回，不进行点云处理
            return;
        }

        // IMU初始化完成后，进行点云去畸变处理
        // 使用计时器评估点云去畸变的性能
        Timer::Evaluate(
            [&, this]()
            {
                // 调用点云去畸变函数
                // 使用IMU数据对激光雷达扫描过程中的运动进行补偿
                UndistortPcl(meas, kf_state, *cur_pcl_un_);
            },
            "Undistort Pcl"); // 计时器标识名称
    }
} // namespace akf_lio

#endif
