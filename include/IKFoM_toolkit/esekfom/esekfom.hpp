/*
 *  Copyright (c) 2019--2023, The University of Hong Kong
 *  All rights reserved.
 *
 *  Author: Dongjiao HE <hdj65822@connect.hku.hk>
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the Universitaet Bremen nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef ESEKFOM_EKF_HPP
#define ESEKFOM_EKF_HPP

#include <cstdlib>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/Sparse>
#include <boost/bind.hpp>

#include "../mtk/build_manifold.hpp"
#include "../mtk/startIdx.hpp"
#include "../mtk/types/S2.hpp"
#include "../mtk/types/SOn.hpp"
#include "../mtk/types/vect.hpp"
#include "util.hpp"

// #define USE_sparse

namespace esekfom
{

    using namespace Eigen;

    // used for iterated error state EKF update
    // for the aim to calculate  measurement (z), estimate measurement (h), partial differention matrices (h_x, h_v) and the
    // noise covariance (R) at the same time, by only one function. applied for measurement as a manifold.
    template <typename S, typename M, int measurement_noise_dof = M::DOF>
    struct share_datastruct
    {
        bool valid;
        bool converge;
        M z;
        Eigen::Matrix<typename S::scalar, M::DOF, measurement_noise_dof> h_v;
        Eigen::Matrix<typename S::scalar, M::DOF, S::DOF> h_x;
        Eigen::Matrix<typename S::scalar, measurement_noise_dof, measurement_noise_dof> R;
    };

    // used for iterated error state EKF update
    // for the aim to calculate  measurement (z), estimate measurement (h), partial differention matrices (h_x, h_v) and the
    // noise covariance (R) at the same time, by only one function. applied for measurement as an Eigen matrix whose
    // dimension is changing
    template <typename T>
    struct dyn_share_datastruct
    {
        bool valid;
        bool converge;
        Eigen::Matrix<T, Eigen::Dynamic, 1> z;
        Eigen::Matrix<T, Eigen::Dynamic, 1> h;
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> h_v;
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> h_x;
        Eigen::Matrix<T, Eigen::Dynamic, 1> R;
        Eigen::Matrix<T, Eigen::Dynamic, 1> R_inv;
    };

    // used for iterated error state EKF update
    // for the aim to calculate  measurement (z), estimate measurement (h), partial differention matrices (h_x, h_v) and the
    // noise covariance (R) at the same time, by only one function. applied for measurement as a dynamic manifold whose
    // dimension or type is changing
    template <typename T>
    struct dyn_runtime_share_datastruct
    {
        bool valid;
        bool converge;
        // Z z;
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> h_v;
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> h_x;
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> R;
    };

    template <typename state, int process_noise_dof, typename input = state, typename measurement = state,
              int measurement_noise_dof = 0>
    class esekf
    {
        typedef esekf self;

        enum
        {
            n = state::DOF,
            m = state::DIM,
            l = measurement::DOF
        };

    public:
        bool global_converge = false;
        typedef typename state::scalar scalar_type;
        typedef Matrix<scalar_type, n, n> cov;
        typedef Matrix<scalar_type, m, n> cov_;
        typedef SparseMatrix<scalar_type> spMt;
        typedef Matrix<scalar_type, n, 1> vectorized_state;
        typedef Matrix<scalar_type, m, 1> flatted_state;
        typedef flatted_state processModel(state &, const input &);
        typedef Eigen::Matrix<scalar_type, m, n> processMatrix1(state &, const input &);
        typedef Eigen::Matrix<scalar_type, m, process_noise_dof> processMatrix2(state &, const input &);
        typedef Eigen::Matrix<scalar_type, process_noise_dof, process_noise_dof> processnoisecovariance;
        typedef measurement measurementModel(state &, bool &);
        typedef measurement measurementModel_share(state &, share_datastruct<state, measurement, measurement_noise_dof> &);
        typedef Eigen::Matrix<scalar_type, Eigen::Dynamic, 1> measurementModel_dyn(state &, bool &);
        using measurementModel_dyn_share = std::function<void(state &, dyn_share_datastruct<scalar_type> &)>;

        typedef Eigen::Matrix<scalar_type, l, n> measurementMatrix1(state &, bool &);
        typedef Eigen::Matrix<scalar_type, Eigen::Dynamic, n> measurementMatrix1_dyn(state &, bool &);
        typedef Eigen::Matrix<scalar_type, l, measurement_noise_dof> measurementMatrix2(state &, bool &);
        typedef Eigen::Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic> measurementMatrix2_dyn(state &, bool &);
        typedef Eigen::Matrix<scalar_type, measurement_noise_dof, measurement_noise_dof> measurementnoisecovariance;
        typedef Eigen::Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic> measurementnoisecovariance_dyn;

        esekf(const state &x = state(), const cov &P = cov::Identity()) : x_(x), P_(P)
        {
#ifdef USE_sparse
            SparseMatrix<scalar_type> ref(n, n);
            ref.setIdentity();
            l_ = ref;
            f_x_2 = ref;
            f_x_1 = ref;
#endif
        };

        // receive system-specific models and their differentions.
        // for measurement as a manifold.
        void init(processModel f_in, processMatrix1 f_x_in, processMatrix2 f_w_in, measurementModel h_in,
                  measurementMatrix1 h_x_in, measurementMatrix2 h_v_in, int maximum_iteration,
                  scalar_type limit_vector[n])
        {
            f = f_in;
            f_x = f_x_in;
            f_w = f_w_in;
            h = h_in;
            h_x = h_x_in;
            h_v = h_v_in;

            maximum_iter = maximum_iteration;
            for (int i = 0; i < n; i++)
            {
                limit[i] = limit_vector[i];
            }

            x_.build_S2_state();
            x_.build_SO3_state();
            x_.build_vect_state();
        }

        // receive system-specific models and their differentions.
        // for measurement as an Eigen matrix whose dimention is chaing.
        void init_dyn(processModel f_in, processMatrix1 f_x_in, processMatrix2 f_w_in, measurementModel_dyn h_in,
                      measurementMatrix1_dyn h_x_in, measurementMatrix2_dyn h_v_in, int maximum_iteration,
                      scalar_type limit_vector[n])
        {
            f = f_in;
            f_x = f_x_in;
            f_w = f_w_in;
            h_dyn = h_in;
            h_x_dyn = h_x_in;
            h_v_dyn = h_v_in;

            maximum_iter = maximum_iteration;
            for (int i = 0; i < n; i++)
            {
                limit[i] = limit_vector[i];
            }

            x_.build_S2_state();
            x_.build_SO3_state();
            x_.build_vect_state();
        }

        // receive system-specific models and their differentions.
        // for measurement as a dynamic manifold whose dimension or type is changing.
        void init_dyn_runtime(processModel f_in, processMatrix1 f_x_in, processMatrix2 f_w_in,
                              measurementMatrix1_dyn h_x_in, measurementMatrix2_dyn h_v_in, int maximum_iteration,
                              scalar_type limit_vector[n])
        {
            f = f_in;
            f_x = f_x_in;
            f_w = f_w_in;
            h_x_dyn = h_x_in;
            h_v_dyn = h_v_in;

            maximum_iter = maximum_iteration;
            for (int i = 0; i < n; i++)
            {
                limit[i] = limit_vector[i];
            }

            x_.build_S2_state();
            x_.build_SO3_state();
            x_.build_vect_state();
        }

        // receive system-specific models and their differentions
        // for measurement as a manifold.
        // calculate  measurement (z), estimate measurement (h), partial differention matrices (h_x, h_v) and the noise
        // covariance (R) at the same time, by only one function (h_share_in).
        void init_share(processModel f_in, processMatrix1 f_x_in, processMatrix2 f_w_in, measurementModel_share h_share_in,
                        int maximum_iteration, scalar_type limit_vector[n])
        {
            f = f_in;
            f_x = f_x_in;
            f_w = f_w_in;
            h_share = h_share_in;

            maximum_iter = maximum_iteration;
            for (int i = 0; i < n; i++)
            {
                limit[i] = limit_vector[i];
            }

            x_.build_S2_state();
            x_.build_SO3_state();
            x_.build_vect_state();
        }

        // receive system-specific models and their differentions
        // for measurement as an Eigen matrix whose dimension is changing.
        // calculate  measurement (z), estimate measurement (h), partial differention matrices (h_x, h_v) and the noise
        // covariance (R) at the same time, by only one function (h_dyn_share_in).
        void init_dyn_share(processModel f_in, processMatrix1 f_x_in, processMatrix2 f_w_in,
                            measurementModel_dyn_share h_dyn_share_in, int maximum_iteration, scalar_type limit_vector[n])
        {
            f = f_in;
            f_x = f_x_in;
            f_w = f_w_in;
            h_dyn_share = h_dyn_share_in;

            maximum_iter = maximum_iteration;
            for (int i = 0; i < n; i++)
            {
                limit[i] = limit_vector[i];
            }

            x_.build_S2_state();
            x_.build_SO3_state();
            x_.build_vect_state();
            last_dx.setZero();
            last_fw_total.setZero();
        }

        // receive system-specific models and their differentions
        // for measurement as a dynamic manifold whose dimension  or type is changing.
        // calculate  measurement (z), estimate measurement (h), partial differention matrices (h_x, h_v) and the noise
        // covariance (R) at the same time, by only one function (h_dyn_share_in). for any scenarios where it is needed
        void init_dyn_runtime_share(processModel f_in, processMatrix1 f_x_in, processMatrix2 f_w_in, int maximum_iteration,
                                    scalar_type limit_vector[n])
        {
            f = f_in;
            f_x = f_x_in;
            f_w = f_w_in;

            maximum_iter = maximum_iteration;
            for (int i = 0; i < n; i++)
            {
                limit[i] = limit_vector[i];
            }

            x_.build_S2_state();
            x_.build_SO3_state();
            x_.build_vect_state();
        }

        // iterated error state EKF propogation
        /**
         * [功能描述]：执行ESEKF（误差状态扩展卡尔曼滤波）的预测步骤
         * @param dt：时间步长，用于状态积分
         * @param Q：过程噪声协方差矩阵
         * @param i_in：输入数据（通常为IMU测量值）
         * 
         * 该函数在流形空间上执行状态预测和协方差传播，支持多种流形结构：
         * - 欧几里得空间的向量状态
         * - SO(3)旋转群（四元数/旋转矩阵）
         * - S2球面流形
         */
        void predict(double &dt, processnoisecovariance &Q, const input &i_in)
        {
            // 计算状态转移函数 f(x,u)，返回状态的导数
            flatted_state f_ = f(x_, i_in);
            
            // 计算状态转移函数对状态的雅可比矩阵 ∂f/∂x
            cov_ f_x_ = f_x(x_, i_in);
            cov f_x_final;  // 最终的状态雅可比矩阵

            // 计算状态转移函数对过程噪声的雅可比矩阵 ∂f/∂w
            Matrix<scalar_type, m, process_noise_dof> f_w_ = f_w(x_, i_in);
            Matrix<scalar_type, n, process_noise_dof> f_w_final;  // 最终的噪声雅可比矩阵
            
            // 保存更新前的状态，用于流形运算
            state x_before = x_;
            
            // 在流形上执行状态更新：x = x ⊕ (f * dt)
            // oplus运算符实现流形上的"加法"操作
            x_.oplus(f_, dt);

            // 初始化状态转移矩阵为单位矩阵
            F_x1 = cov::Identity();
            
            // 处理欧几里得向量状态分量
            for (std::vector<std::pair<std::pair<int, int>, int>>::iterator it = x_.vect_state.begin();
                it != x_.vect_state.end(); it++)
            {
                int idx = (*it).first.first;   // 在状态向量中的起始索引
                int dim = (*it).first.second;  // 在扁平化状态中的起始位置
                int dof = (*it).second;        // 自由度数量
                
                // 复制状态雅可比矩阵的对应块
                for (int i = 0; i < n; i++)
                {
                    for (int j = 0; j < dof; j++)
                    {
                        f_x_final(idx + j, i) = f_x_(dim + j, i);
                    }
                }
                
                // 复制噪声雅可比矩阵的对应块
                for (int i = 0; i < process_noise_dof; i++)
                {
                    for (int j = 0; j < dof; j++)
                    {
                        f_w_final(idx + j, i) = f_w_(dim + j, i);
                    }
                }
            }
            
            // 处理SO(3)旋转群状态分量
            Matrix<scalar_type, 3, 3> res_temp_SO3;  // SO(3)临时变换矩阵
            MTK::vect<3, scalar_type> seg_SO3;       // SO(3)状态分量
            
            for (std::vector<std::pair<int, int>>::iterator it = x_.SO3_state.begin(); it != x_.SO3_state.end(); it++)
            {
                int idx = (*it).first;  // 在状态向量中的索引
                int dim = (*it).second; // 在扁平化状态中的索引
                
                // 提取SO(3)状态增量（注意负号，用于正确的流形运算）
                for (int i = 0; i < 3; i++)
                {
                    seg_SO3(i) = -1 * f_(dim + i) * dt;
                }
                
                // 计算SO(3)指数映射：exp(ω * dt)
                MTK::SO3<scalar_type> res;
                res.w() = MTK::exp<scalar_type, 3>(res.vec(), seg_SO3, scalar_type(1 / 2));
                
#ifdef USE_sparse
                // 稀疏矩阵版本：直接设置系数
                res_temp_SO3 = res.toRotationMatrix();
                for (int i = 0; i < 3; i++)
                {
                    for (int j = 0; j < 3; j++)
                    {
                        f_x_1.coeffRef(idx + i, idx + j) = res_temp_SO3(i, j);
                    }
                }
#else
                // 密集矩阵版本：块操作
                F_x1.template block<3, 3>(idx, idx) = res.toRotationMatrix();
#endif
                
                // 计算SO(3)的左雅可比矩阵
                res_temp_SO3 = MTK::A_matrix(seg_SO3);
                
                // 更新状态雅可比矩阵的SO(3)部分
                for (int i = 0; i < n; i++)
                {
                    f_x_final.template block<3, 1>(idx, i) = res_temp_SO3 * (f_x_.template block<3, 1>(dim, i));
                }
                
                // 更新噪声雅可比矩阵的SO(3)部分
                for (int i = 0; i < process_noise_dof; i++)
                {
                    f_w_final.template block<3, 1>(idx, i) = res_temp_SO3 * (f_w_.template block<3, 1>(dim, i));
                }
            }

            // 处理S2球面流形状态分量
            Matrix<scalar_type, 2, 3> res_temp_S2;   // S2临时变换矩阵（2×3）
            Matrix<scalar_type, 2, 2> res_temp_S2_;  // S2临时变换矩阵（2×2）
            MTK::vect<3, scalar_type> seg_S2;        // S2状态分量（3D表示）
            
            for (std::vector<std::pair<int, int>>::iterator it = x_.S2_state.begin(); it != x_.S2_state.end(); it++)
            {
                int idx = (*it).first;  // 在状态向量中的索引
                int dim = (*it).second; // 在扁平化状态中的索引
                
                // 提取S2状态增量
                for (int i = 0; i < 3; i++)
                {
                    seg_S2(i) = f_(dim + i) * dt;
                }
                
                // 计算S2流形上的更新
                MTK::vect<2, scalar_type> vec = MTK::vect<2, scalar_type>::Zero();
                MTK::SO3<scalar_type> res;
                res.w() = MTK::exp<scalar_type, 3>(res.vec(), seg_S2, scalar_type(1 / 2));
                
                // 计算S2球面的切空间基向量
                Eigen::Matrix<scalar_type, 2, 3> Nx;  // 当前状态的切空间基
                Eigen::Matrix<scalar_type, 3, 2> Mx;  // 更新前状态的切空间基
                x_.S2_Nx_yy(Nx, idx);                 // 计算当前状态的切空间
                x_before.S2_Mx(Mx, vec, idx);         // 计算更新前状态的切空间

#ifdef USE_sparse
                // 稀疏矩阵版本
                res_temp_S2_ = Nx * res.toRotationMatrix() * Mx;
                for (int i = 0; i < 2; i++)
                {
                    for (int j = 0; j < 2; j++)
                    {
                        f_x_1.coeffRef(idx + i, idx + j) = res_temp_S2_(i, j);
                    }
                }
#else
                // 密集矩阵版本
                F_x1.template block<2, 2>(idx, idx) = Nx * res.toRotationMatrix() * Mx;
#endif

                // 计算S2状态的雅可比变换
                Eigen::Matrix<scalar_type, 3, 3> x_before_hat;
                x_before.S2_hat(x_before_hat, idx);  // 计算反对称矩阵
                res_temp_S2 = -Nx * res.toRotationMatrix() * x_before_hat * MTK::A_matrix(seg_S2).transpose();

                // 更新状态雅可比矩阵的S2部分
                for (int i = 0; i < n; i++)
                {
                    f_x_final.template block<2, 1>(idx, i) = res_temp_S2 * (f_x_.template block<3, 1>(dim, i));
                }
                
                // 更新噪声雅可比矩阵的S2部分
                for (int i = 0; i < process_noise_dof; i++)
                {
                    f_w_final.template block<2, 1>(idx, i) = res_temp_S2 * (f_w_.template block<3, 1>(dim, i));
                }
            }

            // 执行协方差传播
#ifdef USE_sparse
            // 稀疏矩阵版本的协方差更新
            f_x_1.makeCompressed();                          // 压缩稀疏矩阵
            spMt f_x2 = f_x_final.sparseView();             // 转换为稀疏视图
            spMt f_w1 = f_w_final.sparseView();             // 转换为稀疏视图
            spMt xp = f_x_1 + f_x2 * dt;                    // 计算总的状态转移矩阵
            // 协方差传播：P = F * P * F^T + G * Q * G^T
            P_ = xp * P_ * xp.transpose() + (f_w1 * dt) * Q * (f_w1 * dt).transpose();
#else
            // 密集矩阵版本的协方差更新
            F_x1 += f_x_final * dt;  // 计算总的状态转移矩阵
            
            // 执行协方差传播：P = F * P * F^T + G * Q * G^T
            P_ = (F_x1)*P_ * (F_x1).transpose() + (dt * f_w_final) * Q * (dt * f_w_final).transpose();
            
            // 累积前向传播雅可比矩阵，用于自适应噪声估计
            Matrix<scalar_type, n, process_noise_dof> cur_fw_total;
            cur_fw_total = F_x1 * last_fw_total + dt * f_w_final;
            last_fw_total = cur_fw_total;
#endif
        }

        // iterated error state EKF update for measurement as a manifold.
        void update_iterated(measurement &z, measurementnoisecovariance &R)
        {
            if (!(is_same<typename measurement::scalar, scalar_type>()))
            {
                std::cerr << "the scalar type of measurment must be the same as the state" << std::endl;
                std::exit(100);
            }
            int t = 0;
            bool converg = true;
            bool valid = true;
            state x_propagated = x_;
            cov P_propagated = P_;

            for (int i = -1; i < maximum_iter; i++)
            {
                vectorized_state dx, dx_new;
                x_.boxminus(dx, x_propagated);
                dx_new = dx;
#ifdef USE_sparse
                spMt h_x_ = h_x(x_, valid).sparseView();
                spMt h_v_ = h_v(x_, valid).sparseView();
                spMt R_ = R.sparseView();
#else
                Matrix<scalar_type, l, n> h_x_ = h_x(x_, valid);
                Matrix<scalar_type, l, Eigen::Dynamic> h_v_ = h_v(x_, valid);
#endif
                if (!valid)
                {
                    continue;
                }

                P_ = P_propagated;

                Matrix<scalar_type, 3, 3> res_temp_SO3;
                MTK::vect<3, scalar_type> seg_SO3;
                for (std::vector<std::pair<int, int>>::iterator it = x_.SO3_state.begin(); it != x_.SO3_state.end(); it++)
                {
                    int idx = (*it).first;
                    int dim = (*it).second;
                    for (int i = 0; i < 3; i++)
                    {
                        seg_SO3(i) = dx(idx + i);
                    }

                    res_temp_SO3 = A_matrix(seg_SO3).transpose();
                    dx_new.template block<3, 1>(idx, 0) = res_temp_SO3 * dx.template block<3, 1>(idx, 0);
                    for (int i = 0; i < n; i++)
                    {
                        P_.template block<3, 1>(idx, i) = res_temp_SO3 * (P_.template block<3, 1>(idx, i));
                    }
                    for (int i = 0; i < n; i++)
                    {
                        P_.template block<1, 3>(i, idx) = (P_.template block<1, 3>(i, idx)) * res_temp_SO3.transpose();
                    }
                }

                Matrix<scalar_type, 2, 2> res_temp_S2;
                MTK::vect<2, scalar_type> seg_S2;
                for (std::vector<std::pair<int, int>>::iterator it = x_.S2_state.begin(); it != x_.S2_state.end(); it++)
                {
                    int idx = (*it).first;
                    int dim = (*it).second;
                    for (int i = 0; i < 2; i++)
                    {
                        seg_S2(i) = dx(idx + i);
                    }

                    Eigen::Matrix<scalar_type, 2, 3> Nx;
                    Eigen::Matrix<scalar_type, 3, 2> Mx;
                    x_.S2_Nx_yy(Nx, idx);
                    x_propagated.S2_Mx(Mx, seg_S2, idx);
                    res_temp_S2 = Nx * Mx;
                    dx_new.template block<2, 1>(idx, 0) = res_temp_S2 * dx.template block<2, 1>(idx, 0);
                    for (int i = 0; i < n; i++)
                    {
                        P_.template block<2, 1>(idx, i) = res_temp_S2 * (P_.template block<2, 1>(idx, i));
                    }
                    for (int i = 0; i < n; i++)
                    {
                        P_.template block<1, 2>(i, idx) = (P_.template block<1, 2>(i, idx)) * res_temp_S2.transpose();
                    }
                }

                Matrix<scalar_type, n, l> K_;
                if (n > l)
                {
#ifdef USE_sparse
                    Matrix<scalar_type, l, l> K_temp = h_x_ * P_ * h_x_.transpose();
                    spMt R_temp = h_v_ * R_ * h_v_.transpose();
                    K_temp += R_temp;
                    K_ = P_ * h_x_.transpose() * K_temp.inverse();
#else
                    K_ = P_ * h_x_.transpose() * (h_x_ * P_ * h_x_.transpose() + h_v_ * R * h_v_.transpose()).inverse();
#endif
                }
                else
                {
#ifdef USE_sparse
                    measurementnoisecovariance b = measurementnoisecovariance::Identity();
                    Eigen::SparseQR<Eigen::SparseMatrix<scalar_type>, Eigen::COLAMDOrdering<int>> solver;
                    solver.compute(R_);
                    measurementnoisecovariance R_in_temp = solver.solve(b);
                    spMt R_in = R_in_temp.sparseView();
                    spMt K_temp = h_x_.transpose() * R_in * h_x_;
                    cov_ P_temp = P_.inverse();
                    P_temp += K_temp;
                    K_ = P_temp.inverse() * h_x_.transpose() * R_in;
#else
                    measurementnoisecovariance R_in = (h_v_ * R * h_v_.transpose()).inverse();
                    K_ = (h_x_.transpose() * R_in * h_x_ + P_.inverse()).inverse() * h_x_.transpose() * R_in;
#endif
                }
                Matrix<scalar_type, l, 1> innovation;
                z.boxminus(innovation, h(x_, valid));
                cov K_x = K_ * h_x_;
                Matrix<scalar_type, n, 1> dx_ = K_ * innovation + (K_x - Matrix<scalar_type, n, n>::Identity()) * dx_new;
                state x_before = x_;
                x_.boxplus(dx_);

                converg = true;
                for (int i = 0; i < n; i++)
                {
                    if (std::fabs(dx_[i]) > limit[i])
                    {
                        converg = false;
                        break;
                    }
                }

                if (converg)
                    t++;

                if (t > 1 || i == maximum_iter - 1)
                {
                    L_ = P_;

                    Matrix<scalar_type, 3, 3> res_temp_SO3;
                    MTK::vect<3, scalar_type> seg_SO3;
                    for (typename std::vector<std::pair<int, int>>::iterator it = x_.SO3_state.begin();
                         it != x_.SO3_state.end(); it++)
                    {
                        int idx = (*it).first;
                        for (int i = 0; i < 3; i++)
                        {
                            seg_SO3(i) = dx_(i + idx);
                        }
                        res_temp_SO3 = A_matrix(seg_SO3).transpose();
                        for (int i = 0; i < n; i++)
                        {
                            L_.template block<3, 1>(idx, i) = res_temp_SO3 * (P_.template block<3, 1>(idx, i));
                        }
                        if (n > l)
                        {
                            for (int i = 0; i < l; i++)
                            {
                                K_.template block<3, 1>(idx, i) = res_temp_SO3 * (K_.template block<3, 1>(idx, i));
                            }
                        }
                        else
                        {
                            for (int i = 0; i < n; i++)
                            {
                                K_x.template block<3, 1>(idx, i) = res_temp_SO3 * (K_x.template block<3, 1>(idx, i));
                            }
                        }
                        for (int i = 0; i < n; i++)
                        {
                            L_.template block<1, 3>(i, idx) = (L_.template block<1, 3>(i, idx)) * res_temp_SO3.transpose();
                            P_.template block<1, 3>(i, idx) = (P_.template block<1, 3>(i, idx)) * res_temp_SO3.transpose();
                        }
                    }

                    Matrix<scalar_type, 2, 2> res_temp_S2;
                    MTK::vect<2, scalar_type> seg_S2;
                    for (typename std::vector<std::pair<int, int>>::iterator it = x_.S2_state.begin();
                         it != x_.S2_state.end(); it++)
                    {
                        int idx = (*it).first;

                        for (int i = 0; i < 2; i++)
                        {
                            seg_S2(i) = dx_(i + idx);
                        }

                        Eigen::Matrix<scalar_type, 2, 3> Nx;
                        Eigen::Matrix<scalar_type, 3, 2> Mx;
                        x_.S2_Nx_yy(Nx, idx);
                        x_propagated.S2_Mx(Mx, seg_S2, idx);
                        res_temp_S2 = Nx * Mx;

                        for (int i = 0; i < n; i++)
                        {
                            L_.template block<2, 1>(idx, i) = res_temp_S2 * (P_.template block<2, 1>(idx, i));
                        }
                        if (n > l)
                        {
                            for (int i = 0; i < l; i++)
                            {
                                K_.template block<2, 1>(idx, i) = res_temp_S2 * (K_.template block<2, 1>(idx, i));
                            }
                        }
                        else
                        {
                            for (int i = 0; i < n; i++)
                            {
                                K_x.template block<2, 1>(idx, i) = res_temp_S2 * (K_x.template block<2, 1>(idx, i));
                            }
                        }
                        for (int i = 0; i < n; i++)
                        {
                            L_.template block<1, 2>(i, idx) = (L_.template block<1, 2>(i, idx)) * res_temp_S2.transpose();
                            P_.template block<1, 2>(i, idx) = (P_.template block<1, 2>(i, idx)) * res_temp_S2.transpose();
                        }
                    }
                    if (n > l)
                    {
                        P_ = L_ - K_ * h_x_ * P_;
                    }
                    else
                    {
                        P_ = L_ - K_x * P_;
                    }
                    return;
                }
            }
        }

        // iterated error state EKF update for measurement as a manifold.
        // calculate measurement (z), estimate measurement (h), partial differention matrices (h_x, h_v) and the noise
        // covariance (R) at the same time, by only one function.
        void update_iterated_share()
        {
            if (!(is_same<typename measurement::scalar, scalar_type>()))
            {
                std::cerr << "the scalar type of measurment must be the same as the state" << std::endl;
                std::exit(100);
            }

            int t = 0;
            share_datastruct<state, measurement, measurement_noise_dof> _share;
            _share.valid = true;
            _share.converge = true;
            state x_propagated = x_;
            cov P_propagated = P_;

            for (int i = -1; i < maximum_iter; i++)
            {
                vectorized_state dx, dx_new;
                x_.boxminus(dx, x_propagated);
                dx_new = dx;
                measurement h = h_share(x_, _share);
                measurement z = _share.z;
                measurementnoisecovariance R = _share.R;
#ifdef USE_sparse
                spMt h_x_ = _share.h_x.sparseView();
                spMt h_v_ = _share.h_v.sparseView();
                spMt R_ = _share.R.sparseView();
#else
                Matrix<scalar_type, l, n> h_x_ = _share.h_x;
                Matrix<scalar_type, l, Eigen::Dynamic> h_v_ = _share.h_v;
#endif
                if (!_share.valid)
                {
                    continue;
                }

                P_ = P_propagated;

                Matrix<scalar_type, 3, 3> res_temp_SO3;
                MTK::vect<3, scalar_type> seg_SO3;
                for (std::vector<std::pair<int, int>>::iterator it = x_.SO3_state.begin(); it != x_.SO3_state.end(); it++)
                {
                    int idx = (*it).first;
                    int dim = (*it).second;
                    for (int i = 0; i < 3; i++)
                    {
                        seg_SO3(i) = dx(idx + i);
                    }

                    res_temp_SO3 = A_matrix(seg_SO3).transpose();
                    dx_new.template block<3, 1>(idx, 0) = res_temp_SO3 * dx.template block<3, 1>(idx, 0);
                    for (int i = 0; i < n; i++)
                    {
                        P_.template block<3, 1>(idx, i) = res_temp_SO3 * (P_.template block<3, 1>(idx, i));
                    }
                    for (int i = 0; i < n; i++)
                    {
                        P_.template block<1, 3>(i, idx) = (P_.template block<1, 3>(i, idx)) * res_temp_SO3.transpose();
                    }
                }

                Matrix<scalar_type, 2, 2> res_temp_S2;
                MTK::vect<2, scalar_type> seg_S2;
                for (std::vector<std::pair<int, int>>::iterator it = x_.S2_state.begin(); it != x_.S2_state.end(); it++)
                {
                    int idx = (*it).first;
                    int dim = (*it).second;
                    for (int i = 0; i < 2; i++)
                    {
                        seg_S2(i) = dx(idx + i);
                    }

                    Eigen::Matrix<scalar_type, 2, 3> Nx;
                    Eigen::Matrix<scalar_type, 3, 2> Mx;
                    x_.S2_Nx_yy(Nx, idx);
                    x_propagated.S2_Mx(Mx, seg_S2, idx);
                    res_temp_S2 = Nx * Mx;
                    dx_new.template block<2, 1>(idx, 0) = res_temp_S2 * dx.template block<2, 1>(idx, 0);
                    for (int i = 0; i < n; i++)
                    {
                        P_.template block<2, 1>(idx, i) = res_temp_S2 * (P_.template block<2, 1>(idx, i));
                    }
                    for (int i = 0; i < n; i++)
                    {
                        P_.template block<1, 2>(i, idx) = (P_.template block<1, 2>(i, idx)) * res_temp_S2.transpose();
                    }
                }

                Matrix<scalar_type, n, l> K_;
                if (n > l)
                {
#ifdef USE_sparse
                    Matrix<scalar_type, l, l> K_temp = h_x_ * P_ * h_x_.transpose();
                    spMt R_temp = h_v_ * R_ * h_v_.transpose();
                    K_temp += R_temp;
                    K_ = P_ * h_x_.transpose() * K_temp.inverse();
#else
                    K_ = P_ * h_x_.transpose() * (h_x_ * P_ * h_x_.transpose() + h_v_ * R * h_v_.transpose()).inverse();
#endif
                }
                else
                {
#ifdef USE_sparse
                    measurementnoisecovariance b = measurementnoisecovariance::Identity();
                    Eigen::SparseQR<Eigen::SparseMatrix<scalar_type>, Eigen::COLAMDOrdering<int>> solver;
                    solver.compute(R_);
                    measurementnoisecovariance R_in_temp = solver.solve(b);
                    spMt R_in = R_in_temp.sparseView();
                    spMt K_temp = h_x_.transpose() * R_in * h_x_;
                    cov_ P_temp = P_.inverse();
                    P_temp += K_temp;
                    K_ = P_temp.inverse() * h_x_.transpose() * R_in;
#else
                    measurementnoisecovariance R_in = (h_v_ * R * h_v_.transpose()).inverse();
                    K_ = (h_x_.transpose() * R_in * h_x_ + P_.inverse()).inverse() * h_x_.transpose() * R_in;
#endif
                }
                Matrix<scalar_type, l, 1> innovation;
                z.boxminus(innovation, h);
                cov K_x = K_ * h_x_;
                Matrix<scalar_type, n, 1> dx_ = K_ * innovation + (K_x - Matrix<scalar_type, n, n>::Identity()) * dx_new;
                state x_before = x_;
                x_.boxplus(dx_);

                _share.converge = true;
                for (int i = 0; i < n; i++)
                {
                    if (std::fabs(dx_[i]) > limit[i])
                    {
                        _share.converge = false;
                        break;
                    }
                }

                if (_share.converge)
                    t++;

                if (t > 1 || i == maximum_iter - 1)
                {
                    L_ = P_;

                    Matrix<scalar_type, 3, 3> res_temp_SO3;
                    MTK::vect<3, scalar_type> seg_SO3;
                    for (typename std::vector<std::pair<int, int>>::iterator it = x_.SO3_state.begin();
                         it != x_.SO3_state.end(); it++)
                    {
                        int idx = (*it).first;
                        for (int i = 0; i < 3; i++)
                        {
                            seg_SO3(i) = dx_(i + idx);
                        }
                        res_temp_SO3 = A_matrix(seg_SO3).transpose();
                        for (int i = 0; i < n; i++)
                        {
                            L_.template block<3, 1>(idx, i) = res_temp_SO3 * (P_.template block<3, 1>(idx, i));
                        }
                        if (n > l)
                        {
                            for (int i = 0; i < l; i++)
                            {
                                K_.template block<3, 1>(idx, i) = res_temp_SO3 * (K_.template block<3, 1>(idx, i));
                            }
                        }
                        else
                        {
                            for (int i = 0; i < n; i++)
                            {
                                K_x.template block<3, 1>(idx, i) = res_temp_SO3 * (K_x.template block<3, 1>(idx, i));
                            }
                        }
                        for (int i = 0; i < n; i++)
                        {
                            L_.template block<1, 3>(i, idx) = (L_.template block<1, 3>(i, idx)) * res_temp_SO3.transpose();
                            P_.template block<1, 3>(i, idx) = (P_.template block<1, 3>(i, idx)) * res_temp_SO3.transpose();
                        }
                    }

                    Matrix<scalar_type, 2, 2> res_temp_S2;
                    MTK::vect<2, scalar_type> seg_S2;
                    for (typename std::vector<std::pair<int, int>>::iterator it = x_.S2_state.begin();
                         it != x_.S2_state.end(); it++)
                    {
                        int idx = (*it).first;

                        for (int i = 0; i < 2; i++)
                        {
                            seg_S2(i) = dx_(i + idx);
                        }

                        Eigen::Matrix<scalar_type, 2, 3> Nx;
                        Eigen::Matrix<scalar_type, 3, 2> Mx;
                        x_.S2_Nx_yy(Nx, idx);
                        x_propagated.S2_Mx(Mx, seg_S2, idx);
                        res_temp_S2 = Nx * Mx;

                        for (int i = 0; i < n; i++)
                        {
                            L_.template block<2, 1>(idx, i) = res_temp_S2 * (P_.template block<2, 1>(idx, i));
                        }
                        if (n > l)
                        {
                            for (int i = 0; i < l; i++)
                            {
                                K_.template block<2, 1>(idx, i) = res_temp_S2 * (K_.template block<2, 1>(idx, i));
                            }
                        }
                        else
                        {
                            for (int i = 0; i < n; i++)
                            {
                                K_x.template block<2, 1>(idx, i) = res_temp_S2 * (K_x.template block<2, 1>(idx, i));
                            }
                        }
                        for (int i = 0; i < n; i++)
                        {
                            L_.template block<1, 2>(i, idx) = (L_.template block<1, 2>(i, idx)) * res_temp_S2.transpose();
                            P_.template block<1, 2>(i, idx) = (P_.template block<1, 2>(i, idx)) * res_temp_S2.transpose();
                        }
                    }
                    if (n > l)
                    {
                        P_ = L_ - K_ * h_x_ * P_;
                    }
                    else
                    {
                        P_ = L_ - K_x * P_;
                    }
                    return;
                }
            }
        }

        // iterated error state EKF update for measurement as an Eigen matrix whose dimension is changing.
        void update_iterated_dyn(Eigen::Matrix<scalar_type, Eigen::Dynamic, 1> z, measurementnoisecovariance_dyn R)
        {
            int t = 0;
            bool valid = true;
            bool converg = true;
            state x_propagated = x_;
            cov P_propagated = P_;
            int dof_Measurement;
            int dof_Measurement_noise = R.rows();
            for (int i = -1; i < maximum_iter; i++)
            {
                valid = true;
#ifdef USE_sparse
                spMt h_x_ = h_x_dyn(x_, valid).sparseView();
                spMt h_v_ = h_v_dyn(x_, valid).sparseView();
                spMt R_ = R.sparseView();
#else
                Matrix<scalar_type, Eigen::Dynamic, n> h_x_ = h_x_dyn(x_, valid);
                Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic> h_v_ = h_v_dyn(x_, valid);
#endif
                Matrix<scalar_type, Eigen::Dynamic, 1> h_ = h_dyn(x_, valid);
                dof_Measurement = h_.rows();
                vectorized_state dx, dx_new;
                x_.boxminus(dx, x_propagated);
                dx_new = dx;
                if (!valid)
                {
                    continue;
                }

                P_ = P_propagated;
                Matrix<scalar_type, 3, 3> res_temp_SO3;
                MTK::vect<3, scalar_type> seg_SO3;
                for (std::vector<std::pair<int, int>>::iterator it = x_.SO3_state.begin(); it != x_.SO3_state.end(); it++)
                {
                    int idx = (*it).first;
                    int dim = (*it).second;
                    for (int i = 0; i < 3; i++)
                    {
                        seg_SO3(i) = dx(idx + i);
                    }

                    res_temp_SO3 = MTK::A_matrix(seg_SO3).transpose();
                    dx_new.template block<3, 1>(idx, 0) = res_temp_SO3 * dx_new.template block<3, 1>(idx, 0);
                    for (int i = 0; i < n; i++)
                    {
                        P_.template block<3, 1>(idx, i) = res_temp_SO3 * (P_.template block<3, 1>(idx, i));
                    }
                    for (int i = 0; i < n; i++)
                    {
                        P_.template block<1, 3>(i, idx) = (P_.template block<1, 3>(i, idx)) * res_temp_SO3.transpose();
                    }
                }

                Matrix<scalar_type, 2, 2> res_temp_S2;
                MTK::vect<2, scalar_type> seg_S2;
                for (std::vector<std::pair<int, int>>::iterator it = x_.S2_state.begin(); it != x_.S2_state.end(); it++)
                {
                    int idx = (*it).first;
                    int dim = (*it).second;
                    for (int i = 0; i < 2; i++)
                    {
                        seg_S2(i) = dx(idx + i);
                    }

                    Eigen::Matrix<scalar_type, 2, 3> Nx;
                    Eigen::Matrix<scalar_type, 3, 2> Mx;
                    x_.S2_Nx_yy(Nx, idx);
                    x_propagated.S2_Mx(Mx, seg_S2, idx);
                    res_temp_S2 = Nx * Mx;
                    dx_new.template block<2, 1>(idx, 0) = res_temp_S2 * dx_new.template block<2, 1>(idx, 0);
                    for (int i = 0; i < n; i++)
                    {
                        P_.template block<2, 1>(idx, i) = res_temp_S2 * (P_.template block<2, 1>(idx, i));
                    }
                    for (int i = 0; i < n; i++)
                    {
                        P_.template block<1, 2>(i, idx) = (P_.template block<1, 2>(i, idx)) * res_temp_S2.transpose();
                    }
                }

                Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic> K_;
                if (n > dof_Measurement)
                {
#ifdef USE_sparse
                    Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic> K_temp = h_x_ * P_ * h_x_.transpose();
                    spMt R_temp = h_v_ * R_ * h_v_.transpose();
                    K_temp += R_temp;
                    K_ = P_ * h_x_.transpose() * K_temp.inverse();
#else
                    K_ = P_ * h_x_.transpose() * (h_x_ * P_ * h_x_.transpose() + h_v_ * R * h_v_.transpose()).inverse();
#endif
                }
                else
                {
#ifdef USE_sparse
                    Eigen::Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic> b =
                        Eigen::Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic>::Identity(dof_Measurement_noise,
                                                                                             dof_Measurement_noise);
                    Eigen::SparseQR<Eigen::SparseMatrix<scalar_type>, Eigen::COLAMDOrdering<int>> solver;
                    solver.compute(R_);
                    Eigen::Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic> R_in_temp = solver.solve(b);
                    spMt R_in = R_in_temp.sparseView();
                    spMt K_temp = h_x_.transpose() * R_in * h_x_;
                    cov_ P_temp = P_.inverse();
                    P_temp += K_temp;
                    K_ = P_temp.inverse() * h_x_.transpose() * R_in;
#else
                    Eigen::Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic> R_in =
                        (h_v_ * R * h_v_.transpose()).inverse();
                    K_ = (h_x_.transpose() * R_in * h_x_ + P_.inverse()).inverse() * h_x_.transpose() * R_in;
#endif
                }
                cov K_x = K_ * h_x_;
                Matrix<scalar_type, n, 1> dx_ = K_ * (z - h_) + (K_x - Matrix<scalar_type, n, n>::Identity()) * dx_new;
                state x_before = x_;
                x_.boxplus(dx_);
                converg = true;
                for (int i = 0; i < n; i++)
                {
                    if (std::fabs(dx_[i]) > limit[i])
                    {
                        converg = false;
                        break;
                    }
                }
                if (converg)
                    t++;
                if (t > 1 || i == maximum_iter - 1)
                {
                    L_ = P_;
                    std::cout << "iteration time:" << t << "," << i << std::endl;

                    Matrix<scalar_type, 3, 3> res_temp_SO3;
                    MTK::vect<3, scalar_type> seg_SO3;
                    for (typename std::vector<std::pair<int, int>>::iterator it = x_.SO3_state.begin();
                         it != x_.SO3_state.end(); it++)
                    {
                        int idx = (*it).first;
                        for (int i = 0; i < 3; i++)
                        {
                            seg_SO3(i) = dx_(i + idx);
                        }
                        res_temp_SO3 = MTK::A_matrix(seg_SO3).transpose();
                        for (int i = 0; i < n; i++)
                        {
                            L_.template block<3, 1>(idx, i) = res_temp_SO3 * (P_.template block<3, 1>(idx, i));
                        }
                        if (n > dof_Measurement)
                        {
                            for (int i = 0; i < dof_Measurement; i++)
                            {
                                K_.template block<3, 1>(idx, i) = res_temp_SO3 * (K_.template block<3, 1>(idx, i));
                            }
                        }
                        else
                        {
                            for (int i = 0; i < n; i++)
                            {
                                K_x.template block<3, 1>(idx, i) = res_temp_SO3 * (K_x.template block<3, 1>(idx, i));
                            }
                        }
                        for (int i = 0; i < n; i++)
                        {
                            L_.template block<1, 3>(i, idx) = (L_.template block<1, 3>(i, idx)) * res_temp_SO3.transpose();
                            P_.template block<1, 3>(i, idx) = (P_.template block<1, 3>(i, idx)) * res_temp_SO3.transpose();
                        }
                    }

                    Matrix<scalar_type, 2, 2> res_temp_S2;
                    MTK::vect<2, scalar_type> seg_S2;
                    for (typename std::vector<std::pair<int, int>>::iterator it = x_.S2_state.begin();
                         it != x_.S2_state.end(); it++)
                    {
                        int idx = (*it).first;

                        for (int i = 0; i < 2; i++)
                        {
                            seg_S2(i) = dx_(i + idx);
                        }

                        Eigen::Matrix<scalar_type, 2, 3> Nx;
                        Eigen::Matrix<scalar_type, 3, 2> Mx;
                        x_.S2_Nx_yy(Nx, idx);
                        x_propagated.S2_Mx(Mx, seg_S2, idx);
                        res_temp_S2 = Nx * Mx;

                        for (int i = 0; i < n; i++)
                        {
                            L_.template block<2, 1>(idx, i) = res_temp_S2 * (P_.template block<2, 1>(idx, i));
                        }
                        if (n > dof_Measurement)
                        {
                            for (int i = 0; i < dof_Measurement; i++)
                            {
                                K_.template block<2, 1>(idx, i) = res_temp_S2 * (K_.template block<2, 1>(idx, i));
                            }
                        }
                        else
                        {
                            for (int i = 0; i < n; i++)
                            {
                                K_x.template block<2, 1>(idx, i) = res_temp_S2 * (K_x.template block<2, 1>(idx, i));
                            }
                        }
                        for (int i = 0; i < n; i++)
                        {
                            L_.template block<1, 2>(i, idx) = (L_.template block<1, 2>(i, idx)) * res_temp_S2.transpose();
                            P_.template block<1, 2>(i, idx) = (P_.template block<1, 2>(i, idx)) * res_temp_S2.transpose();
                        }
                    }
                    if (n > dof_Measurement)
                    {
                        P_ = L_ - K_ * h_x_ * P_;
                    }
                    else
                    {
                        P_ = L_ - K_x * P_;
                    }
                    return;
                }
            }
        }
        // iterated error state EKF update for measurement as an Eigen matrix whose dimension is changing.
        // calculate measurement (z), estimate measurement (h), partial differention matrices (h_x, h_v) and the noise
        // covariance (R) at the same time, by only one function.
        void update_iterated_dyn_share()
        {
            int t = 0;
            dyn_share_datastruct<scalar_type> dyn_share;
            dyn_share.valid = true;
            dyn_share.converge = true;
            state x_propagated = x_;
            cov P_propagated = P_;
            int dof_Measurement;
            int dof_Measurement_noise;
            for (int i = -1; i < maximum_iter; i++)
            {
                dyn_share.valid = true;
                h_dyn_share(x_, dyn_share);
                // Matrix<scalar_type, Eigen::Dynamic, 1> h = h_dyn_share (x_,  dyn_share);
                Matrix<scalar_type, Eigen::Dynamic, 1> z = dyn_share.z;
                Matrix<scalar_type, Eigen::Dynamic, 1> h = dyn_share.h;
#ifdef USE_sparse
                spMt h_x = dyn_share.h_x.sparseView();
                spMt h_v = dyn_share.h_v.sparseView();
                spMt R_ = dyn_share.R.sparseView();
#else
                Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic> R = dyn_share.R;
                Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic> h_x = dyn_share.h_x;
                Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic> h_v = dyn_share.h_v;
#endif
                dof_Measurement = h_x.rows();
                dof_Measurement_noise = dyn_share.R.rows();
                vectorized_state dx, dx_new;
                x_.boxminus(dx, x_propagated);
                dx_new = dx;
                if (!(dyn_share.valid))
                {
                    continue;
                }

                P_ = P_propagated;
                Matrix<scalar_type, 3, 3> res_temp_SO3;
                MTK::vect<3, scalar_type> seg_SO3;
                for (std::vector<std::pair<int, int>>::iterator it = x_.SO3_state.begin(); it != x_.SO3_state.end(); it++)
                {
                    int idx = (*it).first;
                    int dim = (*it).second;
                    for (int i = 0; i < 3; i++)
                    {
                        seg_SO3(i) = dx(idx + i);
                    }

                    res_temp_SO3 = MTK::A_matrix(seg_SO3).transpose();
                    dx_new.template block<3, 1>(idx, 0) = res_temp_SO3 * dx_new.template block<3, 1>(idx, 0);
                    for (int i = 0; i < n; i++)
                    {
                        P_.template block<3, 1>(idx, i) = res_temp_SO3 * (P_.template block<3, 1>(idx, i));
                    }
                    for (int i = 0; i < n; i++)
                    {
                        P_.template block<1, 3>(i, idx) = (P_.template block<1, 3>(i, idx)) * res_temp_SO3.transpose();
                    }
                }

                Matrix<scalar_type, 2, 2> res_temp_S2;
                MTK::vect<2, scalar_type> seg_S2;
                for (std::vector<std::pair<int, int>>::iterator it = x_.S2_state.begin(); it != x_.S2_state.end(); it++)
                {
                    int idx = (*it).first;
                    int dim = (*it).second;
                    for (int i = 0; i < 2; i++)
                    {
                        seg_S2(i) = dx(idx + i);
                    }

                    Eigen::Matrix<scalar_type, 2, 3> Nx;
                    Eigen::Matrix<scalar_type, 3, 2> Mx;
                    x_.S2_Nx_yy(Nx, idx);
                    x_propagated.S2_Mx(Mx, seg_S2, idx);
                    res_temp_S2 = Nx * Mx;
                    dx_new.template block<2, 1>(idx, 0) = res_temp_S2 * dx_new.template block<2, 1>(idx, 0);
                    for (int i = 0; i < n; i++)
                    {
                        P_.template block<2, 1>(idx, i) = res_temp_S2 * (P_.template block<2, 1>(idx, i));
                    }
                    for (int i = 0; i < n; i++)
                    {
                        P_.template block<1, 2>(i, idx) = (P_.template block<1, 2>(i, idx)) * res_temp_S2.transpose();
                    }
                }

                Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic> K_;
                if (n > dof_Measurement)
                {
#ifdef USE_sparse
                    Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic> K_temp = h_x * P_ * h_x.transpose();
                    spMt R_temp = h_v * R_ * h_v.transpose();
                    K_temp += R_temp;
                    K_ = P_ * h_x.transpose() * K_temp.inverse();
#else
                    K_ = P_ * h_x.transpose() * (h_x * P_ * h_x.transpose() + h_v * R * h_v.transpose()).inverse();
#endif
                }
                else
                {
#ifdef USE_sparse
                    Eigen::Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic> b =
                        Eigen::Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic>::Identity(dof_Measurement_noise,
                                                                                             dof_Measurement_noise);
                    Eigen::SparseQR<Eigen::SparseMatrix<scalar_type>, Eigen::COLAMDOrdering<int>> solver;
                    solver.compute(R_);
                    Eigen::Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic> R_in_temp = solver.solve(b);
                    spMt R_in = R_in_temp.sparseView();
                    spMt K_temp = h_x.transpose() * R_in * h_x;
                    cov_ P_temp = P_.inverse();
                    P_temp += K_temp;
                    K_ = P_temp.inverse() * h_x.transpose() * R_in;
#else
                    Eigen::Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic> R_in = (h_v * R * h_v.transpose()).inverse();
                    K_ = (h_x.transpose() * R_in * h_x + P_.inverse()).inverse() * h_x.transpose() * R_in;
#endif
                }

                cov K_x = K_ * h_x;
                Matrix<scalar_type, n, 1> dx_ = K_ * (z - h) + (K_x - Matrix<scalar_type, n, n>::Identity()) * dx_new;
                state x_before = x_;
                x_.boxplus(dx_);
                dyn_share.converge = true;
                for (int i = 0; i < n; i++)
                {
                    if (std::fabs(dx_[i]) > limit[i])
                    {
                        dyn_share.converge = false;
                        break;
                    }
                }
                if (dyn_share.converge)
                    t++;
                if (t > 1 || i == maximum_iter - 1)
                {
                    L_ = P_;
                    std::cout << "iteration time:" << t << "," << i << std::endl;

                    Matrix<scalar_type, 3, 3> res_temp_SO3;
                    MTK::vect<3, scalar_type> seg_SO3;
                    for (typename std::vector<std::pair<int, int>>::iterator it = x_.SO3_state.begin();
                         it != x_.SO3_state.end(); it++)
                    {
                        int idx = (*it).first;
                        for (int i = 0; i < 3; i++)
                        {
                            seg_SO3(i) = dx_(i + idx);
                        }
                        res_temp_SO3 = MTK::A_matrix(seg_SO3).transpose();
                        for (int i = 0; i < int(n); i++)
                        {
                            L_.template block<3, 1>(idx, i) = res_temp_SO3 * (P_.template block<3, 1>(idx, i));
                        }
                        if (n > dof_Measurement)
                        {
                            for (int i = 0; i < dof_Measurement; i++)
                            {
                                K_.template block<3, 1>(idx, i) = res_temp_SO3 * (K_.template block<3, 1>(idx, i));
                            }
                        }
                        else
                        {
                            for (int i = 0; i < n; i++)
                            {
                                K_x.template block<3, 1>(idx, i) = res_temp_SO3 * (K_x.template block<3, 1>(idx, i));
                            }
                        }
                        for (int i = 0; i < n; i++)
                        {
                            L_.template block<1, 3>(i, idx) = (L_.template block<1, 3>(i, idx)) * res_temp_SO3.transpose();
                            P_.template block<1, 3>(i, idx) = (P_.template block<1, 3>(i, idx)) * res_temp_SO3.transpose();
                        }
                    }

                    Matrix<scalar_type, 2, 2> res_temp_S2;
                    MTK::vect<2, scalar_type> seg_S2;
                    for (typename std::vector<std::pair<int, int>>::iterator it = x_.S2_state.begin();
                         it != x_.S2_state.end(); it++)
                    {
                        int idx = (*it).first;

                        for (int i = 0; i < 2; i++)
                        {
                            seg_S2(i) = dx_(i + idx);
                        }

                        Eigen::Matrix<scalar_type, 2, 3> Nx;
                        Eigen::Matrix<scalar_type, 3, 2> Mx;
                        x_.S2_Nx_yy(Nx, idx);
                        x_propagated.S2_Mx(Mx, seg_S2, idx);
                        res_temp_S2 = Nx * Mx;

                        for (int i = 0; i < n; i++)
                        {
                            L_.template block<2, 1>(idx, i) = res_temp_S2 * (P_.template block<2, 1>(idx, i));
                        }
                        if (n > dof_Measurement)
                        {
                            for (int i = 0; i < dof_Measurement; i++)
                            {
                                K_.template block<2, 1>(idx, i) = res_temp_S2 * (K_.template block<2, 1>(idx, i));
                            }
                        }
                        else
                        {
                            for (int i = 0; i < n; i++)
                            {
                                K_x.template block<2, 1>(idx, i) = res_temp_S2 * (K_x.template block<2, 1>(idx, i));
                            }
                        }
                        for (int i = 0; i < n; i++)
                        {
                            L_.template block<1, 2>(i, idx) = (L_.template block<1, 2>(i, idx)) * res_temp_S2.transpose();
                            P_.template block<1, 2>(i, idx) = (P_.template block<1, 2>(i, idx)) * res_temp_S2.transpose();
                        }
                    }
                    if (n > dof_Measurement)
                    {
                        P_ = L_ - K_ * h_x * P_;
                    }
                    else
                    {
                        P_ = L_ - K_x * P_;
                    }
                    return;
                }
            }
        }

        // iterated error state EKF update for measurement as a dynamic manifold, whose dimension or type is changing.
        // the measurement and the measurement model are received in a dynamic manner.
        template <typename measurement_runtime, typename measurementModel_runtime>
        void update_iterated_dyn_runtime(measurement_runtime z, measurementnoisecovariance_dyn R,
                                         measurementModel_runtime h_runtime)
        {
            int t = 0;
            bool valid = true;
            bool converg = true;
            state x_propagated = x_;
            cov P_propagated = P_;
            int dof_Measurement;
            int dof_Measurement_noise;
            for (int i = -1; i < maximum_iter; i++)
            {
                valid = true;
#ifdef USE_sparse
                spMt h_x_ = h_x_dyn(x_, valid).sparseView();
                spMt h_v_ = h_v_dyn(x_, valid).sparseView();
                spMt R_ = R.sparseView();
#else
                Matrix<scalar_type, Eigen::Dynamic, n> h_x_ = h_x_dyn(x_, valid);
                Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic> h_v_ = h_v_dyn(x_, valid);
#endif
                measurement_runtime h_ = h_runtime(x_, valid);
                dof_Measurement = measurement_runtime::DOF;
                dof_Measurement_noise = R.rows();
                vectorized_state dx, dx_new;
                x_.boxminus(dx, x_propagated);
                dx_new = dx;
                if (!valid)
                {
                    continue;
                }

                P_ = P_propagated;
                Matrix<scalar_type, 3, 3> res_temp_SO3;
                MTK::vect<3, scalar_type> seg_SO3;
                for (std::vector<std::pair<int, int>>::iterator it = x_.SO3_state.begin(); it != x_.SO3_state.end(); it++)
                {
                    int idx = (*it).first;
                    int dim = (*it).second;
                    for (int i = 0; i < 3; i++)
                    {
                        seg_SO3(i) = dx(idx + i);
                    }

                    res_temp_SO3 = MTK::A_matrix(seg_SO3).transpose();
                    dx_new.template block<3, 1>(idx, 0) = res_temp_SO3 * dx_new.template block<3, 1>(idx, 0);
                    for (int i = 0; i < n; i++)
                    {
                        P_.template block<3, 1>(idx, i) = res_temp_SO3 * (P_.template block<3, 1>(idx, i));
                    }
                    for (int i = 0; i < n; i++)
                    {
                        P_.template block<1, 3>(i, idx) = (P_.template block<1, 3>(i, idx)) * res_temp_SO3.transpose();
                    }
                }

                Matrix<scalar_type, 2, 2> res_temp_S2;
                MTK::vect<2, scalar_type> seg_S2;
                for (std::vector<std::pair<int, int>>::iterator it = x_.S2_state.begin(); it != x_.S2_state.end(); it++)
                {
                    int idx = (*it).first;
                    int dim = (*it).second;
                    for (int i = 0; i < 2; i++)
                    {
                        seg_S2(i) = dx(idx + i);
                    }

                    Eigen::Matrix<scalar_type, 2, 3> Nx;
                    Eigen::Matrix<scalar_type, 3, 2> Mx;
                    x_.S2_Nx_yy(Nx, idx);
                    x_propagated.S2_Mx(Mx, seg_S2, idx);
                    res_temp_S2 = Nx * Mx;
                    dx_new.template block<2, 1>(idx, 0) = res_temp_S2 * dx_new.template block<2, 1>(idx, 0);
                    for (int i = 0; i < n; i++)
                    {
                        P_.template block<2, 1>(idx, i) = res_temp_S2 * (P_.template block<2, 1>(idx, i));
                    }
                    for (int i = 0; i < n; i++)
                    {
                        P_.template block<1, 2>(i, idx) = (P_.template block<1, 2>(i, idx)) * res_temp_S2.transpose();
                    }
                }

                Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic> K_;
                if (n > dof_Measurement)
                {
#ifdef USE_sparse
                    Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic> K_temp = h_x_ * P_ * h_x_.transpose();
                    spMt R_temp = h_v_ * R_ * h_v_.transpose();
                    K_temp += R_temp;
                    K_ = P_ * h_x_.transpose() * K_temp.inverse();
#else
                    K_ = P_ * h_x_.transpose() * (h_x_ * P_ * h_x_.transpose() + h_v_ * R * h_v_.transpose()).inverse();
#endif
                }
                else
                {
#ifdef USE_sparse
                    Eigen::Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic> b =
                        Eigen::Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic>::Identity(dof_Measurement_noise,
                                                                                             dof_Measurement_noise);
                    Eigen::SparseQR<Eigen::SparseMatrix<scalar_type>, Eigen::COLAMDOrdering<int>> solver;
                    solver.compute(R_);
                    Eigen::Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic> R_in_temp = solver.solve(b);
                    spMt R_in = R_in_temp.sparseView();
                    spMt K_temp = h_x_.transpose() * R_in * h_x_;
                    cov_ P_temp = P_.inverse();
                    P_temp += K_temp;
                    K_ = P_temp.inverse() * h_x_.transpose() * R_in;
#else
                    Eigen::Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic> R_in =
                        (h_v_ * R * h_v_.transpose()).inverse();
                    K_ = (h_x_.transpose() * R_in * h_x_ + P_.inverse()).inverse() * h_x_.transpose() * R_in;
#endif
                }
                cov K_x = K_ * h_x_;
                Eigen::Matrix<scalar_type, measurement_runtime::DOF, 1> innovation;
                z.boxminus(innovation, h_);
                Matrix<scalar_type, n, 1> dx_ = K_ * innovation + (K_x - Matrix<scalar_type, n, n>::Identity()) * dx_new;
                state x_before = x_;
                x_.boxplus(dx_);
                converg = true;
                for (int i = 0; i < n; i++)
                {
                    if (std::fabs(dx_[i]) > limit[i])
                    {
                        converg = false;
                        break;
                    }
                }
                if (converg)
                    t++;
                if (t > 1 || i == maximum_iter - 1)
                {
                    L_ = P_;
                    std::cout << "iteration time:" << t << "," << i << std::endl;

                    Matrix<scalar_type, 3, 3> res_temp_SO3;
                    MTK::vect<3, scalar_type> seg_SO3;
                    for (typename std::vector<std::pair<int, int>>::iterator it = x_.SO3_state.begin();
                         it != x_.SO3_state.end(); it++)
                    {
                        int idx = (*it).first;
                        for (int i = 0; i < 3; i++)
                        {
                            seg_SO3(i) = dx_(i + idx);
                        }
                        res_temp_SO3 = MTK::A_matrix(seg_SO3).transpose();
                        for (int i = 0; i < n; i++)
                        {
                            L_.template block<3, 1>(idx, i) = res_temp_SO3 * (P_.template block<3, 1>(idx, i));
                        }
                        if (n > dof_Measurement)
                        {
                            for (int i = 0; i < dof_Measurement; i++)
                            {
                                K_.template block<3, 1>(idx, i) = res_temp_SO3 * (K_.template block<3, 1>(idx, i));
                            }
                        }
                        else
                        {
                            for (int i = 0; i < n; i++)
                            {
                                K_x.template block<3, 1>(idx, i) = res_temp_SO3 * (K_x.template block<3, 1>(idx, i));
                            }
                        }
                        for (int i = 0; i < n; i++)
                        {
                            L_.template block<1, 3>(i, idx) = (L_.template block<1, 3>(i, idx)) * res_temp_SO3.transpose();
                            P_.template block<1, 3>(i, idx) = (P_.template block<1, 3>(i, idx)) * res_temp_SO3.transpose();
                        }
                    }

                    Matrix<scalar_type, 2, 2> res_temp_S2;
                    MTK::vect<2, scalar_type> seg_S2;
                    for (typename std::vector<std::pair<int, int>>::iterator it = x_.S2_state.begin();
                         it != x_.S2_state.end(); it++)
                    {
                        int idx = (*it).first;

                        for (int i = 0; i < 2; i++)
                        {
                            seg_S2(i) = dx_(i + idx);
                        }

                        Eigen::Matrix<scalar_type, 2, 3> Nx;
                        Eigen::Matrix<scalar_type, 3, 2> Mx;
                        x_.S2_Nx_yy(Nx, idx);
                        x_propagated.S2_Mx(Mx, seg_S2, idx);
                        res_temp_S2 = Nx * Mx;

                        for (int i = 0; i < n; i++)
                        {
                            L_.template block<2, 1>(idx, i) = res_temp_S2 * (P_.template block<2, 1>(idx, i));
                        }
                        if (n > dof_Measurement)
                        {
                            for (int i = 0; i < dof_Measurement; i++)
                            {
                                K_.template block<2, 1>(idx, i) = res_temp_S2 * (K_.template block<2, 1>(idx, i));
                            }
                        }
                        else
                        {
                            for (int i = 0; i < n; i++)
                            {
                                K_x.template block<2, 1>(idx, i) = res_temp_S2 * (K_x.template block<2, 1>(idx, i));
                            }
                        }
                        for (int i = 0; i < n; i++)
                        {
                            L_.template block<1, 2>(i, idx) = (L_.template block<1, 2>(i, idx)) * res_temp_S2.transpose();
                            P_.template block<1, 2>(i, idx) = (P_.template block<1, 2>(i, idx)) * res_temp_S2.transpose();
                        }
                    }
                    if (n > dof_Measurement)
                    {
                        P_ = L_ - K_ * h_x_ * P_;
                    }
                    else
                    {
                        P_ = L_ - K_x * P_;
                    }
                    return;
                }
            }
        }

        // iterated error state EKF update for measurement as a dynamic manifold, whose dimension or type is changing.
        // the measurement and the measurement model are received in a dynamic manner.
        // calculate measurement (z), estimate measurement (h), partial differention matrices (h_x, h_v) and the noise
        // covariance (R) at the same time, by only one function.
        template <typename measurement_runtime, typename measurementModel_dyn_runtime_share>
        void update_iterated_dyn_runtime_share(measurement_runtime z, measurementModel_dyn_runtime_share h)
        {
            int t = 0;
            dyn_runtime_share_datastruct<scalar_type> dyn_share;
            dyn_share.valid = true;
            dyn_share.converge = true;
            state x_propagated = x_;
            cov P_propagated = P_;
            int dof_Measurement;
            int dof_Measurement_noise;
            for (int i = -1; i < maximum_iter; i++)
            {
                dyn_share.valid = true;
                measurement_runtime h_ = h(x_, dyn_share);
                // measurement_runtime z = dyn_share.z;
#ifdef USE_sparse
                spMt h_x = dyn_share.h_x.sparseView();
                spMt h_v = dyn_share.h_v.sparseView();
                spMt R_ = dyn_share.R.sparseView();
#else
                Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic> R = dyn_share.R;
                Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic> h_x = dyn_share.h_x;
                Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic> h_v = dyn_share.h_v;
#endif
                dof_Measurement = measurement_runtime::DOF;
                dof_Measurement_noise = dyn_share.R.rows();
                vectorized_state dx, dx_new;
                x_.boxminus(dx, x_propagated);
                dx_new = dx;
                if (!(dyn_share.valid))
                {
                    continue;
                }

                P_ = P_propagated;
                Matrix<scalar_type, 3, 3> res_temp_SO3;
                MTK::vect<3, scalar_type> seg_SO3;
                for (std::vector<std::pair<int, int>>::iterator it = x_.SO3_state.begin(); it != x_.SO3_state.end(); it++)
                {
                    int idx = (*it).first;
                    int dim = (*it).second;
                    for (int i = 0; i < 3; i++)
                    {
                        seg_SO3(i) = dx(idx + i);
                    }

                    res_temp_SO3 = MTK::A_matrix(seg_SO3).transpose();
                    dx_new.template block<3, 1>(idx, 0) = res_temp_SO3 * dx_new.template block<3, 1>(idx, 0);
                    for (int i = 0; i < n; i++)
                    {
                        P_.template block<3, 1>(idx, i) = res_temp_SO3 * (P_.template block<3, 1>(idx, i));
                    }
                    for (int i = 0; i < n; i++)
                    {
                        P_.template block<1, 3>(i, idx) = (P_.template block<1, 3>(i, idx)) * res_temp_SO3.transpose();
                    }
                }

                Matrix<scalar_type, 2, 2> res_temp_S2;
                MTK::vect<2, scalar_type> seg_S2;
                for (std::vector<std::pair<int, int>>::iterator it = x_.S2_state.begin(); it != x_.S2_state.end(); it++)
                {
                    int idx = (*it).first;
                    int dim = (*it).second;
                    for (int i = 0; i < 2; i++)
                    {
                        seg_S2(i) = dx(idx + i);
                    }

                    Eigen::Matrix<scalar_type, 2, 3> Nx;
                    Eigen::Matrix<scalar_type, 3, 2> Mx;
                    x_.S2_Nx_yy(Nx, idx);
                    x_propagated.S2_Mx(Mx, seg_S2, idx);
                    res_temp_S2 = Nx * Mx;
                    dx_new.template block<2, 1>(idx, 0) = res_temp_S2 * dx_new.template block<2, 1>(idx, 0);
                    for (int i = 0; i < n; i++)
                    {
                        P_.template block<2, 1>(idx, i) = res_temp_S2 * (P_.template block<2, 1>(idx, i));
                    }
                    for (int i = 0; i < n; i++)
                    {
                        P_.template block<1, 2>(i, idx) = (P_.template block<1, 2>(i, idx)) * res_temp_S2.transpose();
                    }
                }

                Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic> K_;
                if (n > dof_Measurement)
                {
#ifdef USE_sparse
                    Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic> K_temp = h_x * P_ * h_x.transpose();
                    spMt R_temp = h_v * R_ * h_v.transpose();
                    K_temp += R_temp;
                    K_ = P_ * h_x.transpose() * K_temp.inverse();
#else
                    K_ = P_ * h_x.transpose() * (h_x * P_ * h_x.transpose() + h_v * R * h_v.transpose()).inverse();
#endif
                }
                else
                {
#ifdef USE_sparse
                    Eigen::Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic> b =
                        Eigen::Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic>::Identity(dof_Measurement_noise,
                                                                                             dof_Measurement_noise);
                    Eigen::SparseQR<Eigen::SparseMatrix<scalar_type>, Eigen::COLAMDOrdering<int>> solver;
                    solver.compute(R_);
                    Eigen::Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic> R_in_temp = solver.solve(b);
                    spMt R_in = R_in_temp.sparseView();
                    spMt K_temp = h_x.transpose() * R_in * h_x;
                    cov_ P_temp = P_.inverse();
                    P_temp += K_temp;
                    K_ = P_temp.inverse() * h_x.transpose() * R_in;
#else
                    Eigen::Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic> R_in = (h_v * R * h_v.transpose()).inverse();
                    K_ = (h_x.transpose() * R_in * h_x + P_.inverse()).inverse() * h_x.transpose() * R_in;
#endif
                }
                cov K_x = K_ * h_x;
                Eigen::Matrix<scalar_type, measurement_runtime::DOF, 1> innovation;
                z.boxminus(innovation, h_);
                Matrix<scalar_type, n, 1> dx_ = K_ * innovation + (K_x - Matrix<scalar_type, n, n>::Identity()) * dx_new;
                state x_before = x_;
                x_.boxplus(dx_);
                dyn_share.converge = true;
                for (int i = 0; i < n; i++)
                {
                    if (std::fabs(dx_[i]) > limit[i])
                    {
                        dyn_share.converge = false;
                        break;
                    }
                }
                if (dyn_share.converge)
                    t++;
                if (t > 1 || i == maximum_iter - 1)
                {
                    L_ = P_;
                    std::cout << "iteration time:" << t << "," << i << std::endl;

                    Matrix<scalar_type, 3, 3> res_temp_SO3;
                    MTK::vect<3, scalar_type> seg_SO3;
                    for (typename std::vector<std::pair<int, int>>::iterator it = x_.SO3_state.begin();
                         it != x_.SO3_state.end(); it++)
                    {
                        int idx = (*it).first;
                        for (int i = 0; i < 3; i++)
                        {
                            seg_SO3(i) = dx_(i + idx);
                        }
                        res_temp_SO3 = MTK::A_matrix(seg_SO3).transpose();
                        for (int i = 0; i < int(n); i++)
                        {
                            L_.template block<3, 1>(idx, i) = res_temp_SO3 * (P_.template block<3, 1>(idx, i));
                        }
                        if (n > dof_Measurement)
                        {
                            for (int i = 0; i < dof_Measurement; i++)
                            {
                                K_.template block<3, 1>(idx, i) = res_temp_SO3 * (K_.template block<3, 1>(idx, i));
                            }
                        }
                        else
                        {
                            for (int i = 0; i < n; i++)
                            {
                                K_x.template block<3, 1>(idx, i) = res_temp_SO3 * (K_x.template block<3, 1>(idx, i));
                            }
                        }
                        for (int i = 0; i < n; i++)
                        {
                            L_.template block<1, 3>(i, idx) = (L_.template block<1, 3>(i, idx)) * res_temp_SO3.transpose();
                            P_.template block<1, 3>(i, idx) = (P_.template block<1, 3>(i, idx)) * res_temp_SO3.transpose();
                        }
                    }

                    Matrix<scalar_type, 2, 2> res_temp_S2;
                    MTK::vect<2, scalar_type> seg_S2;
                    for (typename std::vector<std::pair<int, int>>::iterator it = x_.S2_state.begin();
                         it != x_.S2_state.end(); it++)
                    {
                        int idx = (*it).first;

                        for (int i = 0; i < 2; i++)
                        {
                            seg_S2(i) = dx_(i + idx);
                        }

                        Eigen::Matrix<scalar_type, 2, 3> Nx;
                        Eigen::Matrix<scalar_type, 3, 2> Mx;
                        x_.S2_Nx_yy(Nx, idx);
                        x_propagated.S2_Mx(Mx, seg_S2, idx);
                        res_temp_S2 = Nx * Mx;

                        for (int i = 0; i < n; i++)
                        {
                            L_.template block<2, 1>(idx, i) = res_temp_S2 * (P_.template block<2, 1>(idx, i));
                        }
                        if (n > dof_Measurement)
                        {
                            for (int i = 0; i < dof_Measurement; i++)
                            {
                                K_.template block<2, 1>(idx, i) = res_temp_S2 * (K_.template block<2, 1>(idx, i));
                            }
                        }
                        else
                        {
                            for (int i = 0; i < n; i++)
                            {
                                K_x.template block<2, 1>(idx, i) = res_temp_S2 * (K_x.template block<2, 1>(idx, i));
                            }
                        }
                        for (int i = 0; i < n; i++)
                        {
                            L_.template block<1, 2>(i, idx) = (L_.template block<1, 2>(i, idx)) * res_temp_S2.transpose();
                            P_.template block<1, 2>(i, idx) = (P_.template block<1, 2>(i, idx)) * res_temp_S2.transpose();
                        }
                    }
                    if (n > dof_Measurement)
                    {
                        P_ = L_ - K_ * h_x * P_;
                    }
                    else
                    {
                        P_ = L_ - K_x * P_;
                    }
                    return;
                }
            }
        }

        /**
         * [功能描述]：使用动态共享数据结构的迭代增强卡尔曼滤波更新函数
         * @param solve_time：求解时间的引用，用于返回计算耗时
         * 该函数实现了在流形空间上的迭代卡尔曼滤波更新，支持SO(3)旋转群和S2球面等特殊流形结构
         * 通过迭代优化提高状态估计的精度和收敛性
         */
        void update_iterated_dyn_share_akf(double &solve_time)
        {
            // 初始化动态共享数据结构
            dyn_share_datastruct<scalar_type> dyn_share;
            dyn_share.valid = true;     // 标记数据有效性
            dyn_share.converge = true;  // 标记收敛状态
            int t = 0;                  // 收敛计数器
            
            // 保存传播后的状态和协方差矩阵
            state x_propagated = x_;    // 状态传播值
            cov P_propagated = P_;      // 协方差传播值
            int dof_Measurement;        // 测量自由度
            
            // 卡尔曼增益相关矩阵
            Matrix<scalar_type, n, 1> K_h;     // 观测增益
            Matrix<scalar_type, n, n> K_x;     // 状态增益
            
            // 初始化状态增量
            vectorized_state dx_new = vectorized_state::Zero();
            
            // 迭代优化循环，最多进行maximum_iter次迭代
            for (int i = -1; i < maximum_iter; i++)
            {
                // 重置数据有效性标志
                dyn_share.valid = true;
                
                // 调用动态共享观测函数，计算观测方程及其雅可比矩阵
                h_dyn_share(x_, dyn_share);
                
                // 如果观测数据无效，跳过本次迭代
                if (!dyn_share.valid)
                {
                    continue;
                }
                
                // 获取观测方程的雅可比矩阵（维度为测量数×12）
                Eigen::Matrix<scalar_type, Eigen::Dynamic, 12> h_x_ = dyn_share.h_x;
                
                // 获取测量维度
                dof_Measurement = h_x_.rows();
                
                // 计算当前状态与传播状态的差值
                vectorized_state dx;
                x_.boxminus(dx, x_propagated);  // 在流形上计算状态差
                dx_new = dx;
                
                // 恢复传播后的协方差矩阵
                P_ = P_propagated;
                
                // 处理SO(3)旋转群的流形结构
                Matrix<scalar_type, 3, 3> res_temp_SO3;  // SO(3)临时变换矩阵
                MTK::vect<3, scalar_type> seg_SO3;       // SO(3)状态分量
                
                // 遍历所有SO(3)状态分量
                for (std::vector<std::pair<int, int>>::iterator it = x_.SO3_state.begin();
                    it != x_.SO3_state.end(); it++)
                {
                    int idx = (*it).first;  // 状态索引
                    int dim = (*it).second; // 状态维度
                    
                    // 提取SO(3)状态分量
                    for (int i = 0; i < 3; i++)
                    {
                        seg_SO3(i) = dx(idx + i);
                    }
                    
                    // 计算SO(3)的左雅可比矩阵的转置
                    res_temp_SO3 = MTK::A_matrix(seg_SO3).transpose();
                    
                    // 更新状态增量（考虑SO(3)流形结构）
                    dx_new.template block<3, 1>(idx, 0) = res_temp_SO3 * dx_new.template block<3, 1>(idx, 0);
                    
                    // 更新协方差矩阵的行（左乘变换矩阵）
                    for (int i = 0; i < n; i++)
                    {
                        P_.template block<3, 1>(idx, i) = res_temp_SO3 * (P_.template block<3, 1>(idx, i));
                    }
                    
                    // 更新协方差矩阵的列（右乘变换矩阵转置）
                    for (int i = 0; i < n; i++)
                    {
                        P_.template block<1, 3>(i, idx) = (P_.template block<1, 3>(i, idx)) * res_temp_SO3.transpose();
                    }
                }
                
                // 处理S2球面的流形结构
                Matrix<scalar_type, 2, 2> res_temp_S2;  // S2临时变换矩阵
                MTK::vect<2, scalar_type> seg_S2;       // S2状态分量
                
                // 遍历所有S2状态分量
                for (std::vector<std::pair<int, int>>::iterator it = x_.S2_state.begin(); it != x_.S2_state.end(); it++)
                {
                    int idx = (*it).first;  // 状态索引
                    int dim = (*it).second; // 状态维度
                    
                    // 提取S2状态分量
                    for (int i = 0; i < 2; i++)
                    {
                        seg_S2(i) = dx(idx + i);
                    }
                    
                    // 计算S2球面的切空间变换矩阵
                    Eigen::Matrix<scalar_type, 2, 3> Nx;  // 当前状态的切空间基
                    Eigen::Matrix<scalar_type, 3, 2> Mx;  // 传播状态的切空间基
                    x_.S2_Nx_yy(Nx, idx);                 // 计算当前状态切空间
                    x_propagated.S2_Mx(Mx, seg_S2, idx);  // 计算传播状态切空间
                    res_temp_S2 = Nx * Mx;                // 组合变换矩阵
                    
                    // 更新S2状态增量
                    dx_new.template block<2, 1>(idx, 0) = res_temp_S2 * dx_new.template block<2, 1>(idx, 0);
                    
                    // 更新协方差矩阵（S2部分）
                    for (int i = 0; i < n; i++)
                    {
                        P_.template block<2, 1>(idx, i) = res_temp_S2 * (P_.template block<2, 1>(idx, i));
                    }
                    for (int i = 0; i < n; i++)
                    {
                        P_.template block<1, 2>(i, idx) = (P_.template block<1, 2>(i, idx)) * res_temp_S2.transpose();
                    }
                }
                
                // 计算增强卡尔曼滤波的增益矩阵
                cov P_temp = P_.inverse();  // 协方差矩阵的逆
                
                // 计算加权观测雅可比矩阵 H^T * R^(-1)
                Eigen::Matrix<scalar_type, 12, Eigen::Dynamic> HTR = h_x_.transpose();
                {
                    // 对每个测量分量应用噪声协方差的逆
                    for (int j = 0; j < dof_Measurement; j++)
                    {
                        HTR.block(0, j, 12, 1) = HTR.block(0, j, 12, 1) * dyn_share.R_inv(j);
                    }
                }
                
                // 计算信息矩阵增量 H^T * R^(-1) * H
                Eigen::Matrix<scalar_type, 12, 12> HTRH = HTR * h_x_;
                
                // 更新信息矩阵（协方差逆矩阵）
                P_temp.template block<12, 12>(0, 0) += HTRH;
                cov P_inv = P_temp.inverse();  // 重新计算协方差矩阵
                
                // 计算卡尔曼增益
                K_h = P_inv.template block<n, 12>(0, 0) * HTR * dyn_share.h;  // 观测增益
                
                K_x.setZero();
                K_x.template block<n, 12>(0, 0) = P_inv.template block<n, 12>(0, 0) * HTRH;  // 状态增益
                
                // 计算状态更新量
                Matrix<scalar_type, n, 1> dx_ = K_h + (K_x - Matrix<scalar_type, n, n>::Identity()) * dx_new;
                
                // 保存更新前的状态
                state x_before = x_;
                
                // 在流形上更新状态
                x_.boxplus(dx_);
                
                // 检查收敛性（前6个状态分量的范数）
                dyn_share.converge = true;
                if (dx_.block(0, 0, 6, 1).norm() > 1e-4)  // 如果更新量大于阈值则未收敛
                    dyn_share.converge = false;
                
                // 更新收敛计数器
                if (dyn_share.converge)
                {
                    t++;
                }
                
                // 如果接近最大迭代次数但仍未收敛，强制收敛
                if (!t && i == maximum_iter - 2)
                {
                    dyn_share.converge = true;
                }
                
                // 收敛条件：连续收敛2次或达到最大迭代次数
                if (t > 1 || i == maximum_iter - 1)
                {
                    // 计算最终状态增量
                    x_.boxminus(dx, x_propagated);
                    last_dx = dx;  // 保存最后的状态增量
                    
                    // 更新协方差矩阵，考虑流形结构
                    L_ = P_;  // 临时存储协方差矩阵
                    
                    // 再次处理SO(3)流形结构的协方差更新
                    Matrix<scalar_type, 3, 3> res_temp_SO3;
                    MTK::vect<3, scalar_type> seg_SO3;
                    for (typename std::vector<std::pair<int, int>>::iterator it = x_.SO3_state.begin(); it != x_.SO3_state.end(); it++)
                    {
                        int idx = (*it).first;
                        for (int i = 0; i < 3; i++)
                        {
                            seg_SO3(i) = dx_(i + idx);
                        }
                        res_temp_SO3 = MTK::A_matrix(seg_SO3).transpose();
                        
                        // 更新L矩阵的SO(3)部分
                        for (int i = 0; i < n; i++)
                        {
                            L_.template block<3, 1>(idx, i) = res_temp_SO3 * (P_.template block<3, 1>(idx, i));
                        }
                        
                        // 更新增益矩阵的SO(3)部分
                        for (int i = 0; i < 12; i++)
                        {
                            K_x.template block<3, 1>(idx, i) = res_temp_SO3 * (K_x.template block<3, 1>(idx, i));
                        }
                        
                        // 更新协方差矩阵的对称性
                        for (int i = 0; i < n; i++)
                        {
                            L_.template block<1, 3>(i, idx) = (L_.template block<1, 3>(i, idx)) * res_temp_SO3.transpose();
                            P_.template block<1, 3>(i, idx) = (P_.template block<1, 3>(i, idx)) * res_temp_SO3.transpose();
                        }
                    }
                    
                    // 处理S2流形结构的协方差更新
                    Matrix<scalar_type, 2, 2> res_temp_S2;
                    MTK::vect<2, scalar_type> seg_S2;
                    for (typename std::vector<std::pair<int, int>>::iterator it = x_.S2_state.begin(); it != x_.S2_state.end(); it++)
                    {
                        int idx = (*it).first;
                        
                        for (int i = 0; i < 2; i++)
                        {
                            seg_S2(i) = dx_(i + idx);
                        }
                        
                        // 计算S2变换矩阵
                        Eigen::Matrix<scalar_type, 2, 3> Nx;
                        Eigen::Matrix<scalar_type, 3, 2> Mx;
                        x_.S2_Nx_yy(Nx, idx);
                        x_propagated.S2_Mx(Mx, seg_S2, idx);
                        res_temp_S2 = Nx * Mx;
                        
                        // 更新L矩阵和增益矩阵的S2部分
                        for (int i = 0; i < n; i++)
                        {
                            L_.template block<2, 1>(idx, i) = res_temp_S2 * (P_.template block<2, 1>(idx, i));
                        }
                        
                        for (int i = 0; i < 12; i++)
                        {
                            K_x.template block<2, 1>(idx, i) = res_temp_S2 * (K_x.template block<2, 1>(idx, i));
                        }
                        
                        // 更新协方差矩阵的对称性
                        for (int i = 0; i < n; i++)
                        {
                            L_.template block<1, 2>(i, idx) = (L_.template block<1, 2>(i, idx)) * res_temp_S2.transpose();
                            P_.template block<1, 2>(i, idx) = (P_.template block<1, 2>(i, idx)) * res_temp_S2.transpose();
                        }
                    }
                    
                    // 最终协方差更新：P = L - K*H*P
                    P_ = L_ - K_x.template block<n, 12>(0, 0) * P_.template block<12, n>(0, 0);
                    
                    // 设置全局收敛标志和增益对角线元素
                    global_converge = dyn_share.converge;
                    G_cur = K_x.block(0, 0, 6, 6).diagonal();  // 保存前6个状态的增益对角元素
                    return;  // 函数结束
                }
            }
        }

        void change_x(state &input_state)
        {
            x_ = input_state;
            if ((!x_.vect_state.size()) && (!x_.SO3_state.size()) && (!x_.S2_state.size()))
            {
                x_.build_S2_state();
                x_.build_SO3_state();
                x_.build_vect_state();
            }
        }

        void change_P(cov &input_cov) { P_ = input_cov; }

        const state &get_x() const { return x_; }
        const cov &get_P() const { return P_; }

    private:
        state x_;
        measurement m_;
        cov P_;
        spMt l_;
        spMt f_x_1;
        spMt f_x_2;
        cov F_x1 = cov::Identity();
        cov F_x2 = cov::Identity();
        cov L_ = cov::Identity();

        processModel *f;
        processMatrix1 *f_x;
        processMatrix2 *f_w;

        measurementModel *h;
        measurementMatrix1 *h_x;
        measurementMatrix2 *h_v;

        measurementModel_dyn *h_dyn;
        measurementMatrix1_dyn *h_x_dyn;
        measurementMatrix2_dyn *h_v_dyn;

        measurementModel_share *h_share;
        measurementModel_dyn_share h_dyn_share;

        int maximum_iter = 0;
        scalar_type limit[n];

        template <typename T>
        T check_safe_update(T _temp_vec)
        {
            T temp_vec = _temp_vec;
            if (std::isnan(temp_vec(0, 0)))
            {
                temp_vec.setZero();
                return temp_vec;
            }
            double angular_dis = temp_vec.block(0, 0, 3, 1).norm() * 57.3;
            double pos_dis = temp_vec.block(3, 0, 3, 1).norm();
            if (angular_dis >= 20 || pos_dis > 1)
            {
                printf("Angular dis = %.2f, pos dis = %.2f\r\n", angular_dis, pos_dis);
                temp_vec.setZero();
            }
            return temp_vec;
        }

    public:
        vectorized_state last_dx;
        Matrix<scalar_type, n, process_noise_dof> last_fw_total;
        Eigen::Matrix<double, 6, 1> G_cur;
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };

} // namespace esekfom

#endif //  ESEKFOM_EKF_HPP
