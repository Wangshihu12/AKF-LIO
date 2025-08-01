#include <pcl/common/centroid.h>
#include <algorithm>
#include <cmath>
#include <list>
#include <vector>

#include "hilbert.hpp"

namespace akf_lio
{

    /**
     * [功能描述]：仅更新点的残差相关信息，不改变位置、强度等几何属性
     * @param pt1：第一个输入点，提供残差更新信息
     * @param pt2：第二个输入点，作为更新的基础点
     * @return 更新后的新点，保持pt2的几何信息，但合并了残差信息
     * 
     * 该函数专门用于残差信息的增量更新，在激光雷达SLAM中用于：
     * - 更新点的不确定性估计
     * - 累积使用统计信息
     * - 保持点的位置、强度、协方差等几何属性不变
     * 
     * 模板参数：
     * - PointT: 点的数据类型，需包含use_num和uncertainty字段
     */
    template <typename PointT>
    inline PointT UpdateResidualOnly(const PointT &pt1, const PointT &pt2)
    {
        // 以pt2为基础创建结果点，保持其几何属性（位置、强度、协方差等）不变
        PointT pt3 = pt2;

        // 计算两个点的总使用次数
        int use_sum = pt1.use_num + pt2.use_num;
        
        // 仅当总使用次数不为0时才进行残差信息更新
        if (use_sum != 0)
        {
            // 基于使用次数计算加权比例
            // 使用次数越多的点在不确定性合并中权重越大
            double ratio_3 = (double)pt1.use_num / (double)(use_sum);  // pt1的权重比例
            double ratio_4 = (double)pt2.use_num / (double)(use_sum);  // pt2的权重比例
            
            // 计算加权平均不确定性
            // 这种合并方式能够综合考虑两个点的历史使用情况对不确定性的影响
            pt3.uncertainty = ratio_3 * pt1.uncertainty + ratio_4 * pt2.uncertainty;
            
            // 累加使用次数，反映该点的总体使用频率
            pt3.use_num = use_sum;
            
            // 限制使用次数的上限，防止数值溢出和计算精度问题
            if (pt3.use_num > options::MAX_FEA_NUM)
                pt3.use_num = options::MAX_FEA_NUM;
        }
        
        // 返回更新后的点
        // 注意：只有残差相关字段被更新，几何属性保持pt2的原值
        return pt3;
    }

    /**
     * [功能描述]：通用模板函数，将两个点合并为一个新点，通过加权平均融合各种属性信息
     * @param pt1：第一个要合并的点
     * @param pt2：第二个要合并的点
     * @return 合并后的新点，包含融合后的所有属性信息
     * 
     * 该函数是LaserMapping::Merge2的模板版本，可适用于不同类型的点数据结构
     * 通过考虑观测次数作为权重，实现统计意义上正确的点合并
     * 模板参数：
     * - PointT: 点的数据类型，需包含坐标、强度、协方差等字段
     */
    template <typename PointT>
    inline PointT Merge2(const PointT &pt1, const PointT &pt2)
    {
        // 以pt2为基础创建合并后的点
        PointT pt3 = pt2;
        
        // 将两个点的3D坐标转换为Eigen向量格式，便于数学运算
        common::V3D p1(pt1.x, pt1.y, pt1.z), p2(pt2.x, pt2.y, pt2.z), p3;
        
        // 根据两个点的观测次数计算加权比例
        // pt_num表示该点被观测到的次数，作为可信度权重
        double ratio_1 = (double)pt1.pt_num / (double)(pt1.pt_num + pt2.pt_num);  // pt1的权重比例
        double ratio_2 = (double)pt2.pt_num / (double)(pt1.pt_num + pt2.pt_num);  // pt2的权重比例

        // 计算加权平均位置
        p3 = p1 * ratio_1 + p2 * ratio_2;
        
        // 将融合后的位置坐标赋值给合并点
        pt3.x = p3(0);
        pt3.y = p3(1);
        pt3.z = p3(2);
        
        // 计算加权平均强度值
        pt3.intensity = (pt1.intensity * ratio_1 + pt2.intensity * ratio_2);
        
        // 使用贝叶斯统计公式更新协方差矩阵
        // 公式：Cov_new = w1*(Cov1 + μ1*μ1^T) + w2*(Cov2 + μ2*μ2^T) - μ_new*μ_new^T
        // 这确保了合并后协方差矩阵在统计学上的正确性
        pt3.cov = ratio_1 * (pt1.cov + p1 * p1.transpose()) +
                ratio_2 * (pt2.cov + p2 * p2.transpose()) - p3 * p3.transpose();
        
        // 累加观测次数，反映合并点的总观测可信度
        pt3.pt_num = pt1.pt_num + pt2.pt_num;
        
        // 保留较新的时间戳，确保时间信息的时效性
        pt3.time = pt1.time > pt2.time ? pt1.time : pt2.time;
        
        // 限制观测次数的上限，防止数值溢出和计算误差累积
        if (pt3.pt_num > options::MAX_FEA_NUM)
            pt3.pt_num = options::MAX_FEA_NUM;

        // 处理使用次数和不确定性信息的合并
        int use_sum = pt1.use_num + pt2.use_num;  // 计算总使用次数
        
        // 只有当总使用次数不为0时才进行不确定性信息的合并
        if (use_sum != 0)
        {
            // 基于使用次数重新计算权重比例
            double ratio_3 = (double)pt1.use_num / (double)(use_sum);  // pt1使用次数权重
            double ratio_4 = (double)pt2.use_num / (double)(use_sum);  // pt2使用次数权重
            
            // 基于使用次数的加权平均位置（用于不确定性相关计算）
            common::V3D p4 = ratio_3 * p1 + ratio_4 * p2;
            
            // 计算加权平均不确定性
            pt3.uncertainty = (ratio_3 * pt1.uncertainty + ratio_4 * pt2.uncertainty);
            
            // 累加使用次数
            pt3.use_num = use_sum;
            
            // 限制使用次数的上限，防止数值溢出
            if (pt3.use_num > options::MAX_FEA_NUM)
                pt3.use_num = options::MAX_FEA_NUM;
        }
        
        // 返回合并后的点
        return pt3;
    }

    template <typename PointT>
    inline double Distance2(const PointT &pt1, const PointT &pt2)
    {
        Eigen::Vector3d d = Eigen::Vector3d(pt1.x, pt1.y, pt1.z) - Eigen::Vector3d(pt2.x, pt2.y, pt2.z);
        return d.norm();
    }

    template <typename PointT>
    inline double MalDistance(const PointT &pt1, const PointT &pt2)
    {
        Eigen::Vector3d d = Eigen::Vector3d(pt1.x, pt1.y, pt1.z) - Eigen::Vector3d(pt2.x, pt2.y, pt2.z);
        double mal_dis = d.transpose() * (pt1.cov + pt2.cov).inverse() * d;
        return mal_dis;
    }

    // convert from pcl point to eigen
    template <typename T, int dim, typename PointType>
    inline Eigen::Matrix<T, dim, 1> ToEigen(const PointType &pt)
    {
        return Eigen::Matrix<T, dim, 1>(pt.x, pt.y, pt.z);
    }

    template <typename PointT, int dim = 3>
    class IVoxNode
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        struct DistPoint;

        IVoxNode() = default;
        IVoxNode(const PointT &center, const double &side_length) {} /// same with phc

        void InsertPoint(const PointT &pt);
        inline void Clear();
        inline bool Empty() const;

        inline std::size_t Size() const;

        inline PointT GetPoint(const std::size_t idx) const;

        inline std::vector<PointT> GetPoints() const;

        inline void ErasePoint(const size_t idx);

        inline void ReplacePoint(const std::size_t idx, const PointT &pt);

        int KNNPointByCondition(std::vector<DistPoint> &dis_points, const PointT &point, const int &K,
                                const double &max_range);
        int KNNPointMAL(std::vector<DistPoint> &dis_points, const PointT &point, const int &K,
                        const double &max_range);
        //   private:
        std::vector<PointT> points_;
    };

    template <typename PointT, int dim = 3>
    class IVoxNodePhc
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        struct DistPoint;
        struct PhcCube;

        IVoxNodePhc() = default;
        IVoxNodePhc(const PointT &center, const double &side_length, const int &phc_order = 6);

        void InsertPoint(const PointT &pt);

        void ErasePoint(const PointT &pt, const double erase_distance_th_);

        inline bool Empty() const;

        inline std::size_t Size() const;

        PointT GetPoint(const std::size_t idx) const;

        bool NNPoint(const PointT &cur_pt, DistPoint &dist_point) const;

        int KNNPointByCondition(std::vector<DistPoint> &dis_points, const PointT &cur_pt, const int &K = 5,
                                const double &max_range = 5.0);

    private:
        uint32_t CalculatePhcIndex(const PointT &pt) const;

    private:
        std::vector<PhcCube> phc_cubes_;

        PointT center_;
        double side_length_ = 0;
        int phc_order_ = 6;
        double phc_side_length_ = 0;
        double phc_side_length_inv_ = 0;
        Eigen::Matrix<double, dim, 1> min_cube_;
    };

    template <typename PointT, int dim>
    struct IVoxNode<PointT, dim>::DistPoint
    {
        double dist = 0;
        IVoxNode *node = nullptr;
        int idx = 0;

        DistPoint() = default;
        DistPoint(const double d, IVoxNode *n, const int i) : dist(d), node(n), idx(i) {}

        PointT Get() { return node->GetPoint(idx); }

        inline bool operator()(const DistPoint &p1, const DistPoint &p2) { return p1.dist < p2.dist; }

        inline bool operator<(const DistPoint &rhs) { return dist < rhs.dist; }
    };

    template <typename PointT, int dim>
    void IVoxNode<PointT, dim>::InsertPoint(const PointT &pt)
    {
        points_.template emplace_back(pt);
    }

    template <typename PointT, int dim>
    void IVoxNode<PointT, dim>::Clear()
    {
        points_.clear();
    }

    template <typename PointT, int dim>
    bool IVoxNode<PointT, dim>::Empty() const
    {
        return points_.empty();
    }

    template <typename PointT, int dim>
    std::size_t IVoxNode<PointT, dim>::Size() const
    {
        return points_.size();
    }

    template <typename PointT, int dim>
    PointT IVoxNode<PointT, dim>::GetPoint(const std::size_t idx) const
    {
        return points_[idx];
    }

    template <typename PointT, int dim>
    std::vector<PointT> IVoxNode<PointT, dim>::GetPoints() const
    {
        return points_;
    }

    template <typename PointT, int dim>
    void IVoxNode<PointT, dim>::ReplacePoint(const std::size_t idx, const PointT &pt)
    {
        points_.at(idx) = pt;
    }

    template <typename PointT, int dim>
    void IVoxNode<PointT, dim>::ErasePoint(const std::size_t idx)
    {
        //    LOG(INFO)<<" before erase "<<points_.size();
        //    for (auto &p:points_) {
        //        LOG(INFO)<<" before erase "<<p.x;
        //    }
        points_.erase(points_.begin() + idx);
        //    LOG(INFO)<<" after erase "<<points_.size();
        //    for (auto &p:points_) {
        //        LOG(INFO)<<" after erase "<<p.x;
        //    }
        //    points_.shrink_to_fit();
    }

    /**
     * [功能描述]：在当前体素节点中根据条件搜索K个最近邻点
     * @param dis_points：输出参数，存储找到的距离点集合（追加到现有集合中）
     * @param point：查询点，用于计算距离的参考点
     * @param K：需要返回的最近邻点数量
     * @param max_range：最大搜索范围（当前实现中未使用，可能为接口兼容性保留）
     * @return 返回dis_points的最终大小
     * 
     * 该函数在单个体素节点内进行K最近邻搜索，使用马氏距离作为度量标准
     * 模板参数：
     * - PointT: 点的数据类型
     * - dim: 维度（通常为3D）
     */
    template <typename PointT, int dim>
    int IVoxNode<PointT, dim>::KNNPointByCondition(std::vector<DistPoint> &dis_points, const PointT &point, const int &K,
                                                   const double &max_range)
    {
        // 记录dis_points的原始大小，用于后续的部分排序操作
        // 这样可以只对新添加的点进行排序，而不影响已有的点
        std::size_t old_size = dis_points.size();

        // 遍历当前体素节点中存储的所有点
        for (const auto &pt : points_)
        {
            // 计算当前点与查询点之间的马氏距离
            // 马氏距离考虑了点的协方差信息，比欧几里得距离更准确
            double d = MalDistance(pt, point);

            {
                // 将距离点信息添加到结果集合中
                // DistPoint包含：距离值、节点指针、点在节点中的索引
                // &pt - points_.data() 计算点在points_容器中的索引位置
                dis_points.template emplace_back(DistPoint(d, this, &pt - points_.data()));
            }
        }

        // 根据需要的最近邻数量K进行排序和截取
        // 如果原有点数加上K大于等于总点数，则保留所有点
        if (old_size + K >= dis_points.size())
        {
            // 不需要截取，所有点都保留
        }
        else
        {
            // 使用nth_element进行部分排序，只将第K小的元素放在正确位置
            // 这比完全排序更高效，时间复杂度O(n)而不是O(n log n)
            // 排序范围：从old_size开始的新添加的点
            std::nth_element(dis_points.begin() + old_size, 
                            dis_points.begin() + old_size + K - 1, 
                            dis_points.end());
            
            // 只保留距离最近的K个点（加上原有的old_size个点）
            dis_points.resize(old_size + K);
        }

        // 返回最终的距离点集合大小
        return dis_points.size();
    }

    template <typename PointT, int dim>
    int IVoxNode<PointT, dim>::KNNPointMAL(std::vector<DistPoint> &dis_points, const PointT &point, const int &K,
                                           const double &max_range)
    {
        std::size_t old_size = dis_points.size();

        for (const auto &pt : points_)
        {
            double d = MalDistance(pt, point);
            if (d < options::T_MAL)
            {
                dis_points.template emplace_back(DistPoint(d, this, &pt - points_.data()));
            }
        }
        // sort by distance
        if (old_size + K >= dis_points.size())
        {
        }
        else
        {
            std::nth_element(dis_points.begin() + old_size, dis_points.begin() + old_size + K - 1, dis_points.end());
            dis_points.resize(old_size + K);
        }
        return dis_points.size();
    }

    template <typename PointT, int dim>
    struct IVoxNodePhc<PointT, dim>::DistPoint
    {
        double dist = 0;
        IVoxNodePhc *node = nullptr;
        int idx = 0;

        DistPoint() {}
        DistPoint(const double d, IVoxNodePhc *n, const int i) : dist(d), node(n), idx(i) {}

        PointT Get() { return node->GetPoint(idx); }

        inline bool operator()(const DistPoint &p1, const DistPoint &p2) { return p1.dist < p2.dist; }

        inline bool operator<(const DistPoint &rhs) { return dist < rhs.dist; }
    };

    template <typename PointT, int dim>
    struct IVoxNodePhc<PointT, dim>::PhcCube
    {
        uint32_t idx = 0;
        pcl::CentroidPoint<PointT> mean;

        PhcCube(uint32_t index, const PointT &pt) { mean.add(pt); }

        void AddPoint(const PointT &pt) { mean.add(pt); }

        PointT GetPoint() const
        {
            PointT pt;
            mean.get(pt);
            return std::move(pt);
        }
    };

    template <typename PointT, int dim>
    IVoxNodePhc<PointT, dim>::IVoxNodePhc(const PointT &center, const double &side_length, const int &phc_order)
        : center_(center), side_length_(side_length), phc_order_(phc_order)
    {
        assert(phc_order <= 8);
        phc_side_length_ = side_length_ / (std::pow(2, phc_order_));
        phc_side_length_inv_ = (std::pow(2, phc_order_)) / side_length_;
        min_cube_ = center_.getArray3fMap() - side_length / 2.0;
        phc_cubes_.reserve(64);
    }

    template <typename PointT, int dim>
    void IVoxNodePhc<PointT, dim>::InsertPoint(const PointT &pt)
    {
        uint32_t idx = CalculatePhcIndex(pt);

        PhcCube cube{idx, pt};
        auto it = std::lower_bound(phc_cubes_.begin(), phc_cubes_.end(), cube,
                                   [](const PhcCube &a, const PhcCube &b)
                                   { return a.idx < b.idx; });

        if (it == phc_cubes_.end())
        {
            phc_cubes_.emplace_back(cube);
        }
        else
        {
            if (it->idx == idx)
            {
                it->AddPoint(pt);
            }
            else
            {
                phc_cubes_.insert(it, cube);
            }
        }
    }

    template <typename PointT, int dim>
    void IVoxNodePhc<PointT, dim>::ErasePoint(const PointT &pt, const double erase_distance_th_)
    {
        uint32_t idx = CalculatePhcIndex(pt);

        PhcCube cube{idx, pt};
        auto it = std::lower_bound(phc_cubes_.begin(), phc_cubes_.end(), cube,
                                   [](const PhcCube &a, const PhcCube &b)
                                   { return a.idx < b.idx; });

        if (erase_distance_th_ > 0)
        {
        }
        if (it != phc_cubes_.end() && it->idx == idx)
        {
            phc_cubes_.erase(it);
        }
    }

    template <typename PointT, int dim>
    bool IVoxNodePhc<PointT, dim>::Empty() const
    {
        return phc_cubes_.empty();
    }

    template <typename PointT, int dim>
    std::size_t IVoxNodePhc<PointT, dim>::Size() const
    {
        return phc_cubes_.size();
    }

    template <typename PointT, int dim>
    PointT IVoxNodePhc<PointT, dim>::GetPoint(const std::size_t idx) const
    {
        return phc_cubes_[idx].GetPoint();
    }

    template <typename PointT, int dim>
    bool IVoxNodePhc<PointT, dim>::NNPoint(const PointT &cur_pt, DistPoint &dist_point) const
    {
        if (phc_cubes_.empty())
        {
            return false;
        }
        uint32_t cur_idx = CalculatePhcIndex(cur_pt);
        PhcCube cube{cur_idx, cur_pt};
        auto it = std::lower_bound(phc_cubes_.begin(), phc_cubes_.end(), cube,
                                   [](const PhcCube &a, const PhcCube &b)
                                   { return a.idx < b.idx; });

        if (it == phc_cubes_.end())
        {
            it--;
            dist_point = DistPoint(Distance2(cur_pt, it->GetPoint()), this, it - phc_cubes_.begin());
        }
        else if (it == phc_cubes_.begin())
        {
            dist_point = DistPoint(Distance2(cur_pt, it->GetPoint()), this, it - phc_cubes_.begin());
        }
        else
        {
            auto last_it = it;
            last_it--;
            double d1 = Distance2(cur_pt, it->GetPoint());
            double d2 = Distance2(cur_pt, last_it->GetPoint());
            if (d1 > d2)
            {
                dist_point = DistPoint(d2, this, it - phc_cubes_.begin());
            }
            else
            {
                dist_point = DistPoint(d1, this, it - phc_cubes_.begin());
            }
        }

        return true;
    }

    /**
     * [功能描述]：在PHC（Peano-Hilbert Curve）体素节点中根据条件搜索K个最近邻点
     * @param dis_points：输出参数，存储找到的距离点集合
     * @param cur_pt：查询点，用于搜索最近邻的参考点
     * @param K：需要返回的最近邻点数量
     * @param max_range：最大搜索范围，用于限制搜索区域
     * @return 返回找到的距离点数量
     * 
     * 该函数使用Peano-Hilbert曲线索引实现高效的空间最近邻搜索
     * PHC将3D空间映射到1D索引，保持空间局部性，提高搜索效率
     * 模板参数：
     * - PointT: 点的数据类型
     * - dim: 维度（通常为3D）
     */
    template <typename PointT, int dim>
    int IVoxNodePhc<PointT, dim>::KNNPointByCondition(std::vector<DistPoint> &dis_points, const PointT &cur_pt,
                                                      const int &K, const double &max_range)
    {
        // 计算当前查询点的PHC索引
        // PHC索引将3D坐标映射为1D值，相近的3D点通常有相近的PHC索引
        uint32_t cur_idx = CalculatePhcIndex(cur_pt);
        
        // 创建查询立方体对象，包含索引和点信息
        PhcCube cube{cur_idx, cur_pt};
        
        // 使用二分查找在已排序的PHC立方体数组中找到最接近的位置
        // phc_cubes_按PHC索引排序，可以快速定位
        auto it = std::lower_bound(phc_cubes_.begin(), phc_cubes_.end(), cube,
                                [](const PhcCube &a, const PhcCube &b)
                                { return a.idx < b.idx; });

        // 根据最大搜索范围计算搜索立方体的边长
        // 使用2的幂次方确保覆盖足够的搜索区域
        const int max_search_cube_side_length = std::pow(2, std::ceil(std::log2(max_range * phc_side_length_inv_)));
        
        // 计算PHC索引空间中的最大搜索阈值
        // 立方体体积 × 8，用于限制搜索范围
        const int max_search_idx_th =
            8 * max_search_cube_side_length * max_search_cube_side_length * max_search_cube_side_length;

        // 创建距离点的lambda函数，用于生成DistPoint对象
        auto create_dist_point = [&cur_pt, this](typename std::vector<PhcCube>::const_iterator forward_it)
        {
            // 计算欧几里得距离的平方
            double d = Distance2(forward_it->GetPoint(), cur_pt);
            // 返回距离点对象，包含距离、节点指针和索引
            return DistPoint(d, this, forward_it - phc_cubes_.begin());
        };

        // 设置前向和后向搜索迭代器
        typename std::vector<PhcCube>::const_iterator forward_it(it);           // 向前搜索
        typename std::vector<PhcCube>::const_reverse_iterator backward_it(it);  // 向后搜索
        
        // 如果找到了匹配或最接近的立方体，将其添加到结果中
        if (it != phc_cubes_.end())
        {
            dis_points.emplace_back(create_dist_point(it));
            forward_it++;  // 前向迭代器向前移动
        }
        
        // 设置后向迭代器的起始位置
        if (backward_it != phc_cubes_.rend())
        {
            backward_it++;
        }

        // 定义前向搜索边界检查函数
        auto forward_reach_boundary = [&]()
        {
            // 到达数组末尾或超出搜索阈值
            return forward_it == phc_cubes_.end() || forward_it->idx - cur_idx > max_search_idx_th;
        };
        
        // 定义后向搜索边界检查函数
        auto backward_reach_boundary = [&]()
        {
            // 到达数组开头或超出搜索阈值
            return backward_it == phc_cubes_.rend() || cur_idx - backward_it->idx > max_search_idx_th;
        };

        // 双向搜索：同时向前和向后搜索，直到达到边界
        while (!forward_reach_boundary() && !backward_reach_boundary())
        {
            // 选择PHC索引距离更近的方向进行搜索
            if (forward_it->idx - cur_idx > cur_idx - backward_it->idx)
            {
                // 后向方向更近，选择后向点
                dis_points.emplace_back(create_dist_point(backward_it.base()));
                backward_it++;
            }
            else
            {
                // 前向方向更近，选择前向点
                dis_points.emplace_back(create_dist_point(forward_it));
                forward_it++;
            }
            
            // 如果已收集足够的点，提前退出
            if (dis_points.size() > K)
            {
                break;
            }
        }

        // 如果前向搜索已达边界，继续后向搜索直到收集足够的点
        if (forward_reach_boundary())
        {
            while (!backward_reach_boundary() && dis_points.size() < K)
            {
                dis_points.emplace_back(create_dist_point(backward_it.base()));
                backward_it++;
            }
        }

        // 如果后向搜索已达边界，继续前向搜索直到收集足够的点
        if (backward_reach_boundary())
        {
            while (!forward_reach_boundary() && dis_points.size() < K)
            {
                dis_points.emplace_back(create_dist_point(forward_it));
                forward_it++;
            }
        }

        // 返回找到的距离点总数
        return dis_points.size();
    }

    template <typename PointT, int dim>
    uint32_t IVoxNodePhc<PointT, dim>::CalculatePhcIndex(const PointT &pt) const
    {
        Eigen::Matrix<double, dim, 1> eposf = (pt.getVector3fMap() - min_cube_) * phc_side_length_inv_;
        Eigen::Matrix<int, dim, 1> eposi = eposf.template cast<int>();
        for (int i = 0; i < dim; ++i)
        {
            if (eposi(i, 0) < 0)
            {
                eposi(i, 0) = 0;
            }
            if (eposi(i, 0) > std::pow(2, phc_order_))
            {
                eposi(i, 0) = std::pow(2, phc_order_) - 1;
            }
        }
        std::array<uint8_t, 3> apos{eposi(0), eposi(1), eposi(2)};
        std::array<uint8_t, 3> tmp = hilbert::v2::PositionToIndex(apos);

        uint32_t idx = (uint32_t(tmp[0]) << 16) + (uint32_t(tmp[1]) << 8) + (uint32_t(tmp[2]));
        return idx;
    }

} // namespace akf_lio
