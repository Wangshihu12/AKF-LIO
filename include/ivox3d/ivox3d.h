#ifndef AKF_LIO_IVOX3D_H
#define AKF_LIO_IVOX3D_H

#include <glog/logging.h>
#include <execution>
#include <list>
#include <thread>

#include "eigen_types.h"
#include "ivox3d_node.hpp"

namespace akf_lio
{

    enum class IVoxNodeType
    {
        DEFAULT, // linear ivox
        PHC,     // phc ivox
    };

    /// traits for NodeType
    template <IVoxNodeType node_type, typename PointT, int dim>
    struct IVoxNodeTypeTraits
    {
    };

    template <typename PointT, int dim>
    struct IVoxNodeTypeTraits<IVoxNodeType::DEFAULT, PointT, dim>
    {
        using NodeType = IVoxNode<PointT, dim>;
    };

    template <typename PointT, int dim>
    struct IVoxNodeTypeTraits<IVoxNodeType::PHC, PointT, dim>
    {
        using NodeType = IVoxNodePhc<PointT, dim>;
    };

    template <int dim = 3, IVoxNodeType node_type = IVoxNodeType::DEFAULT, typename PointType = pcl::PointXYZ>
    class IVox
    {
    public:
        using KeyType = Eigen::Matrix<int, dim, 1>;
        using PtType = Eigen::Matrix<double, dim, 1>;
        using NodeType = typename IVoxNodeTypeTraits<node_type, PointType, dim>::NodeType;
        using PointVector = std::vector<PointType, Eigen::aligned_allocator<PointType>>;
        using DistPoint = typename NodeType::DistPoint;

        enum class NearbyType
        {
            CENTER, // center only
            NEARBY6,
            NEARBY18,
            NEARBY26,
        };

        struct Options
        {
            double resolution_ = 0.5;                      // ivox resolution
            double inv_resolution_ = 10.0;                 // inverse resolution
            NearbyType nearby_type_ = NearbyType::NEARBY6; // nearby range
            std::size_t capacity_ = 10000000;              // capacity
        };

        /**
         * constructor
         * @param options  ivox options
         */
        explicit IVox(Options options) : options_(options)
        {
            options_.inv_resolution_ = 1.0 / options_.resolution_;
            GenerateNearbyGrids();
        }

        /**
         * add points
         * @param points_to_add
         */
        void AddPoints(const PointVector &points_to_add);

        void ErasePoints(Eigen::Vector3d pose, double cur_time);

        void UpdateUncertainty(const PointVector &points_to_add);

        /// get nn
        bool GetClosestPoint(const PointType &pt, PointType &closest_pt);

        /// get nn with condition
        bool GetClosestPoint(const PointType &pt, PointVector &closest_pt, int max_num = 1, double max_range = INFINITY);

        /// get nn in cloud
        bool GetClosestPoint(const PointVector &cloud, PointVector &closest_cloud);

        /// get number of points
        size_t NumPoints() const;

        /// get number of valid grids
        size_t NumValidGrids() const;

        /// get statistics of the points

        void GetMapPoints(PointVector &map_points);

    private:
        /// generate the nearby grids according to the given options
        void GenerateNearbyGrids();

        /// position to grid
        KeyType Pos2Grid(const PtType &pt) const;

        Options options_;
        std::unordered_map<KeyType, typename std::list<std::pair<KeyType, NodeType>>::iterator, hash_vec<dim>>
            grids_map_;                                       // voxel hash map
        std::list<std::pair<KeyType, NodeType>> grids_cache_; // voxel cache
        std::vector<KeyType> nearby_grids_;                   // nearbys
    };

    /**
     * [功能描述]：在iVox体素网格中查找距离指定点最近的单个邻近点
     * @param pt：查询点，用于搜索最近邻点的参考位置
     * @param closest_pt：输出参数，存储找到的最近邻点
     * @return 如果成功找到最近邻点返回true，否则返回false
     * 
     * 这是GetClosestPoint函数的重载版本，只返回一个最近的点而不是多个点
     * 模板参数：
     * - dim: 维度（通常为3D）
     * - node_type: iVox节点类型
     * - PointType: 点的数据类型
     */
    template <int dim, IVoxNodeType node_type, typename PointType>
    bool IVox<dim, node_type, PointType>::GetClosestPoint(const PointType &pt, PointType &closest_pt)
    {
        // 创建候选点容器，存储所有找到的最近邻候选点
        std::vector<DistPoint> candidates;
        
        // 将查询点的3D坐标转换为网格坐标索引
        auto key = Pos2Grid(ToEigen<double, dim>(pt));
        
        // 使用std::for_each和lambda表达式遍历所有附近的网格
        std::for_each(nearby_grids_.begin(), nearby_grids_.end(), 
                    [&key, &candidates, &pt, this](const KeyType &delta)
                    {
                        // 计算目标网格的键值（当前网格键值+偏移量）
                        auto dkey = key + delta;
                        
                        // 在网格映射表中查找对应的体素
                        auto iter = grids_map_.find(dkey);
                        
                        // 如果找到了对应的体素
                        if (iter != grids_map_.end()) 
                        {
                            // 创建距离点对象，用于存储最近邻搜索结果
                            DistPoint dist_point;
                            
                            // 调用体素节点的最近邻搜索函数，在该体素中查找最近的点
                            bool found = iter->second->second.NNPoint(pt, dist_point);
                            
                            // 如果在该体素中找到了最近邻点
                            if (found) 
                            {
                                // 将找到的点添加到候选点列表中
                                candidates.emplace_back(dist_point);
                            }
                        }
                    });

        // 如果没有找到任何候选点，返回失败
        if (candidates.empty())
        {
            return false;
        }

        // 在所有候选点中找到距离最小的点
        // std::min_element根据DistPoint的比较运算符找到最小元素
        auto iter = std::min_element(candidates.begin(), candidates.end());
        
        // 获取最近邻点的实际数据并赋值给输出参数
        closest_pt = iter->Get();
        
        // 返回成功找到最近邻点
        return true;
    }

    /**
     * [功能描述]：在iVox体素网格中查找距离指定点最近的K个邻近点
     * @param pt：查询点，用于搜索最近邻点的参考位置
     * @param closest_pt：输出参数，存储找到的最近邻点集合
     * @param max_num：最大返回点数，限制搜索结果的数量
     * @param max_range：最大搜索距离，超过此距离的点将被忽略
     * @return 如果成功找到最近邻点返回true，否则返回false
     * 
     * 模板参数：
     * - dim: 维度（通常为3D）
     * - node_type: iVox节点类型
     * - PointType: 点的数据类型
     */
    template <int dim, IVoxNodeType node_type, typename PointType>
    bool IVox<dim, node_type, PointType>::GetClosestPoint(const PointType &pt, PointVector &closest_pt, int max_num,
                                                        double max_range)
    {
        // 创建候选点容器，用于存储所有符合条件的候选点
        std::vector<DistPoint> candidates;
        // 预先分配内存空间，避免频繁内存重分配（最大数量×附近网格数）
        candidates.reserve(max_num * nearby_grids_.size());

        // 将查询点的3D坐标转换为网格坐标索引
        auto key = Pos2Grid(ToEigen<double, dim>(pt));

        // 遍历所有附近的网格偏移量（包括当前网格和相邻网格）
        for (const KeyType &delta : nearby_grids_)
        {
            // 计算目标网格的键值（当前网格键值+偏移量）
            auto dkey = key + delta;
            
            // 在网格映射表中查找对应的体素
            auto iter = grids_map_.find(dkey);
            
            // 如果找到了对应的体素
            if (iter != grids_map_.end())
            {
                // 调用体素节点的KNN搜索函数，在该体素中查找符合条件的最近邻点
                // 将结果添加到candidates容器中
                auto tmp = iter->second->second.KNNPointByCondition(candidates, pt, max_num, max_range);
            }
        }
        
        // 如果没有找到任何候选点，返回失败
        if (candidates.empty())
        {
            return false;
        }
        
        // 如果候选点数量不超过所需的最大数量，直接使用所有候选点
        if (candidates.size() <= max_num)
        {
            // 不需要额外处理，继续后续步骤
        }
        else
        {
            // 如果候选点数量超过最大数量，需要进行选择
            // 使用nth_element进行部分排序，将第max_num小的元素放在正确位置
            // 这比完全排序更高效，时间复杂度O(n)而不是O(n log n)
            std::nth_element(candidates.begin(), candidates.begin() + max_num - 1, candidates.end());
            
            // 只保留距离最近的max_num个点
            candidates.resize(max_num);
        }
        
        // 对选中的候选点按距离进行完全排序（从近到远）
        std::sort(candidates.begin(), candidates.end());
        
        // 清空输出容器，准备存储结果
        closest_pt.clear();
        
        // 将排序后的候选点转换为输出格式并存储
        for (auto &it : candidates)
        {
            // 使用Get()方法获取DistPoint中的实际点数据
            closest_pt.emplace_back(it.Get());
        }
        
        // 返回是否成功找到最近邻点（closest_pt非空表示成功）
        return closest_pt.empty() == false;
    }

    template <int dim, IVoxNodeType node_type, typename PointType>
    size_t IVox<dim, node_type, PointType>::NumValidGrids() const
    {
        return grids_map_.size();
    }

    /**
     * [功能描述]：根据配置选项生成附近网格的偏移量集合
     * 该函数根据nearby_type_参数生成不同的邻近搜索模式，用于在最近邻搜索时确定需要检查的网格范围
     * 模板参数：
     * - dim: 维度（通常为3D）
     * - node_type: iVox节点类型
     * - PointType: 点的数据类型
     */
    template <int dim, IVoxNodeType node_type, typename PointType>
    void IVox<dim, node_type, PointType>::GenerateNearbyGrids()
    {
        // 根据邻近类型选择相应的网格偏移模式
        if (options_.nearby_type_ == NearbyType::CENTER)
        {
            // CENTER模式：只搜索当前网格，不搜索邻近网格
            // 适用于对性能要求极高但可以接受精度损失的场景
            nearby_grids_.emplace_back(KeyType::Zero());
        }
        else if (options_.nearby_type_ == NearbyType::NEARBY6)
        {
            // NEARBY6模式：搜索当前网格及其6个面相邻的网格（共7个网格）
            // 6个面相邻网格：前后、左右、上下各一个
            // 这是3D空间中的6连通邻域，计算效率较高
            nearby_grids_ = {
                KeyType(0, 0, 0),   // 当前网格（中心）
                KeyType(-1, 0, 0),  // X负方向相邻网格（左）
                KeyType(1, 0, 0),   // X正方向相邻网格（右）
                KeyType(0, 1, 0),   // Y正方向相邻网格（前）
                KeyType(0, -1, 0),  // Y负方向相邻网格（后）
                KeyType(0, 0, -1),  // Z负方向相邻网格（下）
                KeyType(0, 0, 1)    // Z正方向相邻网格（上）
            };
        }
        else if (options_.nearby_type_ == NearbyType::NEARBY18)
        {
            // NEARBY18模式：搜索当前网格及其18个邻近网格（共19个网格）
            // 包括6个面相邻网格 + 12个边相邻网格
            // 18连通邻域，在精度和性能之间取得平衡
            nearby_grids_ = {
                KeyType(0, 0, 0),   // 当前网格（中心）
                
                // 6个面相邻网格（与NEARBY6相同）
                KeyType(-1, 0, 0), KeyType(1, 0, 0), KeyType(0, 1, 0),
                KeyType(0, -1, 0), KeyType(0, 0, -1), KeyType(0, 0, 1),
                
                // 12个边相邻网格（沿坐标轴平面的对角线方向）
                KeyType(1, 1, 0),   // XY平面：右前
                KeyType(-1, 1, 0),  // XY平面：左前  
                KeyType(1, -1, 0),  // XY平面：右后
                KeyType(-1, -1, 0), // XY平面：左后
                KeyType(1, 0, 1),   // XZ平面：右上
                KeyType(-1, 0, 1),  // XZ平面：左上
                KeyType(1, 0, -1),  // XZ平面：右下
                KeyType(-1, 0, -1), // XZ平面：左下
                KeyType(0, 1, 1),   // YZ平面：前上
                KeyType(0, -1, 1),  // YZ平面：后上
                KeyType(0, 1, -1),  // YZ平面：前下
                KeyType(0, -1, -1)  // YZ平面：后下
            };
        }
        else if (options_.nearby_type_ == NearbyType::NEARBY26)
        {
            // NEARBY26模式：搜索当前网格及其26个邻近网格（共27个网格）
            // 包括6个面相邻 + 12个边相邻 + 8个角相邻网格
            // 26连通邻域，提供最高的搜索精度但计算开销最大
            nearby_grids_ = {
                KeyType(0, 0, 0),   // 当前网格（中心）
                
                // 6个面相邻网格
                KeyType(-1, 0, 0), KeyType(1, 0, 0), KeyType(0, 1, 0),
                KeyType(0, -1, 0), KeyType(0, 0, -1), KeyType(0, 0, 1),
                
                // 12个边相邻网格
                KeyType(1, 1, 0), KeyType(-1, 1, 0), KeyType(1, -1, 0), KeyType(-1, -1, 0),
                KeyType(1, 0, 1), KeyType(-1, 0, 1), KeyType(1, 0, -1), KeyType(-1, 0, -1),
                KeyType(0, 1, 1), KeyType(0, -1, 1), KeyType(0, 1, -1), KeyType(0, -1, -1),
                
                // 8个角相邻网格（3D空间的8个顶点方向）
                KeyType(1, 1, 1),   // 右前上
                KeyType(-1, 1, 1),  // 左前上
                KeyType(1, -1, 1),  // 右后上
                KeyType(1, 1, -1),  // 右前下
                KeyType(-1, -1, 1), // 左后上
                KeyType(-1, 1, -1), // 左前下
                KeyType(1, -1, -1), // 右后下
                KeyType(-1, -1, -1) // 左后下
            };
        }
        else
        {
            // 未知的邻近类型，记录错误信息
            LOG(ERROR) << "Unknown nearby_type!";
        }
    }

    /**
     * [功能描述]：批量查找点云中每个点的最近邻点
     * @param cloud：输入点云，包含需要查找最近邻的所有点
     * @param closest_cloud：输出点云，存储找到的每个点对应的最近邻点
     * @return 总是返回true（表示函数执行完成，不表示所有点都找到了最近邻）
     * 
     * 这是GetClosestPoint函数的批量处理版本，使用并行计算提高处理大规模点云的效率
     * 模板参数：
     * - dim: 维度（通常为3D）
     * - node_type: iVox节点类型
     * - PointType: 点的数据类型
     */
    template <int dim, IVoxNodeType node_type, typename PointType>
    bool IVox<dim, node_type, PointType>::GetClosestPoint(const PointVector &cloud, PointVector &closest_cloud)
    {
        // 创建索引向量，用于并行处理时的索引访问
        // 存储0, 1, 2, ..., cloud.size()-1的序列
        std::vector<size_t> index(cloud.size());
        for (int i = 0; i < cloud.size(); ++i)
        {
            index[i] = i;
        }
        
        // 预先调整输出点云的大小，避免并行访问时的竞争条件
        closest_cloud.resize(cloud.size());

        // 使用并行执行策略处理所有点的最近邻搜索
        // std::execution::par_unseq: 并行且无序执行，允许向量化优化
        std::for_each(std::execution::par_unseq, index.begin(), index.end(), 
                    [&cloud, &closest_cloud, this](size_t idx)
                    {
                        // 创建临时点对象存储单个点的最近邻搜索结果
                        PointType pt;
                        
                        // 调用单点版本的GetClosestPoint函数
                        // 对输入点云中索引为idx的点进行最近邻搜索
                        if (GetClosestPoint(cloud[idx], pt)) 
                        {
                            // 如果成功找到最近邻点，将结果存储到对应位置
                            closest_cloud[idx] = pt;
                        } 
                        else 
                        {
                            // 如果未找到最近邻点，存储默认构造的空点
                            // 这种情况通常发生在该点周围没有任何地图点时
                            closest_cloud[idx] = PointType();
                        }
                    });
        
        // 返回true表示批量处理完成
        // 注意：这不意味着所有点都成功找到了最近邻
        return true;
    }

    /**
     * [功能描述]：基于LRU策略和时间约束删除过期或超容量的体素网格点
     * @param pose：当前位置（此参数在当前实现中未使用，可能为将来的空间裁剪功能预留）
     * @param cur_time：当前时间戳，用于判断体素是否过期
     * 
     * 该函数实现两种删除策略：
     * 1. 基于时间的删除：移除超过时间阈值的旧体素
     * 2. 基于容量的删除：当体素数量超过最大容量时，删除最久未使用的体素
     * 模板参数：
     * - dim: 维度（通常为3D）
     * - node_type: iVox节点类型
     * - PointType: 点的数据类型
     */
    template <int dim, IVoxNodeType node_type, typename PointType>
    void IVox<dim, node_type, PointType>::ErasePoints(Eigen::Vector3d pose, double cur_time)
    {
        // LRU策略删除旧体素
        // 根据配置参数计算最大体素容量
        // 容量计算：总容量 / (分辨率 × 分辨率)，用于限制内存使用
        size_t max_capacity_ = double(options_.capacity_) / options_.resolution_ / options_.resolution_;
        
        // 第一阶段：基于时间删除过期体素
        // 从LRU缓存的尾部开始检查（最久未使用的体素）
        while (!grids_map_.empty())
        {
            // 检查缓存尾部体素是否包含点
            if (!grids_cache_.back().second.points_.empty())
            {
                // 获取该体素中最新点的时间戳（points_容器的最后一个点）
                // 如果当前时间与最新点的时间差小于删除阈值，则停止删除
                if (cur_time - grids_cache_.back().second.points_.back().time < options::TIME_TO_DELETE_LOCAL_MAP)
                    break;
            }
            
            // 从网格映射表中删除该体素
            grids_map_.erase(grids_cache_.back().first);
            // 从LRU缓存中移除该体素
            grids_cache_.pop_back();
        }

        // 第二阶段：基于容量限制删除体素
        // 如果网格数量仍然超过最大容量，继续删除最久未使用的体素
        while (grids_map_.size() >= max_capacity_)
        {
            // 从网格映射表中删除最久未使用的体素
            grids_map_.erase(grids_cache_.back().first);
            // 从LRU缓存中移除该体素
            grids_cache_.pop_back();
        }
    }

    // multiple thread fast but not accurate knn
    /**
     * [功能描述]：更新iVox体素网格中点的不确定性信息
     * @param points_to_add：需要更新不确定性的点向量
     * 该函数通过寻找体素网格中的对应点，更新其残差和不确定性信息
     * 模板参数：
     * - dim: 维度（通常为3D）
     * - node_type: iVox节点类型
     * - PointType: 点的数据类型
     */
    template <int dim, IVoxNodeType node_type, typename PointType>
    void IVox<dim, node_type, PointType>::UpdateUncertainty(const PointVector &points_to_add)
    {
        // 获取要添加的点的数量
        int add_pts_size = points_to_add.size();

        // 使用计时器评估函数执行性能
        Timer::Evaluate(
            [&, this]()
            {
                // 遍历所有需要更新不确定性的点
                for (size_t i = 0; i < add_pts_size; i++)
                {
                    // 获取当前点的引用
                    auto &pt = points_to_add.at(i);
                    
                    // 将点的位置转换为网格坐标键值
                    auto key = Pos2Grid(ToEigen<double, dim>(pt));
                    
                    // 在网格映射中查找对应的体素
                    auto iter = grids_map_.find(key);
                    bool update_flag = false; // 标记是否成功更新
                    
                    // 如果在网格映射中没有找到对应的体素
                    if (iter == grids_map_.end())
                    {
                        LOG(INFO) << "not found nn in UpdateUncertainty";
                    }
                    else
                    {
                        // 获取该体素中存储的所有点
                        auto pv = iter->second->second.GetPoints();
                        int pv_size = pv.size();
                        
                        // 遍历体素中的所有点，寻找距离最近的匹配点
                        for (int j = 0; j < pv_size; j++)
                        {
                            auto pj = pv.at(j); // 获取体素中的第j个点
                            
                            // 计算当前输入点与体素中点的欧几里得距离
                            double d2 = (Eigen::Vector3d(pj.x - pt.x, pj.y - pt.y, pj.z - pt.z)).norm();
                            
                            // 如果距离非常小（几乎重合），认为是同一个点
                            if (d2 < 1e-6)
                            {
                                // 只更新残差信息，不改变其他属性
                                auto pt3 = UpdateResidualOnly(pt, pj);
                                
                                // 用更新后的点替换体素中的原点
                                iter->second->second.ReplacePoint(j, pt3);
                                update_flag = true; // 标记更新成功
                                break; // 找到匹配点后退出循环
                            }
                        }
                        
                        // 如果在体素中没有找到匹配的点
                        if (!update_flag)
                        {
                            LOG(INFO) << "not found nn in vector";
                        }
                    }
                }
            },
            "    UpdateUncertainty"); // 计时器的标识名称
    }

    /**
     * [功能描述]：将新点云批量添加到iVox体素网格中，支持点合并和LRU缓存管理
     * @param points_to_add：需要添加到体素网格中的点向量
     * 
     * 该函数实现增量式地图更新，对每个新点：
     * 1. 搜索附近网格中的最近邻点
     * 2. 如果找到最近邻，则合并两点的信息
     * 3. 如果未找到最近邻，则直接添加新点
     * 4. 使用LRU策略维护网格缓存
     * 
     * 模板参数：
     * - dim: 维度（通常为3D）
     * - node_type: iVox节点类型
     * - PointType: 点的数据类型
     */
    template <int dim, IVoxNodeType node_type, typename PointType>
    void IVox<dim, node_type, PointType>::AddPoints(const PointVector &points_to_add)
    {
        // 获取要添加的点数量
        int add_pts_size = points_to_add.size();
        int max_num = 1; // 每个网格最多搜索1个最近邻点

        // 创建并行处理的索引向量，用于遍历附近网格
        std::vector<size_t> index(nearby_grids_.size());
        for (size_t j = 0; j < nearby_grids_.size(); j++)
        {
            index[j] = j;
        }
        
        // 逐个处理每个要添加的点
        for (int i = 0; i < add_pts_size; i++)
        {
            // 存储所有附近网格中找到的最近邻点
            std::vector<DistPoint> total_nn_vec;
            total_nn_vec.clear();
            
            // 为每个附近网格创建最近邻搜索结果容器
            std::vector<std::vector<DistPoint>> nn_vec;
            nn_vec.clear();
            nn_vec.resize(nearby_grids_.size());
            
            // 获取当前要添加的点
            auto &pt = points_to_add.at(i);
            // 计算点所在的网格坐标
            auto key = Pos2Grid(ToEigen<double, dim>(pt));
            
            // 并行搜索所有附近网格中的最近邻点
            // 使用 std::execution::par_unseq 实现无序并行执行
            std::for_each(std::execution::par_unseq, index.begin(), index.end(), 
                        [&](const size_t &j)
                        {
                            // 获取第j个附近网格的偏移量
                            auto& delta = nearby_grids_.at(j);
                            // 计算目标网格的键值
                            auto dkey = key + delta;
                            // 在网格映射中查找该网格
                            auto iter = grids_map_.find(dkey);
                            
                            // 如果找到了对应的网格
                            if (iter != grids_map_.end()) 
                            {
                                // 在该网格中搜索最近邻点（MAL = Map Aided Localization）
                                iter->second->second.KNNPointMAL(nn_vec.at(j), pt, max_num, INFINITY);
                            }
                        });
            
            // 收集所有网格中找到的最近邻点（每个网格最多1个）
            for (size_t j = 0; j < nearby_grids_.size(); j++)
            {
                if (!nn_vec.at(j).empty())
                    total_nn_vec.emplace_back(nn_vec.at(j).at(0));
            }
            
            // 如果找到了最近邻点，进行点合并操作
            if (!total_nn_vec.empty())
            {
                // 对所有候选最近邻点按距离排序
                std::sort(total_nn_vec.begin(), total_nn_vec.end());

                // 获取距离最近的点
                auto leaf = total_nn_vec.at(0).Get();
                // 计算最近邻点所在的网格坐标
                auto leaf_key = Pos2Grid(ToEigen<double, dim>(leaf));
                // 在网格映射中找到该网格
                auto map_iter = grids_map_.find(leaf_key);
                
                if (map_iter != grids_map_.end())
                {
                    // 获取最近邻点的完整信息
                    auto map_point = map_iter->second->second.GetPoint(total_nn_vec.at(0).idx);
                    // 将新点与最近邻点合并
                    auto pt_new = Merge2(pt, map_point);
                    // 从原网格中删除旧的最近邻点
                    map_iter->second->second.ErasePoint(total_nn_vec.at(0).idx);
                    // 更新LRU缓存：将该网格移到最前面（最近使用）
                    grids_cache_.splice(grids_cache_.begin(), grids_cache_, map_iter->second);
                    grids_map_[leaf_key] = grids_cache_.begin();

                    // 将合并后的新点添加到其所属的网格中
                    auto pt_key = Pos2Grid(ToEigen<double, dim>(pt_new));
                    auto iter = grids_map_.find(pt_key);
                    
                    // 如果合并后的点所属网格不存在，创建新网格
                    if (iter == grids_map_.end())
                    {
                        // 在缓存最前面创建新网格
                        grids_cache_.push_front({pt_key, NodeType(pt_new, options_.resolution_)});
                        // 在映射表中添加新网格的引用
                        grids_map_.insert({pt_key, grids_cache_.begin()});
                        // 将合并后的点插入新网格
                        grids_cache_.front().second.InsertPoint(pt_new);
                    }
                    else
                    {
                        // 如果网格已存在，直接插入点
                        iter->second->second.InsertPoint(pt_new);
                        // 更新LRU缓存：将该网格移到最前面
                        grids_cache_.splice(grids_cache_.begin(), grids_cache_, iter->second);
                        grids_map_[pt_key] = grids_cache_.begin();
                    }
                }
            }
            else
            {
                // 如果没有找到最近邻点，直接添加新点
                auto pt_key = Pos2Grid(ToEigen<double, dim>(pt));
                auto iter = grids_map_.find(pt_key);
                
                // 如果该点所属的网格不存在，创建新网格
                if (iter == grids_map_.end())
                {
                    // 在缓存最前面创建新网格
                    grids_cache_.push_front({pt_key, NodeType(pt, options_.resolution_)});
                    // 在映射表中添加新网格的引用
                    grids_map_.insert({pt_key, grids_cache_.begin()});
                    // 将点插入新网格
                    grids_cache_.front().second.InsertPoint(pt);
                }
                else
                {
                    // 如果网格已存在，直接插入点
                    iter->second->second.InsertPoint(pt);
                    // 更新LRU缓存：将该网格移到最前面
                    grids_cache_.splice(grids_cache_.begin(), grids_cache_, iter->second);
                    grids_map_[pt_key] = grids_cache_.begin();
                }
            }
        }
    }

    template <int dim, IVoxNodeType node_type, typename PointType>
    Eigen::Matrix<int, dim, 1> IVox<dim, node_type, PointType>::Pos2Grid(const IVox::PtType &pt) const
    {
        return (pt * options_.inv_resolution_).array().floor().template cast<int>();
    }

    template <int dim, IVoxNodeType node_type, typename PointType>
    void IVox<dim, node_type, PointType>::GetMapPoints(PointVector &map_points)
    {
        map_points.clear();
        for (auto &it : grids_map_)
        {
            auto pv = it.second->second.GetPoints();
            for (const auto &pt : pv)
            {
                map_points.push_back(pt);
            }
        }
    }

} // namespace akf_lio

#endif
