#include <fstream>
#include <math.h>
#include <vector>
#include <mutex>
#include <queue>
#include <thread>
#include <iostream>
#include <string>
#include <optional>
#include <map>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/impl/search.hpp>
#include <pcl/range_image/range_image.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/registration/icp.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/octree/octree_pointcloud_voxelcentroid.h>
#include <pcl/filters/crop_box.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/features/normal_3d.h>

#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/NavSatFix.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>

#include <eigen3/Eigen/Dense>

#include <ceres/ceres.h>

#include "utility/common.h"
#include "utility/tic_toc.h"

#include <pcl/io/pcd_io.h>

#include "feature_manager/FeatureManager.hpp"
#include "../include/utils/Twist.hpp"
#include "../include/utils/CircularBuffer.hpp"
#include "../include/utils/geometry_utils.hpp"
#include "../include/utils/math_utils.hpp"
#include "../include/3rdparty/sophus/se3.hpp"
#include "factor/PoseLocalParameterization.hpp"
#include "factor/MarginalizationFactor.hpp"
#include "factor/PivotPointPlaneFactor.hpp"
#include "factor/PriorFactor.hpp"
#include "factor/ImuFactor.hpp"
#include "imu_processor/ImuInitializer.hpp"
#include "imu_processor/IntegrationBase.hpp"
typedef pcl::PointXYZI PointT;
typedef typename pcl::PointCloud<PointT> PointCloud;
typedef typename pcl::PointCloud<PointT>::Ptr PointCloudPtr;
typedef typename pcl::PointCloud<PointT>::ConstPtr PointCloudConstPtr;
typedef Twist<float> Transform;
typedef Sophus::SO3f SO3;
typedef std::multimap<float, std::pair<PointT, PointT>, std::greater<float> > ScorePointCoeffMap;
using std::cout;
using std::endl;
using namespace std;
using namespace geometryutils;
using Eigen::Vector3d;
using Eigen::Matrix3d;
using std::shared_ptr;
using std::unique_ptr;
//todo 全局变量、类型定义 能直接初始化在定义时初始化 不能的放在main函数中初始化
// IMU参数 + 外参
double imuAccNoise = 3.9939570888238808e-03;          // 加速度噪声标准差
double imuGyrNoise = 1.5636343949698187e-03;          // 角速度噪声标准差
double imuAccBiasN = 6.4356659353532566e-05;          //
double imuGyrBiasN = 3.5640318696367613e-05;
double imuGravity = 9.80511;           // 重力加速度
double imuRPYWeight = 0.01; //not used
//给的是imu to lidar
vector<double> extRotV = {9.999976e-01, 7.553071e-04, -2.035826e-03,
                   -7.854027e-04, 9.998898e-01, -1.482298e-02,
                   2.024406e-03, 1.482454e-02, 9.998881e-01};
vector<double> extRPYV = {9.999976e-01, 7.553071e-04, -2.035826e-03,
                   -7.854027e-04, 9.998898e-01, -1.482298e-02,
                   2.024406e-03, 1.482454e-02, 9.998881e-01};
vector<double> extTransV = {-8.086759e-01, 3.195559e-01, -7.997231e-01};
Eigen::Matrix3d extRot = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(extRotV.data(), 3, 3);     // xyz坐标系旋转
Eigen::Matrix3d extRPY = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(extRPYV.data(), 3, 3);     // RPY欧拉角的变换关系
Eigen::Vector3d extTrans = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(extTransV.data(), 3, 1);   // xyz坐标系平移
Eigen::Quaterniond extQRPY = Eigen::Quaterniond(extRPY).inverse();

//消息buffer + 消息队列锁
std::queue<sensor_msgs::ImuConstPtr> imu_buf;
double last_imu_t = 0;   //imuHandler使用
std::queue<nav_msgs::Odometry::ConstPtr> odometryBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> surfBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> edgeBuf;
std::mutex mBuf;
std::mutex m_state; //没用上
std::condition_variable con;
//measurementmanager和mapping相关放前面 estimator继承它们
//measurementmanager相关---------------------------------------------
double imu_last_time_ = -1;
double curr_time_ = -1;
//自定义的雷达数据结构 包含点云 里程计信息 用于getmeasurements()对齐
class LidarInfo
{
	public:
    	LidarInfo(){};
        sensor_msgs::PointCloud2ConstPtr surf_cloud_;
    	sensor_msgs::PointCloud2ConstPtr edge_cloud_;
		nav_msgs::Odometry::ConstPtr laser_odometry_;
};
//mapping相关--------------------------------------------------------
struct CubeCenter {
  int laser_cloud_cen_length;
  int laser_cloud_cen_width;
  int laser_cloud_cen_height;

  friend std::ostream &operator<<(std::ostream &os, const CubeCenter &cube_center) {
    os << cube_center.laser_cloud_cen_length << " " << cube_center.laser_cloud_cen_width << " "
       << cube_center.laser_cloud_cen_height;
    return os;
  }
};

bool system_inited_ = false;
size_t num_max_iterations_ = 10; //scan to map的最大次数

int laser_cloud_cen_length_ = 10; //记录当前位姿的索引 作为局部地图的中心
int laser_cloud_cen_width_ = 10;
int laser_cloud_cen_height_ = 5;
const size_t laser_cloud_length_ = 21;
const size_t laser_cloud_width_ = 21;
const size_t laser_cloud_height_ = 11;
const size_t laser_cloud_num_ = laser_cloud_length_ * laser_cloud_width_ * laser_cloud_height_;

double delta_r_abort_ = 0.05; //优化终止条件
double delta_t_abort_ = 0.05;
//存接收到的每帧点云
PointCloudPtr laser_cloud_corner_last_(new PointCloud());   ///< last corner points cloud
PointCloudPtr laser_cloud_surf_last_(new PointCloud());     ///< last surface points cloud
PointCloudPtr full_cloud_(new PointCloud());      ///< last full resolution cloud

//存当前一帧点云 当前lidar系坐标 和降采样后的点云
PointCloudPtr laser_cloud_corner_stack_(new PointCloud());
PointCloudPtr laser_cloud_surf_stack_(new PointCloud());
PointCloudPtr laser_cloud_corner_stack_downsampled_(new PointCloud());  ///< down sampled
PointCloudPtr laser_cloud_surf_stack_downsampled_(new PointCloud());    ///< down sampled
//降采样器
pcl::VoxelGrid<pcl::PointXYZI> down_size_filter_corner_;   ///< voxel filter for down sizing corner clouds
pcl::VoxelGrid<pcl::PointXYZI> down_size_filter_surf_;     ///< voxel filter for down sizing surface clouds
pcl::VoxelGrid<pcl::PointXYZI> down_size_filter_map_;      ///< voxel filter for down sizing accumulated map
//分cube存所有点云
std::vector<PointCloudPtr> laser_cloud_corner_array_;
std::vector<PointCloudPtr> laser_cloud_surf_array_;
std::vector<PointCloudPtr> laser_cloud_corner_downsampled_array_;  ///< down sampled
std::vector<PointCloudPtr> laser_cloud_surf_downsampled_array_;    ///< down sampled
//lidar视野内的点云cube索引
std::vector<size_t> laser_cloud_valid_idx_;
//局部地图内的所有点云cube索引
std::vector<size_t> laser_cloud_surround_idx_;

PointCloudPtr laser_cloud_surround_(new PointCloud());
PointCloudPtr laser_cloud_surround_downsampled_(new PointCloud());     ///< down sampled
//局部地图 世界系下的坐标
PointCloudPtr laser_cloud_corner_from_map_(new PointCloud());
PointCloudPtr laser_cloud_surf_from_map_(new PointCloud());

ros::Time time_laser_cloud_corner_last_;   ///< time of current last corner cloud
ros::Time time_laser_cloud_surf_last_;     ///< time of current last surface cloud
ros::Time time_laser_full_cloud_;      ///< time of current full resolution cloud
ros::Time time_laser_odometry_;          ///< time of current laser odometry

bool new_laser_cloud_corner_last_;  ///< flag if a new last corner cloud has been received
bool new_laser_cloud_surf_last_;    ///< flag if a new last surface cloud has been received
bool new_laser_full_cloud_;     ///< flag if a new full resolution cloud has been received
bool new_laser_odometry_;         ///< flag if a new laser odometry has been received
//存lidar帧位姿
Transform transform_sum_;
Transform transform_tobe_mapped_;
Transform transform_bef_mapped_; //scan to map前的位姿
Transform transform_aft_mapped_; //scan to map后的位姿

float scan_period_; //not used
float time_factor_; //not used
const int num_stack_frames_ = 1;
const int num_map_frames_ = 5;
long frame_count_ = num_stack_frames_ - 1;   ///< number of processed frames
long map_frame_count_ = num_map_frames_ - 1;

bool is_ros_setup_ = false; //not used
bool compact_data_ = false; //not used
bool imu_inited_ = false;

multimap<float, pair<PointT, PointT>, greater<float> > score_point_coeff_;
float min_match_sq_dis_ = 1.0; //最近邻在这个阈值内才进行特征拟合
float min_plane_dis_ = 0.2;
PointT point_on_z_axis_;
Eigen::Matrix<float, 6, 6> matP_; //scan to map 使用到了
//LIO estimator------------------------------------------------------
bool first_imu_ = false;
double initial_time_ = -1;
//imu中值积分的时候前一刻的imu信息 临时存储变量
Eigen::Vector3d acc_last_, gyr_last_;
Eigen::Vector3d g_vec_;
enum EstimatorStageFlag {
  NOT_INITED,
  INITED,
};
//需要把变量都改一下
struct EstimatorConfig {
  size_t window_size = 7;
  size_t opt_window_size = 5;
  int init_window_factor = 1;
  int estimate_extrinsic = 1; //not used

  float corner_filter_size = 0.2;
  float surf_filter_size = 0.4;
  float map_filter_size = 0.6;

  float min_match_sq_dis = 1.0; //not used
  float min_plane_dis = 0.2; //not used
  //todo 追踪
  Transform transform_lb{Eigen::Quaternionf(1, 0, 0, 0), Eigen::Vector3f(0, 0, -0.1)};

  bool opt_extrinsic = true; //not used

  bool run_optimization = true;
  bool update_laser_imu = true;
  bool gravity_fix = true; //not used
  bool plane_projection_factor = false; //not used
  bool imu_factor = true;
  bool point_distance_factor = true;
  bool prior_factor = true; //true: 外参参与优化尽量保持不变
  bool marginalization_factor = true;
  bool pcl_viewer = false; //not used

  //todo 这里可能要改 不需要去畸变
  bool enable_deskew = true; ///< if disable, deskew from PointOdometry will be used
  bool cutoff_deskew = true;
  bool keep_features = false;

  leio::IntegrationBaseConfig pim_config;
};
int extrinsic_stage_ = 1; //外参估计
EstimatorStageFlag stage_flag_ = NOT_INITED;
EstimatorConfig estimator_config_;

//所有的buffer
CircularBuffer<leio::PairTimeLaserTransform> all_laser_transforms_{estimator_config_.window_size + 1};
//下面的buffer存的是IMU系位姿
CircularBuffer<Vector3d> Ps_{estimator_config_.window_size + 1};
CircularBuffer<Matrix3d> Rs_{estimator_config_.window_size + 1};
CircularBuffer<Vector3d> Vs_{estimator_config_.window_size + 1};
CircularBuffer<Vector3d> Bas_{estimator_config_.window_size + 1};
CircularBuffer<Vector3d> Bgs_{estimator_config_.window_size + 1};
CircularBuffer<size_t> size_surf_stack_{estimator_config_.window_size + 1};
CircularBuffer<size_t> size_corner_stack_{estimator_config_.window_size + 1};
//endregion
CircularBuffer<std_msgs::Header> Headers_{estimator_config_.window_size + 1};
CircularBuffer<vector<double> > dt_buf_{estimator_config_.window_size + 1};
CircularBuffer<vector<Vector3d> > linear_acceleration_buf_{estimator_config_.window_size + 1};
CircularBuffer<vector<Vector3d> > angular_velocity_buf_{estimator_config_.window_size + 1};
CircularBuffer<shared_ptr<leio::IntegrationBase> > pre_integrations_{estimator_config_.window_size + 1};
CircularBuffer<PointCloudPtr> surf_stack_{estimator_config_.window_size + 1};
CircularBuffer<PointCloudPtr> corner_stack_{estimator_config_.window_size + 1};
CircularBuffer<PointCloudPtr> full_stack_{estimator_config_.window_size + 1};
///> optimization buffers (ProcessLaserOdom中LIO滑窗优化用到的)
CircularBuffer<bool> opt_point_coeff_mask_{estimator_config_.opt_window_size + 1};
CircularBuffer<ScorePointCoeffMap> opt_point_coeff_map_{estimator_config_.opt_window_size + 1};
CircularBuffer<CubeCenter> opt_cube_centers_{estimator_config_.opt_window_size + 1};
CircularBuffer<Transform> opt_transforms_{estimator_config_.opt_window_size + 1};
CircularBuffer<vector<size_t> > opt_valid_idx_{estimator_config_.opt_window_size + 1};
CircularBuffer<PointCloudPtr> opt_corner_stack_{estimator_config_.opt_window_size + 1};
CircularBuffer<PointCloudPtr> opt_surf_stack_{estimator_config_.opt_window_size + 1};
CircularBuffer<Eigen::Matrix<double, 6, 6>> opt_matP_{estimator_config_.opt_window_size + 1};
///< optimization buffers
//todo 需要追踪一下cir_buf_count_
size_t cir_buf_count_ = 0;
size_t laser_odom_recv_count_ = 0; //ProcessLidarOdom中管理
std::shared_ptr<leio::IntegrationBase> tmp_pre_integration_;
struct StampedTransform {
  double time;
  Transform transform;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
//用于点云去畸变
CircularBuffer<StampedTransform> imu_stampedtransforms{100};
Transform transform_tobe_mapped_bef_;
Transform transform_es_;
//double赋值给float需要cast
Eigen::Quaternionf extRPY_f = Eigen::Quaterniond(extRPY).cast<float>();
Eigen::Vector3f extTrans_f = extTrans.cast<float>();
Transform transform_lb_{extRPY_f, extTrans_f}; ///< Base to laser transform
Eigen::Matrix3d R_WI_; ///< R_WI is the rotation from the inertial frame into Lidar's world frame
Eigen::Quaterniond Q_WI_; ///< Q_WI is the rotation from the inertial frame into Lidar's world frame
//tf变换
tf::StampedTransform wi_trans_, laser_local_trans_, laser_predict_trans_;
//tf::TransformBroadcaster tf_broadcaster_est_;
//todo ros publisher还没加
//LIO滑窗优化用的局部地图点云
PointCloudPtr local_surf_points_ptr_, local_surf_points_filtered_ptr_;
PointCloudPtr local_corner_points_ptr_, local_corner_points_filtered_ptr_;
//LIO滑窗优化第一个局部地图构建标识
bool init_local_map_ = false;
bool convergence_flag_ = false;
//ceres优化用的double数组
double **para_pose_;
double **para_speed_bias_;
//todo 外参不估计 和外参相关的内容需要适配 主要是ceres求残差部分
double para_ex_pose_[leio::SIZE_POSE];
double g_norm_;
bool gravity_fixed_ = false;
//边缘化所需
leio::MarginalizationInfo *last_marginalization_info;
vector<double *> last_marginalization_parameter_blocks;
vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d> > marg_coeffi, marg_coeffj;
vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > marg_pointi, marg_pointj;
vector<double> marg_score;
Vector3d P_pivot_;
Matrix3d R_pivot_;



//激光里程计回调函数
void laserOdometryHandler(const nav_msgs::Odometry::ConstPtr &_laserOdometry)
{
//    if (!init_odom)
//    {
//      //其实可以不用跳
//        //skip the first detected feature, which doesn't contain optical flow speed
//        init_odom = 1;
//        return;
//    }
    mBuf.lock();
    odometryBuf.push(_laserOdometry);
    mBuf.unlock();
    con.notify_one();
}

//对imu做积分 把最新状态赋值给最新lidar帧
void processIMU(double dt,
                const Vector3d &linear_acceleration,
                const Vector3d &angular_velocity,
                const std_msgs::Header &header)
{
    if (!first_imu_)
    {
        first_imu_ = true;
        acc_last_ = linear_acceleration;
        gyr_last_ = angular_velocity;
        //for lio
        dt_buf_.push(vector<double>());
    	linear_acceleration_buf_.push(vector<Vector3d>());
    	angular_velocity_buf_.push(vector<Vector3d>());

        Eigen::Matrix3d I3x3;
    	I3x3.setIdentity();
    	Ps_.push(Vector3d{0, 0, 0});
    	Rs_.push(I3x3);
    	Vs_.push(Vector3d{0, 0, 0});
    	Bgs_.push(Vector3d{0, 0, 0});
    	Bas_.push(Vector3d{0, 0, 0});
    }

    // NOTE: Do not update tmp_pre_integration_ until first laser comes
  	if (cir_buf_count_ != 0) {

    tmp_pre_integration_->push_back(dt, linear_acceleration, angular_velocity);

    dt_buf_[cir_buf_count_].push_back(dt);
    linear_acceleration_buf_[cir_buf_count_].push_back(linear_acceleration);
    angular_velocity_buf_[cir_buf_count_].push_back(angular_velocity);

    size_t j = cir_buf_count_;
    Vector3d un_acc_0 = Rs_[j] * (acc_last_ - Bas_[j]) + g_vec_;
    Vector3d un_gyr = 0.5 * (gyr_last_ + angular_velocity) - Bgs_[j];
    Rs_[j] *= DeltaQ(un_gyr * dt).toRotationMatrix();
    Vector3d un_acc_1 = Rs_[j] * (linear_acceleration - Bas_[j]) + g_vec_;
    Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
    Ps_[j] += dt * Vs_[j] + 0.5 * dt * dt * un_acc;
    Vs_[j] += dt * un_acc;

    StampedTransform imu_tt;
    imu_tt.time = header.stamp.toSec();
    imu_tt.transform.pos = Ps_[j].cast<float>();
    imu_tt.transform.rot = Eigen::Quaternionf(Rs_[j].cast<float>());
    imu_stampedtransforms.push(imu_tt);
//    DLOG(INFO) << imu_tt.transform;
  }
  acc_last_ = linear_acceleration;
  gyr_last_ = angular_velocity;

//  if (stage_flag_ == INITED) {
//    predict_odom_.header.stamp = header.stamp;
//    predict_odom_.header.seq += 1;
//    Eigen::Quaterniond quat(Rs_.last());
//    predict_odom_.pose.pose.orientation.x = quat.x();
//    predict_odom_.pose.pose.orientation.y = quat.y();
//    predict_odom_.pose.pose.orientation.z = quat.z();
//    predict_odom_.pose.pose.orientation.w = quat.w();
//    predict_odom_.pose.pose.position.x = Ps_.last().x();
//    predict_odom_.pose.pose.position.y = Ps_.last().y();
//    predict_odom_.pose.pose.position.z = Ps_.last().z();
//    predict_odom_.twist.twist.linear.x = Vs_.last().x();
//    predict_odom_.twist.twist.linear.y = Vs_.last().y();
//    predict_odom_.twist.twist.linear.z = Vs_.last().z();
//    predict_odom_.twist.twist.angular.x = Bas_.last().x();
//    predict_odom_.twist.twist.angular.y = Bas_.last().y();
//    predict_odom_.twist.twist.angular.z = Bas_.last().z();
//
//    pub_predict_odom_.publish(predict_odom_);
//  }
}

//当前lidar系变换到世界系
void PointAssociateToMap(const PointT &pi, PointT &po, const Transform &transform_tobe_mapped) {
  po.x = pi.x;
  po.y = pi.y;
  po.z = pi.z;
  po.intensity = pi.intensity;

  RotatePoint(transform_tobe_mapped.rot, po);

  po.x += transform_tobe_mapped.pos.x();
  po.y += transform_tobe_mapped.pos.y();
  po.z += transform_tobe_mapped.pos.z();
}

//世界系坐标变换到当前lidar系
void PointAssociateTobeMapped(const PointT &pi, PointT &po, const Transform &transform_tobe_mapped) {
  po.x = pi.x - transform_tobe_mapped.pos.x();
  po.y = pi.y - transform_tobe_mapped.pos.y();
  po.z = pi.z - transform_tobe_mapped.pos.z();
  po.intensity = pi.intensity;

  RotatePoint(transform_tobe_mapped.rot.conjugate(), po);
}

#ifdef USE_CORNER
void CalculateFeatures(const pcl::KdTreeFLANN<PointT>::Ptr &kdtree_surf_from_map,
                                  const PointCloudPtr &local_surf_points_filtered_ptr,
                                  const PointCloudPtr &surf_stack,
                                  const pcl::KdTreeFLANN<PointT>::Ptr &kdtree_corner_from_map,
                                  const PointCloudPtr &local_corner_points_filtered_ptr,
                                  const PointCloudPtr &corner_stack,
                                  const Transform &local_transform,
                                  vector<unique_ptr<Feature>> &features) {
#else
void CalculateFeatures(const pcl::KdTreeFLANN<PointT>::Ptr &kdtree_surf_from_map,
                                  const PointCloudPtr &local_surf_points_filtered_ptr,
                                  const PointCloudPtr &surf_stack,
                                  const Transform &local_transform,
                                  vector<unique_ptr<Feature>> &features) {
#endif

  PointT point_sel, point_ori, point_proj, coeff1, coeff2;
  if (!estimator_config_.keep_features) {
    features.clear();
  }

  std::vector<int> point_search_idx(5, 0);
  std::vector<float> point_search_sq_dis(5, 0);
  Eigen::Matrix<float, 5, 3> mat_A0;
  Eigen::Matrix<float, 5, 1> mat_B0;
  Eigen::Vector3f mat_X0;
  Eigen::Matrix3f mat_A1;
  Eigen::Matrix<float, 1, 3> mat_D1;
  Eigen::Matrix3f mat_V1;

  mat_A0.setZero();
  mat_B0.setConstant(-1);
  mat_X0.setZero();

  mat_A1.setZero();
  mat_D1.setZero();
  mat_V1.setZero();

  PointCloud laser_cloud_ori;
  PointCloud coeff_sel;
  vector<float> scores;
  //lidar系下的点云
  const PointCloudPtr &origin_surf_points = surf_stack;
  //i到pivot帧的相对位姿
  const Transform &transform_to_local = local_transform;
  size_t surf_points_size = origin_surf_points->points.size();

#ifdef USE_CORNER
  const PointCloudPtr &origin_corner_points = corner_stack;
  size_t corner_points_size = origin_corner_points->points.size();
#endif

//    DLOG(INFO) << "transform_to_local: " << transform_to_local;

  for (int i = 0; i < surf_points_size; i++) {
    point_ori = origin_surf_points->points[i];
    //变到pivot系下
    PointAssociateToMap(point_ori, point_sel, transform_to_local);

    int num_neighbors = 5;
    kdtree_surf_from_map->nearestKSearch(point_sel, num_neighbors, point_search_idx, point_search_sq_dis);

    if (point_search_sq_dis[num_neighbors - 1] < min_match_sq_dis_) {
      for (int j = 0; j < num_neighbors; j++) {
        mat_A0(j, 0) = local_surf_points_filtered_ptr->points[point_search_idx[j]].x;
        mat_A0(j, 1) = local_surf_points_filtered_ptr->points[point_search_idx[j]].y;
        mat_A0(j, 2) = local_surf_points_filtered_ptr->points[point_search_idx[j]].z;
      }
  //拟合平面参数
      mat_X0 = mat_A0.colPivHouseholderQr().solve(mat_B0);

      float pa = mat_X0(0, 0);
      float pb = mat_X0(1, 0);
      float pc = mat_X0(2, 0);
      float pd = 1;

      float ps = sqrt(pa * pa + pb * pb + pc * pc);
      pa /= ps;
      pb /= ps;
      pc /= ps;
      pd /= ps;

      // NOTE: plane as (x y z)*w+1 = 0

      bool planeValid = true;
      for (int j = 0; j < num_neighbors; j++) {
        if (fabs(pa * local_surf_points_filtered_ptr->points[point_search_idx[j]].x +
            pb * local_surf_points_filtered_ptr->points[point_search_idx[j]].y +
            pc * local_surf_points_filtered_ptr->points[point_search_idx[j]].z + pd) > min_plane_dis_) {
          planeValid = false;
          break;
        }
      }

      if (planeValid) {

        float pd2 = pa * point_sel.x + pb * point_sel.y + pc * point_sel.z + pd;

        float s = 1 - 0.9f * fabs(pd2) / sqrt(CalcPointDistance(point_sel));

        coeff1.x = s * pa;
        coeff1.y = s * pb;
        coeff1.z = s * pc;
        coeff1.intensity = s * pd;

        bool is_in_laser_fov = false;
        PointT transform_pos;
        PointT point_on_z_axis;

        point_on_z_axis.x = 0.0;
        point_on_z_axis.y = 0.0;
        point_on_z_axis.z = 10.0;
        PointAssociateToMap(point_on_z_axis, point_on_z_axis, transform_to_local);

        transform_pos.x = transform_to_local.pos.x();
        transform_pos.y = transform_to_local.pos.y();
        transform_pos.z = transform_to_local.pos.z();
        float squared_side1 = CalcSquaredDiff(transform_pos, point_sel);
        float squared_side2 = CalcSquaredDiff(point_on_z_axis, point_sel);

        float check1 = 100.0f + squared_side1 - squared_side2
            - 10.0f * sqrt(3.0f) * sqrt(squared_side1);

        float check2 = 100.0f + squared_side1 - squared_side2
            + 10.0f * sqrt(3.0f) * sqrt(squared_side1);

        if (check1 < 0 && check2 > 0) { /// within +-60 degree
          is_in_laser_fov = true;
        }

        if (s > 0.1 && is_in_laser_fov) {
          unique_ptr<PointPlaneFeature> feature = std::make_unique<PointPlaneFeature>();
          feature->score = s;
          feature->point = Eigen::Vector3d{point_ori.x, point_ori.y, point_ori.z};
          feature->coeffs = Eigen::Vector4d{coeff1.x, coeff1.y, coeff1.z, coeff1.intensity};
          //feature里存的是点在lidar系下的坐标
          features.push_back(std::move(feature));
        }
      }
    }
  }

#ifdef USE_CORNER
  //region Corner points
  for (int i = 0; i < corner_points_size; i++) {
    point_ori = origin_corner_points->points[i];
    PointAssociateToMap(point_ori, point_sel, transform_to_local);
    kdtree_corner_from_map->nearestKSearch(point_sel, 5, point_search_idx, point_search_sq_dis);

    if (point_search_sq_dis[4] < min_match_sq_dis_) {
      Eigen::Vector3f vc(0, 0, 0);

      for (int j = 0; j < 5; j++) {
        const PointT &point_sel_tmp = local_corner_points_filtered_ptr->points[point_search_idx[j]];
        vc.x() += point_sel_tmp.x;
        vc.y() += point_sel_tmp.y;
        vc.z() += point_sel_tmp.z;
      }
      vc /= 5.0;

      Eigen::Matrix3f mat_a;
      mat_a.setZero();

      for (int j = 0; j < 5; j++) {
        const PointT &point_sel_tmp = local_corner_points_filtered_ptr->points[point_search_idx[j]];
        Eigen::Vector3f a;
        a.x() = point_sel_tmp.x - vc.x();
        a.y() = point_sel_tmp.y - vc.y();
        a.z() = point_sel_tmp.z - vc.z();

        mat_a(0, 0) += a.x() * a.x();
        mat_a(1, 0) += a.x() * a.y();
        mat_a(2, 0) += a.x() * a.z();
        mat_a(1, 1) += a.y() * a.y();
        mat_a(2, 1) += a.y() * a.z();
        mat_a(2, 2) += a.z() * a.z();
      }
      mat_A1 = mat_a / 5.0;

      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> esolver(mat_A1);
      mat_D1 = esolver.eigenvalues().real();
      mat_V1 = esolver.eigenvectors().real();

      if (mat_D1(0, 2) > 3 * mat_D1(0, 1)) {

        float x0 = point_sel.x;
        float y0 = point_sel.y;
        float z0 = point_sel.z;
        float x1 = vc.x() + 0.1 * mat_V1(0, 2);
        float y1 = vc.y() + 0.1 * mat_V1(1, 2);
        float z1 = vc.z() + 0.1 * mat_V1(2, 2);
        float x2 = vc.x() - 0.1 * mat_V1(0, 2);
        float y2 = vc.y() - 0.1 * mat_V1(1, 2);
        float z2 = vc.z() - 0.1 * mat_V1(2, 2);

        Eigen::Vector3f X0(x0, y0, z0);
        Eigen::Vector3f X1(x1, y1, z1);
        Eigen::Vector3f X2(x2, y2, z2);

        Eigen::Vector3f a012_vec = (X0 - X1).cross(X0 - X2);

        Eigen::Vector3f normal_to_point = ((X1 - X2).cross(a012_vec)).normalized();

        Eigen::Vector3f normal_cross_point = (X1 - X2).cross(normal_to_point);

        float a012 = a012_vec.norm();

        float l12 = (X1 - X2).norm();

        float la = normal_to_point.x();
        float lb = normal_to_point.y();
        float lc = normal_to_point.z();

        float ld2 = a012 / l12;

        point_proj = point_sel;
        point_proj.x -= la * ld2;
        point_proj.y -= lb * ld2;
        point_proj.z -= lc * ld2;

        float ld_p1 = -(normal_to_point.x() * point_proj.x + normal_to_point.y() * point_proj.y
            + normal_to_point.z() * point_proj.z);
        float ld_p2 = -(normal_cross_point.x() * point_proj.x + normal_cross_point.y() * point_proj.y
            + normal_cross_point.z() * point_proj.z);

        float s = 1 - 0.9f * fabs(ld2);

        coeff1.x = s * la;
        coeff1.y = s * lb;
        coeff1.z = s * lc;
        coeff1.intensity = s * ld_p1;

        coeff2.x = s * normal_cross_point.x();
        coeff2.y = s * normal_cross_point.y();
        coeff2.z = s * normal_cross_point.z();
        coeff2.intensity = s * ld_p2;

        bool is_in_laser_fov = false;
        PointT transform_pos;
        transform_pos.x = transform_tobe_mapped_.pos.x();
        transform_pos.y = transform_tobe_mapped_.pos.y();
        transform_pos.z = transform_tobe_mapped_.pos.z();
        float squared_side1 = CalcSquaredDiff(transform_pos, point_sel);
        float squared_side2 = CalcSquaredDiff(point_on_z_axis_, point_sel);

        float check1 = 100.0f + squared_side1 - squared_side2
            - 10.0f * sqrt(3.0f) * sqrt(squared_side1);

        float check2 = 100.0f + squared_side1 - squared_side2
            + 10.0f * sqrt(3.0f) * sqrt(squared_side1);

        if (check1 < 0 && check2 > 0) { /// within +-60 degree
          is_in_laser_fov = true;
        }

        if (s > 0.1 && is_in_laser_fov) {
          unique_ptr<PointPlaneFeature> feature1 = std::make_unique<PointPlaneFeature>();
          feature1->score = s * 0.5;
          feature1->point = Eigen::Vector3d{point_ori.x, point_ori.y, point_ori.z};
          feature1->coeffs = Eigen::Vector4d{coeff1.x, coeff1.y, coeff1.z, coeff1.intensity} * 0.5;
          features.push_back(std::move(feature1));

          unique_ptr<PointPlaneFeature> feature2 = std::make_unique<PointPlaneFeature>();
          feature2->score = s * 0.5;
          feature2->point = Eigen::Vector3d{point_ori.x, point_ori.y, point_ori.z};
          feature2->coeffs = Eigen::Vector4d{coeff2.x, coeff2.y, coeff2.z, coeff2.intensity} * 0.5;
          features.push_back(std::move(feature2));
        }
      }
    }
  }
  //endregion
#endif
}

#ifdef USE_CORNER
void CalculateLaserOdom(const pcl::KdTreeFLANN<PointT>::Ptr &kdtree_surf_from_map,
                                   const PointCloudPtr &local_surf_points_filtered_ptr,
                                   const PointCloudPtr &surf_stack,
                                   const pcl::KdTreeFLANN<PointT>::Ptr &kdtree_corner_from_map,
                                   const PointCloudPtr &local_corner_points_filtered_ptr,
                                   const PointCloudPtr &corner_stack,
                                   Transform &local_transform,
                                   vector<unique_ptr<Feature>> &features) {
#else
void CalculateLaserOdom(const pcl::KdTreeFLANN<PointT>::Ptr &kdtree_surf_from_map,
                                   const PointCloudPtr &local_surf_points_filtered_ptr,
                                   const PointCloudPtr &surf_stack,
                                   Transform &local_transform,
                                   vector<unique_ptr<Feature>> &features) {
#endif

  bool is_degenerate = false;
  for (size_t iter_count = 0; iter_count < num_max_iterations_; ++iter_count) {

#ifdef USE_CORNER
    CalculateFeatures(kdtree_surf_from_map, local_surf_points_filtered_ptr, surf_stack,
                      kdtree_corner_from_map, local_corner_points_filtered_ptr, corner_stack,
                      local_transform, features);
#else
    CalculateFeatures(kdtree_surf_from_map, local_surf_points_filtered_ptr, surf_stack,
                      local_transform, features);
#endif

    size_t laser_cloud_sel_size = features.size();
    Eigen::Matrix<float, Eigen::Dynamic, 6> mat_A(laser_cloud_sel_size, 6);
    Eigen::Matrix<float, 6, Eigen::Dynamic> mat_At(6, laser_cloud_sel_size);
    Eigen::Matrix<float, 6, 6> matAtA;
    Eigen::VectorXf mat_B(laser_cloud_sel_size);
    Eigen::VectorXf mat_AtB;
    Eigen::VectorXf mat_X;
    Eigen::Matrix<float, 6, 6> matP;

    PointT point_sel, point_ori, coeff;

    SO3 R_SO3(local_transform.rot); /// SO3

    for (int i = 0; i < laser_cloud_sel_size; i++) {
      PointPlaneFeature feature_i;
      features[i]->GetFeature(&feature_i);
      point_ori.x = feature_i.point.x();
      point_ori.y = feature_i.point.y();
      point_ori.z = feature_i.point.z();
      coeff.x = feature_i.coeffs.x();
      coeff.y = feature_i.coeffs.y();
      coeff.z = feature_i.coeffs.z();
      coeff.intensity = feature_i.coeffs.w();

      Eigen::Vector3f p(point_ori.x, point_ori.y, point_ori.z);
      Eigen::Vector3f w(coeff.x, coeff.y, coeff.z);

//      Eigen::Vector3f J_r = w.transpose() * RotationVectorJacobian(R_SO3, p);
      Eigen::Vector3f J_r = -w.transpose() * (local_transform.rot * SkewSymmetric(p));
      Eigen::Vector3f J_t = w.transpose();

      float d2 = w.transpose() * (local_transform.rot * p + local_transform.pos) + coeff.intensity;

      mat_A(i, 0) = J_r.x();
      mat_A(i, 1) = J_r.y();
      mat_A(i, 2) = J_r.z();
      mat_A(i, 3) = J_t.x();
      mat_A(i, 4) = J_t.y();
      mat_A(i, 5) = J_t.z();
      mat_B(i, 0) = -d2;
    }

    mat_At = mat_A.transpose();
    matAtA = mat_At * mat_A;
    mat_AtB = mat_At * mat_B;
    mat_X = matAtA.colPivHouseholderQr().solve(mat_AtB);

    if (iter_count == 0) {
      Eigen::Matrix<float, 1, 6> mat_E;
      Eigen::Matrix<float, 6, 6> mat_V;
      Eigen::Matrix<float, 6, 6> mat_V2;

      Eigen::SelfAdjointEigenSolver<Eigen::Matrix<float, 6, 6>> esolver(matAtA);
      mat_E = esolver.eigenvalues().real();
      mat_V = esolver.eigenvectors().real();

      mat_V2 = mat_V;

      is_degenerate = false;
      float eignThre[6] = {100, 100, 100, 100, 100, 100};
      for (int i = 0; i < 6; ++i) {
        if (mat_E(0, i) < eignThre[i]) {
          for (int j = 0; j < 6; ++j) {
            mat_V2(i, j) = 0;
          }
          is_degenerate = true;
          DLOG(WARNING) << "degenerate case";
          DLOG(INFO) << mat_E;
        } else {
          break;
        }
      }
      matP = mat_V2 * mat_V.inverse();
    }

    if (is_degenerate) {
      Eigen::Matrix<float, 6, 1> matX2(mat_X);
      mat_X = matP * matX2;
    }

    local_transform.pos.x() += mat_X(3, 0);
    local_transform.pos.y() += mat_X(4, 0);
    local_transform.pos.z() += mat_X(5, 0);

    local_transform.rot = local_transform.rot * DeltaQ(Eigen::Vector3f(mat_X(0, 0), mat_X(1, 0), mat_X(2, 0)));

    if (!isfinite(local_transform.pos.x())) local_transform.pos.x() = 0.0;
    if (!isfinite(local_transform.pos.y())) local_transform.pos.y() = 0.0;
    if (!isfinite(local_transform.pos.z())) local_transform.pos.z() = 0.0;

    float delta_r = RadToDeg(R_SO3.unit_quaternion().angularDistance(local_transform.rot));
    float delta_t = sqrt(pow(mat_X(3, 0) * 100, 2) + pow(mat_X(4, 0) * 100, 2) + pow(mat_X(5, 0) * 100, 2));

    if (delta_r < delta_r_abort_ && delta_t < delta_t_abort_) {
      DLOG(INFO) << "CalculateLaserOdom iter_count: " << iter_count;
      break;
    }
  }
}

//lio滑窗构建局部地图 填充feature_frames 里面存了特征拟合参数 同时calculateOdom更新了一次
void BuildLocalMap(vector<FeaturePerFrame> &feature_frames) {
	feature_frames.clear();

	TicToc t_build_map;
  //滑窗内点云放pivot系下作为局部地图
	local_surf_points_ptr_.reset();
	local_surf_points_ptr_ = boost::make_shared<PointCloud>(PointCloud());

	local_surf_points_filtered_ptr_.reset();
	local_surf_points_filtered_ptr_ = boost::make_shared<PointCloud>(PointCloud());

#ifdef USE_CORNER
	local_corner_points_ptr_.reset();
	local_corner_points_ptr_ = boost::make_shared<PointCloud>(PointCloud());

	local_corner_points_filtered_ptr_.reset();
	local_corner_points_filtered_ptr_ = boost::make_shared<PointCloud>(PointCloud());
#endif

	PointCloud local_normal;
    //存i到pivot帧的相对位姿
	vector<Transform> local_transforms;
	int pivot_idx = estimator_config_.window_size - estimator_config_.opt_window_size;

	Twist<double> transform_lb = transform_lb_.cast<double>();

	Eigen::Vector3d Ps_pivot = Ps_[pivot_idx];
	Eigen::Matrix3d Rs_pivot = Rs_[pivot_idx];

  //lidar位姿
	Eigen::Quaterniond rot_pivot(Rs_pivot * transform_lb.rot.inverse());
	Eigen::Vector3d pos_pivot = Ps_pivot - rot_pivot * transform_lb.pos;

	Twist<double> transform_pivot = Twist<double>(rot_pivot, pos_pivot);

	{
      //构建pivot系下的局部地图
    	if (!init_local_map_) {
    		PointCloud transformed_cloud_surf, tmp_cloud_surf;
#ifdef USE_CORNER
    		PointCloud transformed_cloud_corner, tmp_cloud_corner;
#endif
    		for (int i = 0; i <= pivot_idx; ++i) {
    			Eigen::Vector3d Ps_i = Ps_[i];
    			Eigen::Matrix3d Rs_i = Rs_[i];

    			Eigen::Quaterniond rot_li(Rs_i * transform_lb.rot.inverse());
    			Eigen::Vector3d pos_li = Ps_i - rot_li * transform_lb.pos;

    			Twist<double> transform_li = Twist<double>(rot_li, pos_li);
          //lidar系 i到pivot相对位姿
    			Eigen::Affine3f transform_pivot_i = (transform_pivot.inverse() * transform_li).cast<float>().transform();
          //i系点云变换到pivot系
                pcl::transformPointCloud(*(surf_stack_[i]), transformed_cloud_surf, transform_pivot_i);
    			tmp_cloud_surf += transformed_cloud_surf;

#ifdef USE_CORNER
    			pcl::transformPointCloud(*(corner_stack_[i]), transformed_cloud_corner, transform_pivot_i);
    			tmp_cloud_corner += transformed_cloud_corner;
#endif
			}
        //surf_stack_[pivot]存前几帧共同的点云 在滑窗时便于向后传递作为局部地图
    		*(surf_stack_[pivot_idx]) = tmp_cloud_surf;
#ifdef USE_CORNER
    		*(corner_stack_[pivot_idx]) = tmp_cloud_corner;
#endif
    		init_local_map_ = true;
    	}//第一个局部地图初始化完毕

    	for (int i = 0; i < estimator_config_.window_size + 1; ++i) {
      	  Eigen::Vector3d Ps_i = Ps_[i];
      	  Eigen::Matrix3d Rs_i = Rs_[i];

      	  Eigen::Quaterniond rot_li(Rs_i * transform_lb.rot.inverse());
      	  Eigen::Vector3d pos_li = Ps_i - rot_li * transform_lb.pos;
          //lidar位姿
    	  Twist<double> transform_li = Twist<double>(rot_li, pos_li);
      	  //i到pivot相对位姿
          Eigen::Affine3f transform_pivot_i = (transform_pivot.inverse() * transform_li).cast<float>().transform();

      	  Transform local_transform = transform_pivot_i;
      	  local_transforms.push_back(local_transform);
          //pivot前帧不处理
      	  if (i < pivot_idx) {
        	continue;
      	  }

      	PointCloud transformed_cloud_surf, transformed_cloud_corner;

      	// NOTE: exclude the latest one 最新帧不处理
      	if (i != estimator_config_.window_size) {
            //pivot帧放进来
        	if (i == pivot_idx) {
          	  *local_surf_points_ptr_ += *(surf_stack_[i]);
//	          down_size_filter_surf_.setInputCloud(local_surf_points_ptr_);
//	          down_size_filter_surf_.filter(transformed_cloud_surf);
//	          *local_surf_points_ptr_ = transformed_cloud_surf;
#ifdef USE_CORNER
          	  *local_corner_points_ptr_ += *(corner_stack_[i]);
#endif
          	  continue;
        	}

          //pivot后面的帧变换到pivot系下加入到局部地图点云中
        	pcl::transformPointCloud(*(surf_stack_[i]), transformed_cloud_surf, transform_pivot_i);
#ifdef USE_CORNER
        	pcl::transformPointCloud(*(corner_stack_[i]), transformed_cloud_corner, transform_pivot_i);
#endif
        	//endregion
        	for (int p_idx = 0; p_idx < transformed_cloud_surf.size(); ++p_idx) {
          	  transformed_cloud_surf[p_idx].intensity = i;
        	}
        	*local_surf_points_ptr_ += transformed_cloud_surf;
#ifdef USE_CORNER
        	for (int p_idx = 0; p_idx < transformed_cloud_corner.size(); ++p_idx) {
          	  transformed_cloud_corner[p_idx].intensity = i;
        	}
        	*local_corner_points_ptr_ += transformed_cloud_corner;
#endif
      	}
      }

    	DLOG(INFO) << "local_surf_points_ptr_->size() bef: " << local_surf_points_ptr_->size();
    	down_size_filter_surf_.setInputCloud(local_surf_points_ptr_);
    	down_size_filter_surf_.filter(*local_surf_points_filtered_ptr_);
    	DLOG(INFO) << "local_surf_points_ptr_->size() aft: " << local_surf_points_filtered_ptr_->size();
#ifdef USE_CORNER
    	DLOG(INFO) << "local_corner_points_ptr_->size() bef: " << local_corner_points_ptr_->size();
    	down_size_filter_corner_.setInputCloud(local_corner_points_ptr_);
    	down_size_filter_corner_.filter(*local_corner_points_filtered_ptr_);
    	DLOG(INFO) << "local_corner_points_ptr_->size() aft: " << local_corner_points_filtered_ptr_->size();
#endif
	 }

  ROS_DEBUG_STREAM("t_build_map cost: " << t_build_map.toc() << " ms");
  DLOG(INFO) << "t_build_map cost: " << t_build_map.toc() << " ms";

  pcl::KdTreeFLANN<PointT>::Ptr kdtree_surf_from_map(new pcl::KdTreeFLANN<PointT>());
  kdtree_surf_from_map->setInputCloud(local_surf_points_filtered_ptr_);

#ifdef USE_CORNER
  pcl::KdTreeFLANN<PointT>::Ptr kdtree_corner_from_map(new pcl::KdTreeFLANN<PointT>());
  kdtree_corner_from_map->setInputCloud(local_corner_points_filtered_ptr_);
#endif

	for (int idx = 0; idx < estimator_config_.window_size + 1; ++idx) {
    FeaturePerFrame feature_per_frame;
    vector<unique_ptr<Feature>> features;
//  vector<unique_ptr<Feature>> &features = feature_per_frame.features;

    TicToc t_features;
    //pivot以后的帧计算特征
    if (idx > pivot_idx) {
      if (idx != estimator_config_.window_size || !estimator_config_.imu_factor) {
#ifdef USE_CORNER
        CalculateFeatures(kdtree_surf_from_map, local_surf_points_filtered_ptr_, surf_stack_[idx],
                         kdtree_corner_from_map, local_corner_points_filtered_ptr_, corner_stack_[idx],
                          local_transforms[idx], features);
#else
//kdtree 和 局部地图点云是匹配的
        CalculateFeatures(kdtree_surf_from_map, local_surf_points_filtered_ptr_, surf_stack_[idx],
                          local_transforms[idx], features);
#endif
      } else {
        DLOG(INFO) << "local_transforms[idx] bef" << local_transforms[idx];

#ifdef USE_CORNER
        CalculateLaserOdom(kdtree_surf_from_map, local_surf_points_filtered_ptr_, surf_stack_[idx],
                           kdtree_corner_from_map, local_corner_points_filtered_ptr_, corner_stack_[idx],
                           local_transforms[idx], features);
#else
//更新了local_transforms[idx]
        CalculateLaserOdom(kdtree_surf_from_map, local_surf_points_filtered_ptr_, surf_stack_[idx],
                           local_transforms[idx], features);
#endif

        DLOG(INFO) << "local_transforms[idx] aft" << local_transforms[idx];
      }
    } else { //pivot及之前的帧不处理
      // NOTE: empty features
    }

    feature_per_frame.id = idx;
//    feature_per_frame.features = std::move(features);
    feature_per_frame.features.assign(make_move_iterator(features.begin()), make_move_iterator(features.end()));
    feature_frames.push_back(std::move(feature_per_frame));

    ROS_DEBUG_STREAM("feature cost: " << t_features.toc() << " ms");
  }

}

void VectorToDouble() {
  int i, opt_i, pivot_idx = int(estimator_config_.window_size - estimator_config_.opt_window_size);
  P_pivot_ = Ps_[pivot_idx];
  R_pivot_ = Rs_[pivot_idx];
  for (i = 0, opt_i = pivot_idx; i < estimator_config_.opt_window_size + 1; ++i, ++opt_i) {
    para_pose_[i][0] = Ps_[opt_i].x();
    para_pose_[i][1] = Ps_[opt_i].y();
    para_pose_[i][2] = Ps_[opt_i].z();
    Eigen::Quaterniond q{Rs_[opt_i]};
    para_pose_[i][3] = q.x();
    para_pose_[i][4] = q.y();
    para_pose_[i][5] = q.z();
    para_pose_[i][6] = q.w();

    para_speed_bias_[i][0] = Vs_[opt_i].x();
    para_speed_bias_[i][1] = Vs_[opt_i].y();
    para_speed_bias_[i][2] = Vs_[opt_i].z();

    para_speed_bias_[i][3] = Bas_[opt_i].x();
    para_speed_bias_[i][4] = Bas_[opt_i].y();
    para_speed_bias_[i][5] = Bas_[opt_i].z();

    para_speed_bias_[i][6] = Bgs_[opt_i].x();
    para_speed_bias_[i][7] = Bgs_[opt_i].y();
    para_speed_bias_[i][8] = Bgs_[opt_i].z();
  }

  {
    /// base to lidar
    para_ex_pose_[0] = transform_lb_.pos.x();
    para_ex_pose_[1] = transform_lb_.pos.y();
    para_ex_pose_[2] = transform_lb_.pos.z();
    para_ex_pose_[3] = transform_lb_.rot.x();
    para_ex_pose_[4] = transform_lb_.rot.y();
    para_ex_pose_[5] = transform_lb_.rot.z();
    para_ex_pose_[6] = transform_lb_.rot.w();
  }
}

void DoubleToVector() {
// FIXME: do we need to optimize the first state?
// WARNING: not just yaw angle rot_diff; if it is compared with global features, there should be no need for rot_diff

//  Quaterniond origin_R0{Rs_[0]};
  int pivot_idx = int(estimator_config_.window_size - estimator_config_.opt_window_size);
  Vector3d origin_P0 = Ps_[pivot_idx];
  Vector3d origin_R0 = R2ypr(Rs_[pivot_idx]);

  Vector3d origin_R00 = R2ypr(Eigen::Quaterniond(para_pose_[0][6],
                                          para_pose_[0][3],
                                          para_pose_[0][4],
                                          para_pose_[0][5]).normalized().toRotationMatrix());
  // Z-axix R00 to R0, regard para_pose's R as rotate along the Z-axis first
  double y_diff = origin_R0.x() - origin_R00.x();

  //TODO
  Matrix3d rot_diff = ypr2R(Vector3d(y_diff, 0, 0));
  if (abs(abs(origin_R0.y()) - 90) < 1.0 || abs(abs(origin_R00.y()) - 90) < 1.0) {
    ROS_DEBUG("euler singular point!");
    rot_diff = Rs_[pivot_idx] * Eigen::Quaterniond(para_pose_[0][6],
                                            para_pose_[0][3],
                                            para_pose_[0][4],
                                            para_pose_[0][5]).normalized().toRotationMatrix().transpose();
  }

//  DLOG(INFO) << "origin_P0" << origin_P0.transpose();

  {
    Eigen::Vector3d Ps_pivot = Ps_[pivot_idx];
    Eigen::Matrix3d Rs_pivot = Rs_[pivot_idx];
    Twist<double> trans_pivot{Eigen::Quaterniond{Rs_pivot}, Ps_pivot};

    Matrix3d R_opt_pivot = rot_diff * Eigen::Quaterniond(para_pose_[0][6],
                                                  para_pose_[0][3],
                                                  para_pose_[0][4],
                                                  para_pose_[0][5]).normalized().toRotationMatrix();
    Vector3d P_opt_pivot = origin_P0;

    Twist<double> trans_opt_pivot{Eigen::Quaterniond{R_opt_pivot}, P_opt_pivot};
    for (int idx = 0; idx < pivot_idx; ++idx) {
      Twist<double> trans_idx{Eigen::Quaterniond{Rs_[idx]}, Ps_[idx]};
      Twist<double> trans_opt_idx = trans_opt_pivot * trans_pivot.inverse() * trans_idx;
      Ps_[idx] = trans_opt_idx.pos;
      Rs_[idx] = trans_opt_idx.rot.normalized().toRotationMatrix();

      // wrong -- below
//      Twist<double> trans_pivot_idx = trans_pivot.inverse() * trans_idx;
//      Ps_[idx] = rot_diff * trans_pivot_idx.pos + origin_P0;
//      Rs_[idx] = rot_diff * trans_pivot_idx.rot.normalized().toRotationMatrix();
    }

  }

  int i, opt_i;
  for (i = 0, opt_i = pivot_idx; i < estimator_config_.opt_window_size + 1; ++i, ++opt_i) {
//    DLOG(INFO) << "para aft: " << Vector3d(para_pose_[i][0], para_pose_[i][1], para_pose_[i][2]).transpose();

    Rs_[opt_i] = rot_diff * Eigen::Quaterniond(para_pose_[i][6],
                                        para_pose_[i][3],
                                        para_pose_[i][4],
                                        para_pose_[i][5]).normalized().toRotationMatrix();

    Ps_[opt_i] = rot_diff * Vector3d(para_pose_[i][0] - para_pose_[0][0],
                                     para_pose_[i][1] - para_pose_[0][1],
                                     para_pose_[i][2] - para_pose_[0][2]) + origin_P0;

    Vs_[opt_i] = rot_diff * Vector3d(para_speed_bias_[i][0],
                                     para_speed_bias_[i][1],
                                     para_speed_bias_[i][2]);

    Bas_[opt_i] = Vector3d(para_speed_bias_[i][3],
                           para_speed_bias_[i][4],
                           para_speed_bias_[i][5]);

    Bgs_[opt_i] = Vector3d(para_speed_bias_[i][6],
                           para_speed_bias_[i][7],
                           para_speed_bias_[i][8]);
  }
  {
    transform_lb_.pos = Vector3d(para_ex_pose_[0],
                                 para_ex_pose_[1],
                                 para_ex_pose_[2]).template cast<float>();
    transform_lb_.rot = Eigen::Quaterniond(para_ex_pose_[6],
                                    para_ex_pose_[3],
                                    para_ex_pose_[4],
                                    para_ex_pose_[5]).template cast<float>();
  }
}

//优化核心部分 ceres实现滑窗优化
void SolveOptimization() {
    if (cir_buf_count_ < estimator_config_.window_size && estimator_config_.imu_factor) {
    	LOG(ERROR) << "enter optimization before enough count: " << cir_buf_count_ << " < "
               	<< estimator_config_.window_size;
    	return;
  	}

    TicToc tic_toc_opt;

    bool turn_off = true;
    ceres::Problem problem;
  	ceres::LossFunction *loss_function;
    //  loss_function = new ceres::HuberLoss(0.5);
  	loss_function = new ceres::CauchyLoss(1.0);
   	// NOTE: update from laser transform
    //如果没有IMU就直接用scan to map得到的位姿计算增量更新 但本身好像也是用scan to map得到的位姿
  	if (estimator_config_.update_laser_imu) {
    	DLOG(INFO) << "======= bef opt =======";
    	if (!estimator_config_.imu_factor) {
      		Twist<double>
          		incre = (transform_lb_.inverse() * all_laser_transforms_[cir_buf_count_ - 1].second.transform.inverse()
          		* all_laser_transforms_[cir_buf_count_].second.transform * transform_lb_).cast<double>();
      		Ps_[cir_buf_count_] = Rs_[cir_buf_count_ - 1] * incre.pos + Ps_[cir_buf_count_ - 1];
      		Rs_[cir_buf_count_] = Rs_[cir_buf_count_ - 1] * incre.rot;
    	}
  	}

  	vector<FeaturePerFrame> feature_frames;
//lio滑窗构建局部地图 填充feature_frames 里面存了特征拟合参数 同时calculateOdom更新了一次
  	BuildLocalMap(feature_frames);
  	vector<double *> para_ids;
  	//region Add pose and speed bias parameters
  	for (int i = 0; i < estimator_config_.opt_window_size + 1;
    	   ++i) {
    	ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
    	problem.AddParameterBlock(para_pose_[i], leio::SIZE_POSE, local_parameterization);
    	problem.AddParameterBlock(para_speed_bias_[i], leio::SIZE_SPEED_BIAS);
    	para_ids.push_back(para_pose_[i]);
    	para_ids.push_back(para_speed_bias_[i]);
  	}
  //endregion
//这里去掉了外参估计部分
    VectorToDouble();
    //边缘化因子
	vector<ceres::internal::ResidualBlock *> res_ids_marg;
	ceres::internal::ResidualBlock *res_id_marg = NULL;

  //region Marginalization residual
  if (estimator_config_.marginalization_factor) {
    if (last_marginalization_info) {
      // construct new marginlization_factor
      leio::MarginalizationFactor *marginalization_factor = new leio::MarginalizationFactor(last_marginalization_info);
      res_id_marg = problem.AddResidualBlock(marginalization_factor, NULL,
                                             last_marginalization_parameter_blocks);
      res_ids_marg.push_back(res_id_marg);
    }
  }
  //endregion

  //imu预积分因子
  vector<ceres::internal::ResidualBlock *> res_ids_pim;
  if (estimator_config_.imu_factor) {
    for (int i = 0; i < estimator_config_.opt_window_size;
         ++i) {
      int j = i + 1;
      int opt_i = int(estimator_config_.window_size - estimator_config_.opt_window_size + i);
      int opt_j = opt_i + 1;
      if (pre_integrations_[opt_j]->sum_dt_ > 10.0) {
        continue;
      }

      leio::ImuFactor *f = new leio::ImuFactor(pre_integrations_[opt_j]);
      ceres::internal::ResidualBlock *res_id =
          problem.AddResidualBlock(f,
                                   NULL,
                                   para_pose_[i],
                                   para_speed_bias_[i],
                                   para_pose_[j],
                                   para_speed_bias_[j]
          );

      res_ids_pim.push_back(res_id);
    }
  }

  //点云几何关系约束 点面残差 点线残差
  vector<ceres::internal::ResidualBlock *> res_ids_proj;
  if (estimator_config_.point_distance_factor) {
    for (int i = 0; i < estimator_config_.opt_window_size + 1; ++i) {
      int opt_i = int(estimator_config_.window_size - estimator_config_.opt_window_size + i);
      //取特征拟合参数
      FeaturePerFrame &feature_per_frame = feature_frames[opt_i];
      LOG_ASSERT(opt_i == feature_per_frame.id);

      vector<unique_ptr<Feature>> &features = feature_per_frame.features;

      DLOG(INFO) << "features.size(): " << features.size();

      for (int j = 0; j < features.size(); ++j) {
        PointPlaneFeature feature_j;
        features[j]->GetFeature(&feature_j);

        const double &s = feature_j.score;

        const Eigen::Vector3d &p_eigen = feature_j.point;
        const Eigen::Vector4d &coeff_eigen = feature_j.coeffs;

        Eigen::Matrix<double, 6, 6> info_mat_in;

        if (i == 0) {
//          Eigen::Matrix<double, 6, 6> mat_in;
//          PointDistanceFactor *f = new PointDistanceFactor(p_eigen,
//                                                           coeff_eigen,
//                                                           mat_in);
//          ceres::internal::ResidualBlock *res_id =
//              problem.AddResidualBlock(f,
//                                       loss_function,
////                                     NULL,
//                                       para_pose_[i],
//                                       para_ex_pose_);
//
//          res_ids_proj.push_back(res_id);
        } else {
          //todo 这里需要改一下 不估计外参 外参要传进去 但不需要修改
          leio::PivotPointPlaneFactor *f = new leio::PivotPointPlaneFactor(p_eigen,
                                                               coeff_eigen);
          ceres::internal::ResidualBlock *res_id =
              problem.AddResidualBlock(f,
                                       loss_function,
//                                     NULL,
                                       para_pose_[0],
                                       para_pose_[i],
                                       para_ex_pose_);

          res_ids_proj.push_back(res_id);
        }

//      {
//        double **tmp_parameters = new double *[3];
//        tmp_parameters[0] = para_pose_[0];
//        tmp_parameters[1] = para_pose_[i];
//        tmp_parameters[2] = para_ex_pose_;
//        f->Check(tmp_parameters);
//      }
      }
    }
  }

  //todo 这里不想加 不想优化外参 这个先验尝试让外参不变
  if (estimator_config_.prior_factor) {
    {
      Twist<double> trans_tmp = transform_lb_.cast<double>();
      leio::PriorFactor *f = new leio::PriorFactor(trans_tmp.pos, trans_tmp.rot);
      problem.AddResidualBlock(f,
                               NULL,
                               para_ex_pose_);
      //    {
      //      double **tmp_parameters = new double *[1];
      //      tmp_parameters[0] = para_ex_pose_;
      //      f->Check(tmp_parameters);
      //    }
    }
  }

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_SCHUR;
//  options.linear_solver_type = ceres::DENSE_QR;
//  options.num_threads = 8;
  options.trust_region_strategy_type = ceres::DOGLEG;
//  options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
  options.max_num_iterations = 10;
  //options.use_explicit_schur_complement = true;
  //options.minimizer_progress_to_stdout = true;
  //options.use_nonmonotonic_steps = true;
  options.max_solver_time_in_seconds = 0.10;

  //这里会决定是否进行边缘化
  //region residual before optimization
  {
    double cost_pim = 0.0, cost_ppp = 0.0, cost_marg = 0.0;
    ///< Bef
    ceres::Problem::EvaluateOptions e_option;
    if (estimator_config_.imu_factor) {
      e_option.parameter_blocks = para_ids;
      e_option.residual_blocks = res_ids_pim;
      problem.Evaluate(e_option, &cost_pim, NULL, NULL, NULL);
      DLOG(INFO) << "bef_pim: " << cost_pim;

//      if (cost > 1e3 || !convergence_flag_) {
      if (cost_pim > 1e3) {
        turn_off = true;
      } else {
        turn_off = false;
      }
    }
    if (estimator_config_.point_distance_factor) {
      e_option.parameter_blocks = para_ids;
      e_option.residual_blocks = res_ids_proj;
      problem.Evaluate(e_option, &cost_ppp, NULL, NULL, NULL);
      DLOG(INFO) << "bef_proj: " << cost_ppp;
    }
    if (estimator_config_.marginalization_factor) {
      if (last_marginalization_info) {
        e_option.parameter_blocks = para_ids;
        e_option.residual_blocks = res_ids_marg;
        problem.Evaluate(e_option, &cost_marg, NULL, NULL, NULL);
        DLOG(INFO) << "bef_marg: " << cost_marg;
        ///>
      }
    }

    {
      double ratio = cost_marg / (cost_ppp + cost_pim);

      if (!convergence_flag_ && !turn_off && ratio <= 2 && ratio != 0) {
        DLOG(WARNING) << "CONVERGE RATIO: " << ratio;
        convergence_flag_ = true;
      }

      if (!convergence_flag_) {
        ///<
        problem.SetParameterBlockConstant(para_ex_pose_);
        DLOG(WARNING) << "TURN OFF EXTRINSIC AND MARGINALIZATION";
        DLOG(WARNING) << "RATIO: " << ratio;

        if (last_marginalization_info) {
          delete last_marginalization_info;
          last_marginalization_info = nullptr;
        }

        if (res_id_marg) {
          problem.RemoveResidualBlock(res_id_marg);
          res_ids_marg.clear();
        }
      }

    }

  }
  //endregion

  TicToc t_opt;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  DLOG(INFO) << summary.BriefReport();

  ROS_DEBUG_STREAM("t_opt: " << t_opt.toc() << " ms");
  DLOG(INFO) <<"t_opt: " << t_opt.toc() << " ms";

  DoubleToVector();

  //边缘化
  //region Constraint Marginalization
  if (estimator_config_.marginalization_factor && !turn_off) {

    TicToc t_whole_marginalization;

    leio::MarginalizationInfo *marginalization_info = new leio::MarginalizationInfo();

    VectorToDouble();

    if (last_marginalization_info) {
      vector<int> drop_set;
      for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++) {
        if (last_marginalization_parameter_blocks[i] == para_pose_[0] ||
            last_marginalization_parameter_blocks[i] == para_speed_bias_[0])
          drop_set.push_back(i);
      }
      // construct new marginlization_factor
      leio::MarginalizationFactor *marginalization_factor = new leio::MarginalizationFactor(last_marginalization_info);
      leio::ResidualBlockInfo *residual_block_info = new leio::ResidualBlockInfo(marginalization_factor, NULL,
                                                                     last_marginalization_parameter_blocks,
                                                                     drop_set);

      marginalization_info->AddResidualBlockInfo(residual_block_info);
    }

    if (estimator_config_.imu_factor) {
      int pivot_idx = estimator_config_.window_size - estimator_config_.opt_window_size;
      if (pre_integrations_[pivot_idx + 1]->sum_dt_ < 10.0) {
        leio::ImuFactor *imu_factor_ = new leio::ImuFactor(pre_integrations_[pivot_idx + 1]);
        leio::ResidualBlockInfo *residual_block_info = new leio::ResidualBlockInfo(imu_factor_, NULL,
                                                                       vector<double *>{para_pose_[0],
                                                                                        para_speed_bias_[0],
                                                                                        para_pose_[1],
                                                                                        para_speed_bias_[1]},
                                                                       vector<int>{0, 1});
        marginalization_info->AddResidualBlockInfo(residual_block_info);
      }
    }

    if (estimator_config_.point_distance_factor) {
      for (int i = 1; i < estimator_config_.opt_window_size + 1; ++i) {
        int opt_i = int(estimator_config_.window_size - estimator_config_.opt_window_size + i);

        FeaturePerFrame &feature_per_frame = feature_frames[opt_i];
        LOG_ASSERT(opt_i == feature_per_frame.id);

        vector<unique_ptr<Feature>> &features = feature_per_frame.features;

//        DLOG(INFO) << "features.size(): " << features.size();

        for (int j = 0; j < features.size(); ++j) {

          PointPlaneFeature feature_j;
          features[j]->GetFeature(&feature_j);

          const double &s = feature_j.score;

          const Eigen::Vector3d &p_eigen = feature_j.point;
          const Eigen::Vector4d &coeff_eigen = feature_j.coeffs;

          leio::PivotPointPlaneFactor *pivot_point_plane_factor = new leio::PivotPointPlaneFactor(p_eigen,
                                                                                      coeff_eigen);

          leio::ResidualBlockInfo *residual_block_info = new leio::ResidualBlockInfo(pivot_point_plane_factor, loss_function,
                                                                         vector<double *>{para_pose_[0],
                                                                                          para_pose_[i],
                                                                                          para_ex_pose_},
                                                                         vector<int>{0});
          marginalization_info->AddResidualBlockInfo(residual_block_info);

        }

      }
    }

    TicToc t_pre_margin;
    marginalization_info->PreMarginalize();
    ROS_DEBUG("pre marginalization %f ms", t_pre_margin.toc());
    ROS_DEBUG_STREAM("pre marginalization: " << t_pre_margin.toc() << " ms");

    TicToc t_margin;
    marginalization_info->Marginalize();
    ROS_DEBUG("marginalization %f ms", t_margin.toc());
    ROS_DEBUG_STREAM("marginalization: " << t_margin.toc() << " ms");

    std::unordered_map<long, double *> addr_shift;
    for (int i = 1; i < estimator_config_.opt_window_size + 1; ++i) {
      addr_shift[reinterpret_cast<long>(para_pose_[i])] = para_pose_[i - 1];
      addr_shift[reinterpret_cast<long>(para_speed_bias_[i])] = para_speed_bias_[i - 1];
    }

    addr_shift[reinterpret_cast<long>(para_ex_pose_)] = para_ex_pose_;

    vector<double *> parameter_blocks = marginalization_info->GetParameterBlocks(addr_shift);

    if (last_marginalization_info) {
      delete last_marginalization_info;
    }
    last_marginalization_info = marginalization_info;
    last_marginalization_parameter_blocks = parameter_blocks;

    DLOG(INFO) << "whole marginalization costs: " << t_whole_marginalization.toc();
    ROS_DEBUG_STREAM("whole marginalization costs: " << t_whole_marginalization.toc() << " ms");
  }//边缘化结束
  //endregion

//todo 这里和ROS发布消息 tf坐标有关 暂时不用
 // NOTE: update to laser transform
//  if (estimator_config_.update_laser_imu) {
//    DLOG(INFO) << "======= aft opt =======";
//    Twist<double> transform_lb = transform_lb_.cast<double>();
//    Transform &opt_l0_transform = opt_transforms_[0];
//    int opt_0 = int(estimator_config_.window_size - estimator_config_.opt_window_size + 0);
//    Eigen::Quaterniond rot_l0(Rs_[opt_0] * transform_lb.rot.conjugate().normalized());
//    Eigen::Vector3d pos_l0 = Ps_[opt_0] - rot_l0 * transform_lb.pos;
//    opt_l0_transform = Twist<double>{rot_l0, pos_l0}.cast<float>(); // for updating the map

//    vector<Transform> imu_poses, lidar_poses;

//    for (int i = 0; i < estimator_config_.opt_window_size + 1; ++i) {
//      int opt_i = int(estimator_config_.window_size - estimator_config_.opt_window_size + i);

//      Eigen::Quaterniond rot_li(Rs_[opt_i] * transform_lb.rot.conjugate().normalized());
//      Eigen::Vector3d pos_li = Ps_[opt_i] - rot_li * transform_lb.pos;
//      Twist<double> transform_li = Twist<double>(rot_li, pos_li);

//      Twist<double> transform_bi = Twist<double>(Eigen::Quaterniond(Rs_[opt_i]), Ps_[opt_i]);
//      imu_poses.push_back(transform_bi.cast<float>());
//      lidar_poses.push_back(transform_li.cast<float>());

//    }

//    DLOG(INFO) << "velocity: " << Vs_.last().norm();
//    DLOG(INFO) << "transform_lb_: " << transform_lb_;

//    ROS_DEBUG_STREAM("lb in world: " << (rot_l0.normalized() * transform_lb.pos).transpose());

//    {
//      geometry_msgs::PoseStamped ex_lb_msg;
//      ex_lb_msg.header = Headers_.last();
//      ex_lb_msg.pose.position.x = transform_lb.pos.x();
//      ex_lb_msg.pose.position.y = transform_lb.pos.y();
//      ex_lb_msg.pose.position.z = transform_lb.pos.z();
//      ex_lb_msg.pose.orientation.w = transform_lb.rot.w();
//      ex_lb_msg.pose.orientation.x = transform_lb.rot.x();
//      ex_lb_msg.pose.orientation.y = transform_lb.rot.y();
//      ex_lb_msg.pose.orientation.z = transform_lb.rot.z();
//      pub_extrinsic_.publish(ex_lb_msg);

//      int pivot_idx = estimator_config_.window_size - estimator_config_.opt_window_size;

//      Eigen::Vector3d Ps_pivot = Ps_[pivot_idx];
//      Eigen::Matrix3d Rs_pivot = Rs_[pivot_idx];

//      Eigen::Quaterniond rot_pivot(Rs_pivot * transform_lb.rot.inverse());
//      Eigen::Vector3d pos_pivot = Ps_pivot - rot_pivot * transform_lb.pos;
//      PublishCloudMsg(pub_local_surf_points_,
//                      *surf_stack_[pivot_idx + 1],
//                      Headers_[pivot_idx + 1].stamp,
//                      "/laser_local");

//      PublishCloudMsg(pub_local_corner_points_,
//                      *corner_stack_[pivot_idx + 1],
//                      Headers_[pivot_idx + 1].stamp,
//                      "/laser_local");

//      PublishCloudMsg(pub_local_full_points_,
//                      *full_stack_[pivot_idx + 1],
//                      Headers_[pivot_idx + 1].stamp,
//                      "/laser_local");

//      PublishCloudMsg(pub_map_surf_points_,
//                      *local_surf_points_filtered_ptr_,
//                      Headers_.last().stamp,
//                      "/laser_local");

// #ifdef USE_CORNER
//      PublishCloudMsg(pub_map_corner_points_,
//                      *local_corner_points_filtered_ptr_,
//                      Headers_.last().stamp,
//                      "/laser_local");
// #endif

//      laser_local_trans_.setOrigin(tf::Vector3{pos_pivot.x(), pos_pivot.y(), pos_pivot.z()});
//      laser_local_trans_.setRotation(tf::Quaternion{rot_pivot.x(), rot_pivot.y(), rot_pivot.z(), rot_pivot.w()});
//      laser_local_trans_.stamp_ = Headers_.last().stamp;
//      tf_broadcaster_est_.sendTransform(laser_local_trans_);

//      Eigen::Vector3d Ps_last = Ps_.last();
//      Eigen::Matrix3d Rs_last = Rs_.last();

//      Eigen::Quaterniond rot_last(Rs_last * transform_lb.rot.inverse());
//      Eigen::Vector3d pos_last = Ps_last - rot_last * transform_lb.pos;

//      Eigen::Quaterniond rot_predict = (rot_pivot.inverse() * rot_last).normalized();
//      Eigen::Vector3d pos_predict = rot_pivot.inverse() * (Ps_last - Ps_pivot);

//      PublishCloudMsg(pub_predict_surf_points_, *(surf_stack_.last()), Headers_.last().stamp, "/laser_predict");
//      PublishCloudMsg(pub_predict_full_points_, *(full_stack_.last()), Headers_.last().stamp, "/laser_predict");

//      {
//        // NOTE: full stack into end of the scan
// //        PointCloudPtr tmp_points_ptr = boost::make_shared<PointCloud>(PointCloud());
// //        *tmp_points_ptr = *(full_stack_.last());
// //        TransformToEnd(tmp_points_ptr, transform_es_, 10);
// //        PublishCloudMsg(pub_predict_corrected_full_points_,
// //                        *tmp_points_ptr,
// //                        Headers_.last().stamp,
// //                        "/laser_predict");

//        TransformToEnd(full_stack_.last(), transform_es_, 10, true);
//        PublishCloudMsg(pub_predict_corrected_full_points_,
//                        *(full_stack_.last()),
//                        Headers_.last().stamp,
//                        "/laser_predict");
//      }

// #ifdef USE_CORNER
//      PublishCloudMsg(pub_predict_corner_points_, *(corner_stack_.last()), Headers_.last().stamp, "/laser_predict");
// #endif
//      laser_predict_trans_.setOrigin(tf::Vector3{pos_predict.x(), pos_predict.y(), pos_predict.z()});
//      laser_predict_trans_.setRotation(tf::Quaternion{rot_predict.x(), rot_predict.y(), rot_predict.z(),
//                                                      rot_predict.w()});
//      laser_predict_trans_.stamp_ = Headers_.last().stamp;
//      tf_broadcaster_est_.sendTransform(laser_predict_trans_);
//    }

//  }

  DLOG(INFO) << "tic_toc_opt: " << tic_toc_opt.toc() << " ms";
  ROS_DEBUG_STREAM("tic_toc_opt: " << tic_toc_opt.toc() << " ms");
}

//滑窗
void SlideWindow() { // NOTE: this function is only for the states and the local map
  {
    if (init_local_map_) {
      int pivot_idx = estimator_config_.window_size - estimator_config_.opt_window_size;

      Twist<double> transform_lb = transform_lb_.cast<double>();

      Eigen::Vector3d Ps_pivot = Ps_[pivot_idx];
      Eigen::Matrix3d Rs_pivot = Rs_[pivot_idx];

      Eigen::Quaterniond rot_pivot(Rs_pivot * transform_lb.rot.inverse());
      Eigen::Vector3d pos_pivot = Ps_pivot - rot_pivot * transform_lb.pos;

      Twist<double> transform_pivot = Twist<double>(rot_pivot, pos_pivot);

      PointCloudPtr transformed_cloud_surf_ptr(new PointCloud);
      PointCloudPtr transformed_cloud_corner_ptr(new PointCloud);
      PointCloud filtered_surf_points;
      PointCloud filtered_corner_points;

      int i = pivot_idx + 1; // the index of the next pivot
      Eigen::Vector3d Ps_i = Ps_[i];
      Eigen::Matrix3d Rs_i = Rs_[i];

      Eigen::Quaterniond rot_li(Rs_i * transform_lb.rot.inverse());
      Eigen::Vector3d pos_li = Ps_i - rot_li * transform_lb.pos;

      Twist<double> transform_li = Twist<double>(rot_li, pos_li);
      Eigen::Affine3f transform_i_pivot = (transform_li.inverse() * transform_pivot).cast<float>().transform();
      pcl::ExtractIndices<PointT> extract;

      pcl::transformPointCloud(*(surf_stack_[pivot_idx]), *transformed_cloud_surf_ptr, transform_i_pivot);
      pcl::PointIndices::Ptr inliers_surf(new pcl::PointIndices());

      for (int i = 0; i < size_surf_stack_[0]; ++i) {
        inliers_surf->indices.push_back(i);
      }
      extract.setInputCloud(transformed_cloud_surf_ptr);
      extract.setIndices(inliers_surf);
      extract.setNegative(true);
      extract.filter(filtered_surf_points);

      filtered_surf_points += *(surf_stack_[i]);

      *(surf_stack_[i]) = filtered_surf_points;

#ifdef USE_CORNER
      pcl::transformPointCloud(*(corner_stack_[pivot_idx]), *transformed_cloud_corner_ptr, transform_i_pivot);
      pcl::PointIndices::Ptr inliers_corner(new pcl::PointIndices());

      for (int i = 0; i < size_corner_stack_[0]; ++i) {
        inliers_corner->indices.push_back(i);
      }
      extract.setInputCloud(transformed_cloud_corner_ptr);
      extract.setIndices(inliers_corner);
      extract.setNegative(true);
      extract.filter(filtered_corner_points);

      filtered_corner_points += *(corner_stack_[i]);

      *(corner_stack_[i]) = filtered_corner_points;
#endif
    }
  }

  dt_buf_.push(vector<double>());
  linear_acceleration_buf_.push(vector<Vector3d>());
  angular_velocity_buf_.push(vector<Vector3d>());

//  Headers_.push(Headers_[cir_buf_count_]);
  Ps_.push(Ps_[cir_buf_count_]);
  Vs_.push(Vs_[cir_buf_count_]);
  Rs_.push(Rs_[cir_buf_count_]);
  Bas_.push(Bas_[cir_buf_count_]);
  Bgs_.push(Bgs_[cir_buf_count_]);

//  pre_integrations_.push(std::make_shared<IntegrationBase>(IntegrationBase(acc_last_, gyr_last_,
//                                                                           Bas_[cir_buf_count_],
//                                                                           Bgs_[cir_buf_count_],
//                                                                           estimator_config_.pim_config)));

//  all_laser_transforms_.push(all_laser_transforms_[cir_buf_count_]);

// TODO: slide new lidar points
}

bool RunInitialization() {
  // NOTE: check IMU observibility, adapted from VINS-mono
  //计算滑窗内IMU平均加速度和方差 防止方差过小 代表IMU激励不充分
  {
    leio::PairTimeLaserTransform laser_trans_i, laser_trans_j;
    Vector3d sum_g;

    for (size_t i = 0; i < estimator_config_.window_size;
         ++i) {
      laser_trans_j = all_laser_transforms_[i + 1];

      double dt = laser_trans_j.second.pre_integration->sum_dt_;
      Vector3d tmp_g = laser_trans_j.second.pre_integration->delta_v_ / dt;
      sum_g += tmp_g;
    }

    Vector3d aver_g;
    aver_g = sum_g * 1.0 / (estimator_config_.window_size);
    double var = 0;

    for (size_t i = 0; i < estimator_config_.window_size;
         ++i) {
      laser_trans_j = all_laser_transforms_[i + 1];
      double dt = laser_trans_j.second.pre_integration->sum_dt_;
      Vector3d tmp_g = laser_trans_j.second.pre_integration->delta_v_ / dt;
      var += (tmp_g - aver_g).transpose() * (tmp_g - aver_g);
    }

    var = sqrt(var / (estimator_config_.window_size));

    DLOG(INFO) << "IMU variation: " << var;

    if (var < 0.25) {
      ROS_INFO("IMU excitation not enough!");
      return false;
    }
  }

  Eigen::Vector3d g_vec_in_laser;
  bool init_result
      = leio::ImuInitializer::Initialization(all_laser_transforms_, Vs_, Bas_, Bgs_, g_vec_in_laser, transform_lb_, R_WI_);
//  init_result = false;

//  Q_WI_ = R_WI_;
//  g_vec_ = R_WI_ * Eigen::Vector3d(0.0, 0.0, -1.0) * g_norm_;
//  g_vec_ = Eigen::Vector3d(0.0, 0.0, -1.0) * g_norm_;

  // TODO: update states Ps_
  for (size_t i = 0; i < estimator_config_.window_size + 1;
       ++i) {
    const Transform &trans_li = all_laser_transforms_[i].second.transform;
    Transform trans_bi = trans_li * transform_lb_;
    Ps_[i] = trans_bi.pos.template cast<double>();
    Rs_[i] = trans_bi.rot.normalized().toRotationMatrix().template cast<double>();
  }

  Matrix3d R0 = R_WI_.transpose();

  double yaw = R2ypr(R0 * Rs_[0]).x();
  R0 = (ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0).eval();

  R_WI_ = R0.transpose();
  Q_WI_ = R_WI_;

  g_vec_ = R0 * g_vec_in_laser;

  for (int i = 0; i <= cir_buf_count_; i++) {
    pre_integrations_[i]->Repropagate(Bas_[i], Bgs_[i]);
  }

  Matrix3d rot_diff = R0;
  for (int i = 0; i <= cir_buf_count_; i++) {
    Ps_[i] = (rot_diff * Ps_[i]).eval();
    Rs_[i] = (rot_diff * Rs_[i]).eval();
    Vs_[i] = (rot_diff * Vs_[i]).eval();
  }

  DLOG(WARNING) << "refined gravity:  " << g_vec_.transpose();

  if (!init_result) {
    DLOG(WARNING) << "Imu initialization failed!";
    return false;
  } else {
    DLOG(WARNING) << "Imu initialization successful!";
    return true;
  }
}

void SetInitFlag(bool set_init) {
  imu_inited_ = set_init;
}

size_t ToIndex(int i, int j, int k) {
  return i + laser_cloud_length_ * j + laser_cloud_length_ * laser_cloud_width_ * k;
}

void FromIndex(const size_t &index, int &i, int &j, int &k) {
  int residual = index % (laser_cloud_length_ * laser_cloud_width_);
  k = index / (laser_cloud_length_ * laser_cloud_width_);
  j = residual / laser_cloud_length_;
  i = residual % laser_cloud_length_;
}

void UpdateMapDatabase(PointCloudPtr margin_corner_stack_downsampled,
                                     PointCloudPtr margin_surf_stack_downsampled,
                                     std::vector<size_t> margin_valid_idx,
                                     const Transform &margin_transform_tobe_mapped,
                                     const CubeCenter &margin_cube_center) {

  PointT point_sel;
  size_t margin_corner_stack_ds_size = margin_corner_stack_downsampled->points.size();
  size_t margin_surf_stack_ds_size = margin_surf_stack_downsampled->points.size();
  size_t margin_valid_size = margin_valid_idx.size();

  for (int i = 0; i < margin_corner_stack_ds_size; ++i) {
    //转到世界系
    PointAssociateToMap(margin_corner_stack_downsampled->points[i], point_sel, margin_transform_tobe_mapped);

    int cube_i = int((point_sel.x + 25.0) / 50.0) + laser_cloud_cen_length_;
    int cube_j = int((point_sel.y + 25.0) / 50.0) + laser_cloud_cen_width_;
    int cube_k = int((point_sel.z + 25.0) / 50.0) + laser_cloud_cen_height_;

    if (point_sel.x + 25.0 < 0) --cube_i;
    if (point_sel.y + 25.0 < 0) --cube_j;
    if (point_sel.z + 25.0 < 0) --cube_k;

    if (cube_i >= 0 && cube_i < laser_cloud_length_ &&
        cube_j >= 0 && cube_j < laser_cloud_width_ &&
        cube_k >= 0 && cube_k < laser_cloud_height_) {
      size_t cube_idx = ToIndex(cube_i, cube_j, cube_k);
      //todo 需要确认一下一开始存入的逻辑是否正确
      laser_cloud_corner_array_[cube_idx]->push_back(point_sel);
    }
  }

  // store down sized surface stack points in corresponding cube clouds
  for (int i = 0; i < margin_surf_stack_ds_size; ++i) {
    PointAssociateToMap(margin_surf_stack_downsampled->points[i], point_sel, margin_transform_tobe_mapped);

    int cube_i = int((point_sel.x + 25.0) / 50.0) + laser_cloud_cen_length_;
    int cube_j = int((point_sel.y + 25.0) / 50.0) + laser_cloud_cen_width_;
    int cube_k = int((point_sel.z + 25.0) / 50.0) + laser_cloud_cen_height_;

    if (point_sel.x + 25.0 < 0) --cube_i;
    if (point_sel.y + 25.0 < 0) --cube_j;
    if (point_sel.z + 25.0 < 0) --cube_k;

    if (cube_i >= 0 && cube_i < laser_cloud_length_ &&
        cube_j >= 0 && cube_j < laser_cloud_width_ &&
        cube_k >= 0 && cube_k < laser_cloud_height_) {
      size_t cube_idx = ToIndex(cube_i, cube_j, cube_k);
      laser_cloud_surf_array_[cube_idx]->push_back(point_sel);
    }
  }

  // TODO: varlidate margin_valid_idx
  // down size all valid (within field of view) feature cube clouds
  for (int i = 0; i < margin_valid_size; ++i) {
    size_t index = margin_valid_idx[i];

//    DLOG(INFO) << "index before: " << index;

    int last_i, last_j, last_k;

    FromIndex(index, last_i, last_j, last_k);

    float center_x = 50.0f * (last_i - margin_cube_center.laser_cloud_cen_length);
    float center_y = 50.0f * (last_j - margin_cube_center.laser_cloud_cen_width);
    float center_z = 50.0f * (last_k - margin_cube_center.laser_cloud_cen_height); // NOTE: center of the margin cube

    int cube_i = int((center_x + 25.0) / 50.0) + laser_cloud_cen_length_;
    int cube_j = int((center_y + 25.0) / 50.0) + laser_cloud_cen_width_;
    int cube_k = int((center_z + 25.0) / 50.0) + laser_cloud_cen_height_;

    if (center_x + 25.0 < 0) --cube_i;
    if (center_y + 25.0 < 0) --cube_j;
    if (center_z + 25.0 < 0) --cube_k;

    if (cube_i >= 0 && cube_i < laser_cloud_length_ &&
        cube_j >= 0 && cube_j < laser_cloud_width_ &&
        cube_k >= 0 && cube_k < laser_cloud_height_) {

      index = ToIndex(cube_i, cube_j, cube_k); // NOTE: update to current index

//      DLOG(INFO) << "index after: " << index;

      laser_cloud_corner_downsampled_array_[index]->clear();
      down_size_filter_corner_.setInputCloud(laser_cloud_corner_array_[index]);
      down_size_filter_corner_.filter(*laser_cloud_corner_downsampled_array_[index]);

      laser_cloud_surf_downsampled_array_[index]->clear();
      down_size_filter_surf_.setInputCloud(laser_cloud_surf_array_[index]);
      down_size_filter_surf_.filter(*laser_cloud_surf_downsampled_array_[index]);

      // swap cube clouds for next processing
      laser_cloud_corner_array_[index].swap(laser_cloud_corner_downsampled_array_[index]);
      laser_cloud_surf_array_[index].swap(laser_cloud_surf_downsampled_array_[index]);
    }

  }

}

//处理完lidar_info消息后做LIO
void ProcessLaserOdom(const Transform &transform_in, const std_msgs::Header &header) {

  ROS_DEBUG(">>>>>>> new laser odom coming <<<<<<<");

  ++laser_odom_recv_count_;

  if (stage_flag_ != INITED
      && laser_odom_recv_count_ % estimator_config_.init_window_factor != 0) { /// better for initialization
    return;
  }

  Headers_.push(header);

  // TODO: LaserFrame Object
  // LaserFrame laser_frame(laser, header.stamp.toSec());

  leio::LaserTransform laser_transform(header.stamp.toSec(), transform_in);

  laser_transform.pre_integration = tmp_pre_integration_;
  pre_integrations_.push(tmp_pre_integration_);

  // reset tmp_pre_integration_
  tmp_pre_integration_.reset();
  tmp_pre_integration_ = std::make_shared<leio::IntegrationBase>(leio::IntegrationBase(acc_last_,
                                                                           gyr_last_,
                                                                           Bas_[cir_buf_count_],
                                                                           Bgs_[cir_buf_count_],
                                                                           estimator_config_.pim_config));
  all_laser_transforms_.push(make_pair(header.stamp.toSec(), laser_transform));

  // TODO: check extrinsic parameter estimation
  // NOTE: push PointMapping's point_coeff_map_
  ///> optimization buffers
  opt_point_coeff_mask_.push(false); // default new frame
  opt_point_coeff_map_.push(score_point_coeff_);
  opt_cube_centers_.push(CubeCenter{laser_cloud_cen_length_, laser_cloud_cen_width_, laser_cloud_cen_height_});
  opt_transforms_.push(laser_transform.transform);
  opt_valid_idx_.push(laser_cloud_valid_idx_);

  //这里往滑窗里放点云
  if (stage_flag_ != INITED || (!estimator_config_.enable_deskew && !estimator_config_.cutoff_deskew)) {
    surf_stack_.push(boost::make_shared<PointCloud>(*laser_cloud_surf_stack_downsampled_));
    size_surf_stack_.push(laser_cloud_surf_stack_downsampled_->size());

    corner_stack_.push(boost::make_shared<PointCloud>(*laser_cloud_corner_stack_downsampled_));
    size_corner_stack_.push(laser_cloud_corner_stack_downsampled_->size());
  }

  //todo 不加全点云
  // full_stack_.push(boost::make_shared<PointCloud>(*full_cloud_));

  opt_surf_stack_.push(surf_stack_.last());
  opt_corner_stack_.push(corner_stack_.last());

  //todo ?
  opt_matP_.push(matP_.cast<double>());
  ///< optimization buffers

  if (estimator_config_.run_optimization) {
    switch (stage_flag_) {
      case NOT_INITED: {

        {
          DLOG(INFO) << "surf_stack_: " << surf_stack_.size();
          DLOG(INFO) << "corner_stack_: " << corner_stack_.size();
          DLOG(INFO) << "pre_integrations_: " << pre_integrations_.size();
          DLOG(INFO) << "Ps_: " << Ps_.size();
          DLOG(INFO) << "size_surf_stack_: " << size_surf_stack_.size();
          DLOG(INFO) << "size_corner_stack_: " << size_corner_stack_.size();
          DLOG(INFO) << "all_laser_transforms_: " << all_laser_transforms_.size();
        }

        bool init_result = false;
        if (cir_buf_count_ == estimator_config_.window_size) {
          // tic_toc_.Tic();

          if (!estimator_config_.imu_factor) {
            init_result = true;
            // TODO: update states Ps_
            for (size_t i = 0; i < estimator_config_.window_size + 1;
                 ++i) {
              const Transform &trans_li = all_laser_transforms_[i].second.transform;
              //lidar系位姿变IMU系位姿
              Transform trans_bi = trans_li * transform_lb_;
              Ps_[i] = trans_bi.pos.template cast<double>();
              Rs_[i] = trans_bi.rot.normalized().toRotationMatrix().template cast<double>();
            }
          } else {

//            if (extrinsic_stage_ == 2) {
//              // TODO: move before initialization
//              bool extrinsic_result = ImuInitializer::EstimateExtrinsicRotation(all_laser_transforms_, transform_lb_);
//              LOG(INFO) << ">>>>>>> extrinsic calibration"
//                        << (extrinsic_result ? " successful"
//                                             : " failed")
//                        << " <<<<<<<";
//              if (extrinsic_result) {
//                extrinsic_stage_ = 1;
//                DLOG(INFO) << "change extrinsic stage to 1";
//              }
//            }

            if (extrinsic_stage_ != 2 && (header.stamp.toSec() - initial_time_) > 0.1) {
              DLOG(INFO) << "EXTRINSIC STAGE: " << extrinsic_stage_;
              init_result = RunInitialization();
              initial_time_ = header.stamp.toSec();
            }
          }

          // DLOG(INFO) << "initialization time: " << tic_toc_.toc() << " ms";

          if (init_result) {
            stage_flag_ = INITED;
            SetInitFlag(true);

            Q_WI_ = R_WI_;
//            wi_trans_.setRotation(tf::Quaternion{Q_WI_.x(), Q_WI_.y(), Q_WI_.z(), Q_WI_.w()});

            ROS_WARN_STREAM(">>>>>>> IMU initialized <<<<<<<");

          //  if (estimator_config_.enable_deskew || estimator_config_.cutoff_deskew) {
          //    ros::ServiceClient client = nh_.serviceClient<std_srvs::SetBool>("/enable_odom");
          //    std_srvs::SetBool srv;
          //    srv.request.data = 0;
          //    if (client.call(srv)) {
          //      DLOG(INFO) << "TURN OFF THE ORIGINAL LASER ODOM";
          //    } else {
          //      LOG(FATAL) << "FAILED TO CALL TURNING OFF THE ORIGINAL LASER ODOM";
          //    }
          //  }

            for (size_t i = 0; i < estimator_config_.window_size + 1;
                 ++i) {
              Twist<double> transform_lb = transform_lb_.cast<double>();
              //lidar位姿
              Eigen::Quaterniond Rs_li(Rs_[i] * transform_lb.rot.inverse());
              Eigen::Vector3d Ps_li = Ps_[i] - Rs_li * transform_lb.pos;

              Twist<double> trans_li{Rs_li, Ps_li};

              DLOG(INFO) << "TEST trans_li " << i << ": " << trans_li;
              DLOG(INFO) << "TEST all_laser_transforms " << i << ": " << all_laser_transforms_[i].second.transform;
            }

            SolveOptimization();

            SlideWindow();

            for (size_t i = 0; i < estimator_config_.window_size + 1;
                 ++i) {
              const Transform &trans_li = all_laser_transforms_[i].second.transform;
              Transform trans_bi = trans_li * transform_lb_;
              DLOG(INFO) << "TEST " << i << ": " << trans_bi.pos.transpose();
            }

            for (size_t i = 0; i < estimator_config_.window_size + 1;
                 ++i) {
              Twist<double> transform_lb = transform_lb_.cast<double>();

              Eigen::Quaterniond Rs_li(Rs_[i] * transform_lb.rot.inverse());
              Eigen::Vector3d Ps_li = Ps_[i] - Rs_li * transform_lb.pos;

              Twist<double> trans_li{Rs_li, Ps_li};

              DLOG(INFO) << "TEST trans_li " << i << ": " << trans_li;
            }

            // WARNING

          } else {
            SlideWindow();
          }
        } else {
          DLOG(INFO) << "Ps size: " << Ps_.size();
          DLOG(INFO) << "pre size: " << pre_integrations_.size();
          DLOG(INFO) << "all_laser_transforms_ size: " << all_laser_transforms_.size();

          SlideWindow();

          DLOG(INFO) << "Ps size: " << Ps_.size();
          DLOG(INFO) << "pre size: " << pre_integrations_.size();

          ++cir_buf_count_;
        }

        opt_point_coeff_mask_.last() = true;

        break;
      }
      case INITED: {
        if (opt_point_coeff_map_.size() == estimator_config_.opt_window_size + 1) {
          //这里去掉了imu信息 点云去畸变
          DLOG(INFO) << ">>>>>>> solving optimization <<<<<<<";
          SolveOptimization();

          if (!opt_point_coeff_mask_.first()) {
            UpdateMapDatabase(opt_corner_stack_.first(),
                              opt_surf_stack_.first(),
                              opt_valid_idx_.first(),
                              opt_transforms_.first(),
                              opt_cube_centers_.first());

            DLOG(INFO) << "all_laser_transforms_: " << all_laser_transforms_[estimator_config_.window_size
                - estimator_config_.opt_window_size].second.transform;
            DLOG(INFO) << "opt_transforms_: " << opt_transforms_.first();
          }
        } else {
          LOG(ERROR) << "opt_point_coeff_map_.size(): " << opt_point_coeff_map_.size()
            << " != estimator_config_.opt_window_size + 1: " << estimator_config_.opt_window_size + 1;
        }
//      PublishResults();
        SlideWindow();

//        {
//          int pivot_idx = estimator_config_.window_size - estimator_config_.opt_window_size;
//          local_odom_.header.stamp = Headers_[pivot_idx + 1].stamp;
//          local_odom_.header.seq += 1;
//          Twist<double> transform_lb = transform_lb_.cast<double>();
//          Eigen::Vector3d Ps_pivot = Ps_[pivot_idx];
//          Eigen::Matrix3d Rs_pivot = Rs_[pivot_idx];
//          Eigen::Quaterniond rot_pivot(Rs_pivot * transform_lb.rot.inverse());
//          Eigen::Vector3d pos_pivot = Ps_pivot - rot_pivot * transform_lb.pos;
//          Twist<double> transform_pivot = Twist<double>(rot_pivot, pos_pivot);
//          local_odom_.pose.pose.orientation.x = transform_pivot.rot.x();
//          local_odom_.pose.pose.orientation.y = transform_pivot.rot.y();
//          local_odom_.pose.pose.orientation.z = transform_pivot.rot.z();
//          local_odom_.pose.pose.orientation.w = transform_pivot.rot.w();
//          local_odom_.pose.pose.position.x = transform_pivot.pos.x();
//          local_odom_.pose.pose.position.y = transform_pivot.pos.y();
//          local_odom_.pose.pose.position.z = transform_pivot.pos.z();
//          pub_local_odom_.publish(local_odom_);
//
//          laser_odom_.header.stamp = header.stamp;
//          laser_odom_.header.seq += 1;
//          Eigen::Vector3d Ps_last = Ps_.last();
//          Eigen::Matrix3d Rs_last = Rs_.last();
//          Eigen::Quaterniond rot_last(Rs_last * transform_lb.rot.inverse());
//          Eigen::Vector3d pos_last = Ps_last - rot_last * transform_lb.pos;
//          Twist<double> transform_last = Twist<double>(rot_last, pos_last);
//          laser_odom_.pose.pose.orientation.x = transform_last.rot.x();
//          laser_odom_.pose.pose.orientation.y = transform_last.rot.y();
//          laser_odom_.pose.pose.orientation.z = transform_last.rot.z();
//          laser_odom_.pose.pose.orientation.w = transform_last.rot.w();
//          laser_odom_.pose.pose.position.x = transform_last.pos.x();
//          laser_odom_.pose.pose.position.y = transform_last.pos.y();
//          laser_odom_.pose.pose.position.z = transform_last.pos.z();
//          //滑窗内最新帧位姿
//          pub_laser_odom_.publish(laser_odom_);
//
//          //lidar里程计路径
//          geometry_msgs::PoseStamped pose_stamped;
//          pose_stamped.header.stamp = laser_odom_.header.stamp;
//          pose_stamped.header.frame_id = "world";
//          pose_stamped.pose.position.x = laser_odom_.pose.pose.position.x;
//          pose_stamped.pose.position.y = laser_odom_.pose.pose.position.y;
//          pose_stamped.pose.position.z = laser_odom_.pose.pose.position.z;
//          pose_stamped.pose.orientation.x = laser_odom_.pose.pose.orientation.x;
//          pose_stamped.pose.orientation.y = laser_odom_.pose.pose.orientation.y;
//          pose_stamped.pose.orientation.z = laser_odom_.pose.pose.orientation.z;
//          pose_stamped.pose.orientation.w = laser_odom_.pose.pose.orientation.w;
//          global_path.poses.push_back(pose_stamped);
//          pub_path.publish(global_path);
//
//          //保存轨迹，path_save是文件目录,txt文件提前建好 tum格式 time x y z
//          std::ofstream pose1("/media/ctx/0BE20E8D0BE20E8D/dataset/result/lio-mapping/result_0018.txt", std::ios::app);
//          pose1.setf(std::ios::scientific, std::ios::floatfield);
//          //kitti数据集转换tum格式的数据是18位
//          pose1.precision(9);
//          //第一个激光帧时间 static变量 只赋值一次
//          static double timeStart = laser_odom_.header.stamp.toSec();
//          auto T1 =ros::Time().fromSec(timeStart) ;
//
//          pose1<< laser_odom_.header.stamp -T1<< " "
//              << -laser_odom_.pose.pose.position.x << " "
//              << -laser_odom_.pose.pose.position.y << " "
//              << -laser_odom_.pose.pose.position.z << " "
//              << laser_odom_.pose.pose.orientation.x << " "
//              << laser_odom_.pose.pose.orientation.y << " "
//              << laser_odom_.pose.pose.orientation.z << " "
//              << laser_odom_.pose.pose.orientation.w << std::endl;
//          pose1.close();
//        }

        break;
      }//end INITED
      default: {
        break;
      }

    }
  }
  // std::cout << "发布wi_trans_" <<std::endl;
//  wi_trans_.setRotation(tf::Quaternion{Q_WI_.x(), Q_WI_.y(), Q_WI_.z(), Q_WI_.w()});
//  wi_trans_.stamp_ = header.stamp;
//  tf_broadcaster_est_.sendTransform(wi_trans_);

}

void LidarInfoHandler(const LidarInfo &lidar_info) {

  TicToc tic_toc_decoder;

  //位姿取出来
  {
    transform_sum_.pos.x() = lidar_info.laser_odometry_->pose.pose.position.x;
    transform_sum_.pos.y() = lidar_info.laser_odometry_->pose.pose.position.y;
    transform_sum_.pos.z() = lidar_info.laser_odometry_->pose.pose.position.z;
    transform_sum_.rot.x() = lidar_info.laser_odometry_->pose.pose.orientation.x;
    transform_sum_.rot.y() = lidar_info.laser_odometry_->pose.pose.orientation.y;
    transform_sum_.rot.z() = lidar_info.laser_odometry_->pose.pose.orientation.z;
    transform_sum_.rot.w() = lidar_info.laser_odometry_->pose.pose.orientation.w;
  }

  //点云取出来
  {
    laser_cloud_corner_last_->clear();
    laser_cloud_surf_last_->clear();
    full_cloud_->clear();

    // 将 ROS 消息转换为 PCL 点云
    pcl::fromROSMsg(*lidar_info.surf_cloud_, *laser_cloud_surf_last_);
    pcl::fromROSMsg(*lidar_info.edge_cloud_, *laser_cloud_corner_last_);
    //todo 没加full_cloud_
  }

  time_laser_cloud_corner_last_ = lidar_info.laser_odometry_->header.stamp;
  time_laser_cloud_surf_last_ = lidar_info.laser_odometry_->header.stamp;
  time_laser_full_cloud_ = lidar_info.laser_odometry_->header.stamp;
  time_laser_odometry_ = lidar_info.laser_odometry_->header.stamp;

  new_laser_cloud_corner_last_ = true;
  new_laser_cloud_surf_last_ = true;
  new_laser_full_cloud_ = true;
  new_laser_odometry_ = true;

  DLOG(INFO) << "decode lidar_info time: " << tic_toc_decoder.toc() << " ms";
}

void TransformAssociateToMap() {
  Transform transform_incre(transform_bef_mapped_.inverse() * transform_sum_.transform());
  transform_tobe_mapped_ = transform_tobe_mapped_ * transform_incre;
}

bool HasNewData() {
  return new_laser_cloud_corner_last_ && new_laser_cloud_surf_last_ &&
      new_laser_full_cloud_ && new_laser_odometry_ &&
      fabs((time_laser_cloud_corner_last_ - time_laser_odometry_).toSec()) < 0.005 &&
      fabs((time_laser_cloud_surf_last_ - time_laser_odometry_).toSec()) < 0.005 &&
      fabs((time_laser_full_cloud_ - time_laser_odometry_).toSec()) < 0.005;
}

void Reset() {
  new_laser_cloud_corner_last_ = false;
  new_laser_cloud_surf_last_ = false;
  new_laser_full_cloud_ = false;
  new_laser_odometry_ = false;
}

void TransformUpdate() {
  transform_bef_mapped_ = transform_sum_;
  transform_aft_mapped_ = transform_tobe_mapped_;
}

void OptimizeTransformTobeMapped() {
  //todo 这里可能需要修改 用哪种点云
  if (laser_cloud_corner_from_map_->points.size() <= 10 || laser_cloud_surf_from_map_->points.size() <= 100) {
    return;
  }

  PointT point_sel, point_ori, coeff, abs_coeff;

  std::vector<int> point_search_idx(5, 0);
  std::vector<float> point_search_sq_dis(5, 0);

  pcl::KdTreeFLANN<PointT>::Ptr kdtree_corner_from_map(new pcl::KdTreeFLANN<PointT>());
  pcl::KdTreeFLANN<PointT>::Ptr kdtree_surf_from_map(new pcl::KdTreeFLANN<PointT>());

  kdtree_corner_from_map->setInputCloud(laser_cloud_corner_from_map_);
  kdtree_surf_from_map->setInputCloud(laser_cloud_surf_from_map_);

  Eigen::Matrix<float, 5, 3> mat_A0;
  Eigen::Matrix<float, 5, 1> mat_B0;
  Eigen::Vector3f mat_X0;
  Eigen::Matrix3f mat_A1;
  //特征值
  Eigen::Matrix<float, 1, 3> mat_D1;
  //特征向量
  Eigen::Matrix3f mat_V1;

  mat_A0.setZero();
  mat_B0.setConstant(-1);
  mat_X0.setZero();

  mat_A1.setZero();
  mat_D1.setZero();
  mat_V1.setZero();

  bool is_degenerate = false;
  matP_.setIdentity();

  size_t laser_cloud_corner_stack_size = laser_cloud_corner_stack_downsampled_->points.size();
  size_t laser_cloud_surf_stack_size = laser_cloud_surf_stack_downsampled_->points.size();

//两种特征的点和特征信息
  PointCloud laser_cloud_ori;
  PointCloud coeff_sel;
//只包含面点的相关信息
  PointCloud laser_cloud_ori_spc;
  PointCloud coeff_sel_spc;
  PointCloud abs_coeff_sel_spc;
  score_point_coeff_.clear();

  // tic_toc_.Tic();

  for (size_t iter_count = 0; iter_count < num_max_iterations_; ++iter_count) {
    laser_cloud_ori.clear();
    coeff_sel.clear();

    laser_cloud_ori_spc.clear();
    coeff_sel_spc.clear();
    abs_coeff_sel_spc.clear();

    for (int i = 0; i < laser_cloud_corner_stack_size; ++i) {
      point_ori = laser_cloud_corner_stack_downsampled_->points[i];
      //转到世界系
      PointAssociateToMap(point_ori, point_sel, transform_tobe_mapped_);
      kdtree_corner_from_map->nearestKSearch(point_sel, 5, point_search_idx, point_search_sq_dis);

      if (point_search_sq_dis[4] < min_match_sq_dis_) {
//        Vector3Intl vc(0, 0, 0);
        Eigen::Vector3f vc(0, 0, 0);

        for (int j = 0; j < 5; j++) {
//          vc += Vector3Intl(laser_cloud_corner_from_map_->points[point_search_idx[j]]);
          const PointT &point_sel_tmp = laser_cloud_corner_from_map_->points[point_search_idx[j]];
          vc.x() += point_sel_tmp.x;
          vc.y() += point_sel_tmp.y;
          vc.z() += point_sel_tmp.z;
        }
        vc /= 5.0;

        Eigen::Matrix3f mat_a;
        mat_a.setZero();

        for (int j = 0; j < 5; j++) {
//          Vector3Intl a = Vector3Intl(laser_cloud_corner_from_map_->points[point_search_idx[j]]) - vc;
          const PointT &point_sel_tmp = laser_cloud_corner_from_map_->points[point_search_idx[j]];
          Eigen::Vector3f a;
          a.x() = point_sel_tmp.x - vc.x();
          a.y() = point_sel_tmp.y - vc.y();
          a.z() = point_sel_tmp.z - vc.z();

          mat_a(0, 0) += a.x() * a.x();
          mat_a(1, 0) += a.x() * a.y();
          mat_a(2, 0) += a.x() * a.z();
          mat_a(1, 1) += a.y() * a.y();
          mat_a(2, 1) += a.y() * a.z();
          mat_a(2, 2) += a.z() * a.z();
        }
        mat_A1 = mat_a / 5.0;

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> esolver(mat_A1);
        mat_D1 = esolver.eigenvalues().real();
        mat_V1 = esolver.eigenvectors().real();

        //线特征
        if (mat_D1(0, 2) > 3 * mat_D1(0, 1)) {

          float x0 = point_sel.x;
          float y0 = point_sel.y;
          float z0 = point_sel.z;
          float x1 = vc.x() + 0.1 * mat_V1(0, 2);
          float y1 = vc.y() + 0.1 * mat_V1(1, 2);
          float z1 = vc.z() + 0.1 * mat_V1(2, 2);
          float x2 = vc.x() - 0.1 * mat_V1(0, 2);
          float y2 = vc.y() - 0.1 * mat_V1(1, 2);
          float z2 = vc.z() - 0.1 * mat_V1(2, 2);

          Eigen::Vector3f X0(x0, y0, z0);
          Eigen::Vector3f X1(x1, y1, z1);
          Eigen::Vector3f X2(x2, y2, z2);

          // NOTE: to make sure on a line with the same as eigen vector

          // NOTE: (P1 - P2) x ((P0 - P1)x(P0 - P2)), point to the point

//          float a012 = sqrt(((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1))
//                                * ((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1))
//                                + ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1))
//                                    * ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1))
//                                + ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1))
//                                    * ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1)));

//          float l12 = sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2));

//          float la = ((y1 - y2) * ((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1))
//              + (z1 - z2) * ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1))) / a012 / l12;
//
//          float lb = -((x1 - x2) * ((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1))
//              - (z1 - z2) * ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1))) / a012 / l12;
//
//          float lc = -((x1 - x2) * ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1))
//              + (y1 - y2) * ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1))) / a012 / l12;

          Eigen::Vector3f a012_vec = (X0 - X1).cross(X0 - X2);
          //单位方向向量
          Eigen::Vector3f normal_to_point = ((X1 - X2).cross(a012_vec)).normalized();

          float a012 = a012_vec.norm();

          float l12 = (X1 - X2).norm();
          //la, lb, lc表示法向量的三个分量，ld2为垂直距离的绝对值
          float la = normal_to_point.x();
          float lb = normal_to_point.y();
          float lc = normal_to_point.z();

          float ld2 = a012 / l12;

          float s = 1 - 0.9f * fabs(ld2);

          coeff.x = s * la;
          coeff.y = s * lb;
          coeff.z = s * lc;
          coeff.intensity = s * ld2;

          abs_coeff.x = la;
          abs_coeff.y = lb;
          abs_coeff.z = lc;
          abs_coeff.intensity = (ld2 - normal_to_point.dot(X0));

          bool is_in_laser_fov = false;
          PointT transform_pos;
          transform_pos.x = transform_tobe_mapped_.pos.x();
          transform_pos.y = transform_tobe_mapped_.pos.y();
          transform_pos.z = transform_tobe_mapped_.pos.z();
          float squared_side1 = CalcSquaredDiff(transform_pos, point_sel);
          float squared_side2 = CalcSquaredDiff(point_on_z_axis_, point_sel);

          float check1 = 100.0f + squared_side1 - squared_side2
              - 10.0f * sqrt(3.0f) * sqrt(squared_side1);

          float check2 = 100.0f + squared_side1 - squared_side2
              + 10.0f * sqrt(3.0f) * sqrt(squared_side1);

          if (check1 < 0 && check2 > 0) { /// within +-60 degree
            is_in_laser_fov = true;
          }

          if (s > 0.1 && is_in_laser_fov) {
            laser_cloud_ori.push_back(point_ori);
            coeff_sel.push_back(coeff);
//            abs_coeff_sel_spc.push_back(abs_coeff);
          }
        }
      }
    }

    for (int i = 0; i < laser_cloud_surf_stack_size; i++) {
      point_ori = laser_cloud_surf_stack_downsampled_->points[i];
      PointAssociateToMap(point_ori, point_sel, transform_tobe_mapped_);

      int num_neighbors = 5;
      kdtree_surf_from_map->nearestKSearch(point_sel, num_neighbors, point_search_idx, point_search_sq_dis);

      if (point_search_sq_dis[num_neighbors - 1] < min_match_sq_dis_) {
        for (int j = 0; j < num_neighbors; j++) {
          mat_A0(j, 0) = laser_cloud_surf_from_map_->points[point_search_idx[j]].x;
          mat_A0(j, 1) = laser_cloud_surf_from_map_->points[point_search_idx[j]].y;
          mat_A0(j, 2) = laser_cloud_surf_from_map_->points[point_search_idx[j]].z;
        }
        mat_X0 = mat_A0.colPivHouseholderQr().solve(mat_B0);

        float pa = mat_X0(0, 0);
        float pb = mat_X0(1, 0);
        float pc = mat_X0(2, 0);
        float pd = 1;

        float ps = sqrt(pa * pa + pb * pb + pc * pc);
        pa /= ps;
        pb /= ps;
        pc /= ps;
        pd /= ps;

        // NOTE: plane as (x y z)*w+1 = 0

        bool planeValid = true;
        for (int j = 0; j < num_neighbors; j++) {
          if (fabs(pa * laser_cloud_surf_from_map_->points[point_search_idx[j]].x +
              pb * laser_cloud_surf_from_map_->points[point_search_idx[j]].y +
              pc * laser_cloud_surf_from_map_->points[point_search_idx[j]].z + pd) > min_plane_dis_) {
            planeValid = false;
            break;
          }
        }

        if (planeValid) {
          float pd2 = pa * point_sel.x + pb * point_sel.y + pc * point_sel.z + pd;

          float s = 1 - 0.9f * fabs(pd2) / sqrt(CalcPointDistance(point_sel));

          if (pd2 > 0) {
            coeff.x = s * pa;
            coeff.y = s * pb;
            coeff.z = s * pc;
            coeff.intensity = s * pd2;

            abs_coeff.x = pa;
            abs_coeff.y = pb;
            abs_coeff.z = pc;
            abs_coeff.intensity = pd;
          } else {
            coeff.x = -s * pa;
            coeff.y = -s * pb;
            coeff.z = -s * pc;
            coeff.intensity = -s * pd2;

            abs_coeff.x = -pa;
            abs_coeff.y = -pb;
            abs_coeff.z = -pc;
            abs_coeff.intensity = -pd;
          }

          bool is_in_laser_fov = false;
          PointT transform_pos;
          transform_pos.x = transform_tobe_mapped_.pos.x();
          transform_pos.y = transform_tobe_mapped_.pos.y();
          transform_pos.z = transform_tobe_mapped_.pos.z();
          float squared_side1 = CalcSquaredDiff(transform_pos, point_sel);
          float squared_side2 = CalcSquaredDiff(point_on_z_axis_, point_sel);

          float check1 = 100.0f + squared_side1 - squared_side2
              - 10.0f * sqrt(3.0f) * sqrt(squared_side1);

          float check2 = 100.0f + squared_side1 - squared_side2
              + 10.0f * sqrt(3.0f) * sqrt(squared_side1);

          if (check1 < 0 && check2 > 0) { /// within +-60 degree
            is_in_laser_fov = true;
          }

          if (s > 0.1 && is_in_laser_fov) {
            laser_cloud_ori.push_back(point_ori);
            coeff_sel.push_back(coeff);

            laser_cloud_ori_spc.push_back(point_ori);
            coeff_sel_spc.push_back(coeff);
            abs_coeff_sel_spc.push_back(abs_coeff);
          }
        }
      }
    }

    size_t laser_cloud_sel_size = laser_cloud_ori.points.size();
    if (laser_cloud_sel_size < 50) {
      continue;
    }

    Eigen::Matrix<float, Eigen::Dynamic, 6> mat_A(laser_cloud_sel_size, 6);
    Eigen::Matrix<float, 6, Eigen::Dynamic> mat_At(6, laser_cloud_sel_size);
    Eigen::Matrix<float, 6, 6> matAtA;
    Eigen::VectorXf mat_B(laser_cloud_sel_size);
    Eigen::VectorXf mat_AtB;
    Eigen::VectorXf mat_X;

    SO3 R_SO3(transform_tobe_mapped_.rot); /// SO3

    for (int i = 0; i < laser_cloud_sel_size; i++) {
      point_ori = laser_cloud_ori.points[i];
      coeff = coeff_sel.points[i];

      Eigen::Vector3f p(point_ori.x, point_ori.y, point_ori.z);
      Eigen::Vector3f w(coeff.x, coeff.y, coeff.z);

//      Eigen::Vector3f J_r = w.transpose() * RotationVectorJacobian(R_SO3, p);
      Eigen::Vector3f J_r = -w.transpose() * (transform_tobe_mapped_.rot * SkewSymmetric(p));
      Eigen::Vector3f J_t = w.transpose();

      float d2 = coeff.intensity;

      mat_A(i, 0) = J_r.x();
      mat_A(i, 1) = J_r.y();
      mat_A(i, 2) = J_r.z();
      mat_A(i, 3) = J_t.x();
      mat_A(i, 4) = J_t.y();
      mat_A(i, 5) = J_t.z();
      mat_B(i, 0) = -d2;
    }

    mat_At = mat_A.transpose();
    matAtA = mat_At * mat_A;
    mat_AtB = mat_At * mat_B;
    mat_X = matAtA.colPivHouseholderQr().solve(mat_AtB);

    if (iter_count == 0) {
      Eigen::Matrix<float, 1, 6> mat_E;
      Eigen::Matrix<float, 6, 6> mat_V;
      Eigen::Matrix<float, 6, 6> mat_V2;

      Eigen::SelfAdjointEigenSolver<Eigen::Matrix<float, 6, 6> > esolver(matAtA);
      mat_E = esolver.eigenvalues().real();
      mat_V = esolver.eigenvectors().real();

      mat_V2 = mat_V;

      is_degenerate = false;
      float eignThre[6] = {100, 100, 100, 100, 100, 100};
      for (int i = 0; i < 6; ++i) {
        if (mat_E(0, i) < eignThre[i]) {
          for (int j = 0; j < 6; ++j) {
            mat_V2(i, j) = 0;
          }
          is_degenerate = true;
          DLOG(WARNING) << "degenerate case";
          DLOG(INFO) << mat_E;
        } else {
          break;
        }
      }
      matP_ = mat_V2 * mat_V.inverse();
    }

    if (is_degenerate) {
      Eigen::Matrix<float, 6, 1> matX2(mat_X);
      mat_X = matP_ * matX2;
    }

    Eigen::Vector3f r_so3 = R_SO3.log();

    r_so3.x() += mat_X(0, 0);
    r_so3.y() += mat_X(1, 0);
    r_so3.z() += mat_X(2, 0);
    transform_tobe_mapped_.pos.x() += mat_X(3, 0);
    transform_tobe_mapped_.pos.y() += mat_X(4, 0);
    transform_tobe_mapped_.pos.z() += mat_X(5, 0);

    if (!isfinite(r_so3.x())) r_so3.x() = 0;
    if (!isfinite(r_so3.y())) r_so3.y() = 0;
    if (!isfinite(r_so3.z())) r_so3.z() = 0;

    SO3 tobe_mapped_SO3 = SO3::exp(r_so3);
//    transform_tobe_mapped_.rot = tobe_mapped_SO3.unit_quaternion().normalized();

    transform_tobe_mapped_.rot =
        transform_tobe_mapped_.rot * DeltaQ(Eigen::Vector3f(mat_X(0, 0), mat_X(1, 0), mat_X(2, 0)));

    if (!isfinite(transform_tobe_mapped_.pos.x())) transform_tobe_mapped_.pos.x() = 0.0;
    if (!isfinite(transform_tobe_mapped_.pos.y())) transform_tobe_mapped_.pos.y() = 0.0;
    if (!isfinite(transform_tobe_mapped_.pos.z())) transform_tobe_mapped_.pos.z() = 0.0;

    float delta_r = RadToDeg(R_SO3.unit_quaternion().angularDistance(transform_tobe_mapped_.rot));
    float delta_t = sqrt(pow(mat_X(3, 0) * 100, 2) +
        pow(mat_X(4, 0) * 100, 2) +
        pow(mat_X(5, 0) * 100, 2));

    if (delta_r < delta_r_abort_ && delta_t < delta_t_abort_) {
      DLOG(INFO) << "iter_count: " << iter_count;
      break;
    }
  }

  TransformUpdate();

  size_t laser_cloud_sel_spc_size = laser_cloud_ori_spc.points.size();
  if (laser_cloud_sel_spc_size >= 50) {
    for (int i = 0; i < laser_cloud_sel_spc_size; i++) {
      pair<float, pair<PointT, PointT>> spc;
      const PointT &p_ori = laser_cloud_ori_spc.points[i];
      const PointT &abs_coeff_in_map = abs_coeff_sel_spc.points[i];
      const PointT &coeff_in_map = coeff_sel_spc.points[i];

      PointT p_in_map;
      PointAssociateToMap(p_ori, p_in_map, transform_tobe_mapped_);

      spc.first = CalcPointDistance(coeff_in_map); // score
      spc.second.first = p_ori; // p_ori

//      spc.second.second = coeff_in_map; // coeff in map
//      spc.second.second.intensity = coeff_in_map.intensity
//          - (coeff_in_map.x * p_in_map.x + coeff_in_map.y * p_in_map.y + coeff_in_map.z * p_in_map.z);

      spc.second.second = abs_coeff_in_map;
//      LOG_IF(INFO, p_in_map.x * abs_coeff_in_map.x + p_in_map.y * abs_coeff_in_map.y + p_in_map.z * abs_coeff_in_map.z
//          + abs_coeff_in_map.intensity < 0)
//      << "distance: " << p_in_map.x * abs_coeff_in_map.x + p_in_map.y * abs_coeff_in_map.y
//          + p_in_map.z * abs_coeff_in_map.z + abs_coeff_in_map.intensity << " < 0";

      score_point_coeff_.insert(spc);


//      DLOG(INFO) << "distance * scale: "
//                << coeff_world.x * p_in_map.x + coeff_world.y * p_in_map.y + coeff_world.z * p_in_map.z
//                    + spc.second.second.intensity;

    }
//    DLOG(INFO) << "^^^^^^^^: " << transform_aft_mapped_;
  }
}



// void PublishResults() {

//   if (!is_ros_setup_) {
//     DLOG(WARNING) << "ros is not set up, and no results will be published";
//     return;
//   }

//   // publish new map cloud according to the input output ratio
//   ++map_frame_count_;
//   if (map_frame_count_ >= num_map_frames_) {
//     map_frame_count_ = 0;

//     // accumulate map cloud
//     laser_cloud_surround_->clear();
//     size_t laser_cloud_surround_size = laser_cloud_surround_idx_.size();
//     for (int i = 0; i < laser_cloud_surround_size; ++i) {
//       size_t index = laser_cloud_surround_idx_[i];
//       *laser_cloud_surround_ += *laser_cloud_corner_array_[index];
//       *laser_cloud_surround_ += *laser_cloud_surf_array_[index];
//     }

//     // down size map cloud
//     laser_cloud_surround_downsampled_->clear();
//     down_size_filter_map_.setInputCloud(laser_cloud_surround_);
//     down_size_filter_map_.filter(*laser_cloud_surround_downsampled_);

//     // publish new map cloud
//     PublishCloudMsg(pub_laser_cloud_surround_,
//                     *laser_cloud_surround_downsampled_,
//                     time_laser_odometry_,
//                     "/camera_init");
//   }


//   // transform full resolution input cloud to map
//   size_t laser_full_cloud_size = full_cloud_->points.size();
//   for (int i = 0; i < laser_full_cloud_size; i++) {
//     PointAssociateToMap(full_cloud_->points[i], full_cloud_->points[i], transform_tobe_mapped_);
//   }

//   // publish transformed full resolution input cloud
//   PublishCloudMsg(pub_full_cloud_, *full_cloud_, time_laser_odometry_, "/camera_init");


//   // publish odometry after mapped transformations
//   geometry_msgs::Quaternion geo_quat;
//   geo_quat.w = transform_aft_mapped_.rot.w();
//   geo_quat.x = transform_aft_mapped_.rot.x();
//   geo_quat.y = transform_aft_mapped_.rot.y();
//   geo_quat.z = transform_aft_mapped_.rot.z();

//   odom_aft_mapped_.header.stamp = time_laser_odometry_;
//   odom_aft_mapped_.pose.pose.orientation.x = geo_quat.x;
//   odom_aft_mapped_.pose.pose.orientation.y = geo_quat.y;
//   odom_aft_mapped_.pose.pose.orientation.z = geo_quat.z;
//   odom_aft_mapped_.pose.pose.orientation.w = geo_quat.w;
//   odom_aft_mapped_.pose.pose.position.x = transform_aft_mapped_.pos.x();
//   odom_aft_mapped_.pose.pose.position.y = transform_aft_mapped_.pos.y();
//   odom_aft_mapped_.pose.pose.position.z = transform_aft_mapped_.pos.z();

// //  odom_aft_mapped_.twist.twist.angular.x = transform_bef_mapped_.rot.x();
// //  odom_aft_mapped_.twist.twist.angular.y = transform_bef_mapped_.rot.y();
// //  odom_aft_mapped_.twist.twist.angular.z = transform_bef_mapped_.rot.z();
// //  odom_aft_mapped_.twist.twist.linear.x = transform_bef_mapped_.pos.x();
// //  odom_aft_mapped_.twist.twist.linear.y = transform_bef_mapped_.pos.y();
// //  odom_aft_mapped_.twist.twist.linear.z = transform_bef_mapped_.pos.z();
//   pub_odom_aft_mapped_.publish(odom_aft_mapped_);

//   aft_mapped_trans_.stamp_ = time_laser_odometry_;
//   aft_mapped_trans_.setRotation(tf::Quaternion(geo_quat.x, geo_quat.y, geo_quat.z, geo_quat.w));
//   aft_mapped_trans_.setOrigin(tf::Vector3(transform_aft_mapped_.pos.x(),
//                                           transform_aft_mapped_.pos.y(),
//                                           transform_aft_mapped_.pos.z()));
//   tf_broadcaster_.sendTransform(aft_mapped_trans_);
// }

void Process() {
  if (!HasNewData()) {
    // waiting for new data to arrive...
    // DLOG(INFO) << "no data received or dropped";
    return;
  }

  Reset();

  ++frame_count_;
  if (frame_count_ < num_stack_frames_) {
    return;
  }
  frame_count_ = 0;

  PointT point_sel;

  // relate incoming data to map
  // WARNING
  if (!imu_inited_) {
    //用lidar odom计算位姿增量 加到每次优化后的位姿上作为最新帧位姿优化初始值
    TransformAssociateToMap();
  }

  // NOTE: the stack points are the last corner or surf poitns
  //这里laser_cloud_corner_last_已经填满了
  size_t laser_cloud_corner_last_size = laser_cloud_corner_last_->points.size();
  for (int i = 0; i < laser_cloud_corner_last_size; i++) {
    //点云变换到世界系下
    PointAssociateToMap(laser_cloud_corner_last_->points[i], point_sel, transform_tobe_mapped_);
    laser_cloud_corner_stack_->push_back(point_sel);
  }

  size_t laser_cloud_surf_last_size = laser_cloud_surf_last_->points.size();
  for (int i = 0; i < laser_cloud_surf_last_size; i++) {
    PointAssociateToMap(laser_cloud_surf_last_->points[i], point_sel, transform_tobe_mapped_);
    laser_cloud_surf_stack_->push_back(point_sel);
  }

  // NOTE: above codes update the transform with incremental value and update them to the map coordinate
  point_on_z_axis_.x = 0.0;
  point_on_z_axis_.y = 0.0;
  point_on_z_axis_.z = 10.0;
  PointAssociateToMap(point_on_z_axis_, point_on_z_axis_, transform_tobe_mapped_);

  // NOTE: in which cube
  int center_cube_i = int((transform_tobe_mapped_.pos.x() + 25.0) / 50.0) + laser_cloud_cen_length_;
  int center_cube_j = int((transform_tobe_mapped_.pos.y() + 25.0) / 50.0) + laser_cloud_cen_width_;
  int center_cube_k = int((transform_tobe_mapped_.pos.z() + 25.0) / 50.0) + laser_cloud_cen_height_;

  // NOTE: negative index
  if (transform_tobe_mapped_.pos.x() + 25.0 < 0) --center_cube_i;
  if (transform_tobe_mapped_.pos.y() + 25.0 < 0) --center_cube_j;
  if (transform_tobe_mapped_.pos.z() + 25.0 < 0) --center_cube_k;

 DLOG(INFO) << "center_before: " << center_cube_i << " " << center_cube_j << " " << center_cube_k;
  {
    while (center_cube_i < 3) {
      for (int j = 0; j < laser_cloud_width_; ++j) {
        for (int k = 0; k < laser_cloud_height_; ++k) {
          for (int i = laser_cloud_length_ - 1; i >= 1; --i) {
            const size_t index_a = ToIndex(i, j, k);
            const size_t index_b = ToIndex(i - 1, j, k);
            std::swap(laser_cloud_corner_array_[index_a], laser_cloud_corner_array_[index_b]);
            std::swap(laser_cloud_surf_array_[index_a], laser_cloud_surf_array_[index_b]);
          }
          laser_cloud_corner_array_[ToIndex(0, j, k)]->clear();
          laser_cloud_surf_array_[ToIndex(0, j, k)]->clear();
        }
      }
      ++center_cube_i;
      ++laser_cloud_cen_length_;
    }

    while (center_cube_i >= laser_cloud_length_ - 3) {
      for (int j = 0; j < laser_cloud_width_; ++j) {
        for (int k = 0; k < laser_cloud_height_; ++k) {
          for (int i = 0; i < laser_cloud_length_ - 1; ++i) {
            const size_t index_a = ToIndex(i, j, k);
            const size_t index_b = ToIndex(i + 1, j, k);
            std::swap(laser_cloud_corner_array_[index_a], laser_cloud_corner_array_[index_b]);
            std::swap(laser_cloud_surf_array_[index_a], laser_cloud_surf_array_[index_b]);
          }
          laser_cloud_corner_array_[ToIndex(laser_cloud_length_ - 1, j, k)]->clear();
          laser_cloud_surf_array_[ToIndex(laser_cloud_length_ - 1, j, k)]->clear();
        }
      }
      --center_cube_i;
      --laser_cloud_cen_length_;
    }

    while (center_cube_j < 3) {
      for (int i = 0; i < laser_cloud_length_; ++i) {
        for (int k = 0; k < laser_cloud_height_; ++k) {
          for (int j = laser_cloud_width_ - 1; j >= 1; --j) {
            const size_t index_a = ToIndex(i, j, k);
            const size_t index_b = ToIndex(i, j - 1, k);
            std::swap(laser_cloud_corner_array_[index_a], laser_cloud_corner_array_[index_b]);
            std::swap(laser_cloud_surf_array_[index_a], laser_cloud_surf_array_[index_b]);
          }
          laser_cloud_corner_array_[ToIndex(i, 0, k)]->clear();
          laser_cloud_surf_array_[ToIndex(i, 0, k)]->clear();
        }
      }
      ++center_cube_j;
      ++laser_cloud_cen_width_;
    }

    while (center_cube_j >= laser_cloud_width_ - 3) {
      for (int i = 0; i < laser_cloud_length_; ++i) {
        for (int k = 0; k < laser_cloud_height_; ++k) {
          for (int j = 0; j < laser_cloud_width_ - 1; ++j) {
            const size_t index_a = ToIndex(i, j, k);
            const size_t index_b = ToIndex(i, j + 1, k);
            std::swap(laser_cloud_corner_array_[index_a], laser_cloud_corner_array_[index_b]);
            std::swap(laser_cloud_surf_array_[index_a], laser_cloud_surf_array_[index_b]);
          }
          laser_cloud_corner_array_[ToIndex(i, laser_cloud_width_ - 1, k)]->clear();
          laser_cloud_surf_array_[ToIndex(i, laser_cloud_width_ - 1, k)]->clear();
        }
      }
      --center_cube_j;
      --laser_cloud_cen_width_;
    }

    while (center_cube_k < 3) {
      for (int i = 0; i < laser_cloud_length_; ++i) {
        for (int j = 0; j < laser_cloud_width_; ++j) {
          for (int k = laser_cloud_height_ - 1; k >= 1; --k) {
            const size_t index_a = ToIndex(i, j, k);
            const size_t index_b = ToIndex(i, j, k - 1);
            std::swap(laser_cloud_corner_array_[index_a], laser_cloud_corner_array_[index_b]);
            std::swap(laser_cloud_surf_array_[index_a], laser_cloud_surf_array_[index_b]);
          }
          laser_cloud_corner_array_[ToIndex(i, j, 0)]->clear();
          laser_cloud_surf_array_[ToIndex(i, j, 0)]->clear();
        }
      }
      ++center_cube_k;
      ++laser_cloud_cen_height_;
    }

    while (center_cube_k >= laser_cloud_height_ - 3) {
      for (int i = 0; i < laser_cloud_length_; ++i) {
        for (int j = 0; j < laser_cloud_width_; ++j) {
          for (int k = 0; k < laser_cloud_height_ - 1; ++k) {
            const size_t index_a = ToIndex(i, j, k);
            const size_t index_b = ToIndex(i, j, k + 1);
            std::swap(laser_cloud_corner_array_[index_a], laser_cloud_corner_array_[index_b]);
            std::swap(laser_cloud_surf_array_[index_a], laser_cloud_surf_array_[index_b]);
          }
          laser_cloud_corner_array_[ToIndex(i, j, laser_cloud_height_ - 1)]->clear();
          laser_cloud_surf_array_[ToIndex(i, j, laser_cloud_height_ - 1)]->clear();
        }
      }
      --center_cube_k;
      --laser_cloud_cen_height_;
    }
  }

  // NOTE: above slide cubes

//存lidar视野内的点云
  laser_cloud_valid_idx_.clear();
//存局部地图全部点云
  laser_cloud_surround_idx_.clear();

//  DLOG(INFO) << "center_after: " << center_cube_i << " " << center_cube_j << " " << center_cube_k;
//  DLOG(INFO) << "laser_cloud_cen: " << laser_cloud_cen_length_ << " " << laser_cloud_cen_width_ << " "
//            << laser_cloud_cen_height_;

  for (int i = center_cube_i - 2; i <= center_cube_i + 2; ++i) {
    for (int j = center_cube_j - 2; j <= center_cube_j + 2; ++j) {
      for (int k = center_cube_k - 2; k <= center_cube_k + 2; ++k) {
        if (i >= 0 && i < laser_cloud_length_ &&
            j >= 0 && j < laser_cloud_width_ &&
            k >= 0 && k < laser_cloud_height_) { /// Should always in this condition

          float center_x = 50.0f * (i - laser_cloud_cen_length_);
          float center_y = 50.0f * (j - laser_cloud_cen_width_);
          float center_z = 50.0f * (k - laser_cloud_cen_height_); // NOTE: center of the cube

          PointT transform_pos;
          transform_pos.x = transform_tobe_mapped_.pos.x();
          transform_pos.y = transform_tobe_mapped_.pos.y();
          transform_pos.z = transform_tobe_mapped_.pos.z();

          bool is_in_laser_fov = false;
          for (int ii = -1; ii <= 1; ii += 2) {
            for (int jj = -1; jj <= 1; jj += 2) {
              for (int kk = -1; kk <= 1; kk += 2) {
                PointT corner;
                corner.x = center_x + 25.0f * ii;
                corner.y = center_y + 25.0f * jj;
                corner.z = center_z + 25.0f * kk;

                float squared_side1 = CalcSquaredDiff(transform_pos, corner);
                float squared_side2 = CalcSquaredDiff(point_on_z_axis_, corner);

                float check1 = 100.0f + squared_side1 - squared_side2
                    - 10.0f * sqrt(3.0f) * sqrt(squared_side1);

                float check2 = 100.0f + squared_side1 - squared_side2
                    + 10.0f * sqrt(3.0f) * sqrt(squared_side1);

                if (check1 < 0 && check2 > 0) { /// within +-60 degree
                  is_in_laser_fov = true;
                }
              }
            }
          }

          size_t cube_idx = ToIndex(i, j, k);

//          DLOG(INFO) << "ToIndex, i, j, k " << cube_idx << " " << i << " " << j << " " << k;
//          int tmpi, tmpj, tmpk;
//          FromIndex(cube_idx, tmpi, tmpj, tmpk);
//          DLOG(INFO) << "FromIndex, i, j, k " << cube_idx << " " << tmpi << " " << tmpj << " " << tmpk;

          if (is_in_laser_fov) {
            laser_cloud_valid_idx_.push_back(cube_idx);
          }
          laser_cloud_surround_idx_.push_back(cube_idx);
        }
      }
    }
  }

  // prepare valid map corner and surface cloud for pose optimization
  laser_cloud_corner_from_map_->clear();
  laser_cloud_surf_from_map_->clear();
  size_t laser_cloud_valid_size = laser_cloud_valid_idx_.size();
  for (int i = 0; i < laser_cloud_valid_size; ++i) {
    *laser_cloud_corner_from_map_ += *laser_cloud_corner_array_[laser_cloud_valid_idx_[i]];
    *laser_cloud_surf_from_map_ += *laser_cloud_surf_array_[laser_cloud_valid_idx_[i]];
  }

  // prepare feature stack clouds for pose optimization
  //点云由世界系坐标变换到当前lidar系
  size_t laser_cloud_corner_stack_size2 = laser_cloud_corner_stack_->points.size();
  for (int i = 0; i < laser_cloud_corner_stack_size2; ++i) {
    PointAssociateTobeMapped(laser_cloud_corner_stack_->points[i],
                             laser_cloud_corner_stack_->points[i],
                             transform_tobe_mapped_);
  }

  size_t laserCloudSurfStackNum2 = laser_cloud_surf_stack_->points.size();
  for (int i = 0; i < laserCloudSurfStackNum2; ++i) {
    PointAssociateTobeMapped(laser_cloud_surf_stack_->points[i],
                             laser_cloud_surf_stack_->points[i],
                             transform_tobe_mapped_);
  }

  // down sample feature stack clouds
  laser_cloud_corner_stack_downsampled_->clear();
  down_size_filter_corner_.setInputCloud(laser_cloud_corner_stack_);
  down_size_filter_corner_.filter(*laser_cloud_corner_stack_downsampled_);
  size_t laser_cloud_corner_stack_ds_size = laser_cloud_corner_stack_downsampled_->points.size();

  laser_cloud_surf_stack_downsampled_->clear();
  down_size_filter_surf_.setInputCloud(laser_cloud_surf_stack_);
  down_size_filter_surf_.filter(*laser_cloud_surf_stack_downsampled_);
  size_t laser_cloud_surf_stack_ds_size = laser_cloud_surf_stack_downsampled_->points.size();

  laser_cloud_corner_stack_->clear();
  laser_cloud_surf_stack_->clear();

  // NOTE: keeps the downsampled points

  // NOTE: run pose optimization
  //scan to map优化出当前帧位姿
  OptimizeTransformTobeMapped();

  if (!imu_inited_) {
    // store down sized corner stack points in corresponding cube clouds

    CubeCenter cube_center;
    cube_center.laser_cloud_cen_length = laser_cloud_cen_length_;
    cube_center.laser_cloud_cen_width = laser_cloud_cen_width_;
    cube_center.laser_cloud_cen_height = laser_cloud_cen_height_;

    UpdateMapDatabase(laser_cloud_corner_stack_downsampled_,
                      laser_cloud_surf_stack_downsampled_,
                      laser_cloud_valid_idx_,
                      transform_tobe_mapped_,
                      cube_center);
    // publish result、
    //todo 实验再加
    // PublishResults();
  }
  // DLOG(INFO) << "mapping: " << tic_toc_.toc() << " ms";
}

// lio-mapping 版本
void processLidarInfo(const LidarInfo &lidar_info, const std_msgs::Header &header)
{
    //1. process lidar_info
  	LidarInfoHandler(lidar_info);

  	if (stage_flag_ == INITED) {
    	Transform trans_prev(Eigen::Quaterniond(Rs_[estimator_config_.window_size - 1]).cast<float>(),
        	                 Ps_[estimator_config_.window_size - 1].cast<float>());
    	Transform trans_curr(Eigen::Quaterniond(Rs_.last()).cast<float>(),
        	                 Ps_.last().cast<float>());

    	Transform d_trans = trans_prev.inverse() * trans_curr;
		//transform_sum_ LidarInfoHandler取出来的
    	Transform transform_incre(transform_bef_mapped_.inverse() * transform_sum_.transform());

    	if (estimator_config_.imu_factor) {
      	   // WARNING: or using direct date?
      		transform_tobe_mapped_bef_ = transform_tobe_mapped_ * transform_lb_ * d_trans * transform_lb_.inverse();
      		transform_tobe_mapped_ = transform_tobe_mapped_bef_;
    	} else {
      		TransformAssociateToMap();
      		DLOG(INFO) << ">>>>> transform original tobe <<<<<: " << transform_tobe_mapped_;
    	}
  	}

  	if (stage_flag_ != INITED || !estimator_config_.imu_factor) {
    	/// 2. process decoded data
    	Process();
  	} else {
  	}

  	DLOG(INFO) << "laser_cloud_surf_last_[" << header.stamp.toSec() << "]: "
    	        << laser_cloud_surf_last_->size();
  	DLOG(INFO) << "laser_cloud_corner_last_[" << header.stamp.toSec() << "]: "
    	        << laser_cloud_corner_last_->size();
	  DLOG(INFO) << endl << "transform_aft_mapped_[" << header.stamp.toSec() << "]: " << transform_aft_mapped_;
  	DLOG(INFO) << "laser_cloud_surf_stack_downsampled_[" << header.stamp.toSec() << "]: "
    	        << laser_cloud_surf_stack_downsampled_->size();
  	DLOG(INFO) << "laser_cloud_corner_stack_downsampled_[" << header.stamp.toSec() << "]: "
    	        << laser_cloud_corner_stack_downsampled_->size();

    //获取scan to map优化后的当前帧位姿
  	Transform transform_to_init_ = transform_aft_mapped_;
  	ProcessLaserOdom(transform_to_init_, header);

// NOTE: will be updated in PointMapping's OptimizeTransformTobeMapped
//  if (stage_flag_ == INITED && !estimator_config_.imu_factor) {
//    TransformUpdate();
//    DLOG(INFO) << ">>>>> transform sum <<<<<: " << transform_sum_;
//  }
}

void SurfHandler(const sensor_msgs::PointCloud2ConstPtr &_laserSurf)
{
    mBuf.lock();
    surfBuf.push(_laserSurf);
    mBuf.unlock();
    con.notify_one();
}

void EdgeHandler(const sensor_msgs::PointCloud2ConstPtr &_laserEdge)
{
    mBuf.lock();
    edgeBuf.push(_laserEdge);
    mBuf.unlock();
    con.notify_one();
}

//void pubPath_lio( void )
//{
//    std::cout << "mark pubPath_lio" << std::endl;
//    // pub odom and path
//    nav_msgs::Odometry odomAftPGO;
//    nav_msgs::Path pathAftPGO;
//    pathAftPGO.header.frame_id = "camera_init";
//    //todo 如果很卡再把两个锁分开用
////    mBuf.lock();
//    //[0, 9]
//    for (int node_idx=0; node_idx < windowSize; node_idx++) // -1 is just delayed visualization (because sometimes mutexed while adding(push_back) a new one)
//    {
//        // 经过isam回环修正后的位姿
//        const Frame& pose_set = frameWindow[node_idx];
//        nav_msgs::Odometry odomAftPGOthis;
//        odomAftPGOthis.header.frame_id = "camera_init";
//        odomAftPGOthis.child_frame_id = "/aft_pgo";
//        // 修正前和修正后的位姿的时间是相同的
//        odomAftPGOthis.header.stamp = ros::Time().fromSec(frameTimes[node_idx]);
//		// 获取位置并赋值
//		odomAftPGOthis.pose.pose.position.x = pose_set.pose.translation().x();
//		odomAftPGOthis.pose.pose.position.y = pose_set.pose.translation().y();
//		odomAftPGOthis.pose.pose.position.z = pose_set.pose.translation().z();
//
//		// 获取姿态并赋值
//		gtsam::Quaternion quaternion = pose_set.pose.rotation().toQuaternion();
//		odomAftPGOthis.pose.pose.orientation.x = quaternion.x();
//		odomAftPGOthis.pose.pose.orientation.y = quaternion.y();
//		odomAftPGOthis.pose.pose.orientation.z = quaternion.z();
//		odomAftPGOthis.pose.pose.orientation.w = quaternion.w();
//
//        odomAftPGO = odomAftPGOthis;
//
//        geometry_msgs::PoseStamped poseStampAftPGO;
//        poseStampAftPGO.header = odomAftPGOthis.header;
//        poseStampAftPGO.pose = odomAftPGOthis.pose.pose;
//
//        pathAftPGO.header.stamp = odomAftPGOthis.header.stamp;
//        pathAftPGO.header.frame_id = "camera_init";
//        pathAftPGO.poses.push_back(poseStampAftPGO);
//    }
////    mBuf.unlock();
//
//    //保存轨迹，path_save是文件目录,txt文件提前建好 tum格式 time x y z
////    std::ofstream pose1("/media/ctx/0BE20E8D0BE20E8D/dataset/kitti_dataset/result/loamgba/result_pose_09_30_0027_07_loop.txt", std::ios::app);
////    pose1.setf(std::ios::scientific, std::ios::floatfield);
////    //kitti数据集转换tum格式的数据是18位
////    pose1.precision(9);
////    //第一个激光帧时间 static变量 只赋值一次
////    static double timeStart = odomAftPGO.header.stamp.toSec();
////    auto T1 =ros::Time().fromSec(timeStart) ;
////    // tf::Quaternion quat;
////    // tf::createQuaternionMsgFromRollPitchYaw(double r, double p, double y);//返回四元数
////    pose1<< odomAftPGO.header.stamp -T1<< " "
////        << -odomAftPGO.pose.pose.position.x << " "
////        << -odomAftPGO.pose.pose.position.z << " "
////        << -odomAftPGO.pose.pose.position.y<< " "
////        << odomAftPGO.pose.pose.orientation.x << " "
////        << odomAftPGO.pose.pose.orientation.y << " "
////        << odomAftPGO.pose.pose.orientation.z << " "
////        << odomAftPGO.pose.pose.orientation.w << std::endl;
////    pose1.close();
//
//
//    pubOdomAftPGO.publish(odomAftPGO); // 滑窗内最后一帧位姿
//    pubPathAftPGO.publish(pathAftPGO); // 轨迹
//
//    //cout << "pathAftPGO.poses = " << pathAftPGO.poses.size() << endl;
//    globalPath = pathAftPGO;
//    if(follow)
//    {
//        static tf::TransformBroadcaster br;
//        tf::Transform transform;
//        tf::Quaternion q;
//        transform.setOrigin(tf::Vector3(odomAftPGO.pose.pose.position.x, odomAftPGO.pose.pose.position.y, odomAftPGO.pose.pose.position.z));
//        q.setW(odomAftPGO.pose.pose.orientation.w);
//        q.setX(odomAftPGO.pose.pose.orientation.x);
//        q.setY(odomAftPGO.pose.pose.orientation.y);
//        q.setZ(odomAftPGO.pose.pose.orientation.z);
//        transform.setRotation(q);
//        br.sendTransform(tf::StampedTransform(transform, odomAftPGO.header.stamp, "camera_init", "/aft_pgo"));
//
//    }
//
//} // pubPath_lio


void imuHandler(const sensor_msgs::ImuConstPtr &imu_msg)
{
    if (imu_msg->header.stamp.toSec() <= last_imu_t)
    {
        ROS_WARN("imu message in disorder!");
        return;
    }
    last_imu_t = imu_msg->header.stamp.toSec();

    mBuf.lock();
    imu_buf.push(imu_msg);
    mBuf.unlock();
    con.notify_one();

    {
//        std::lock_guard<std::mutex> lg(m_state);
        //vins在这里做的imu积分 lio-mapping没在这里做
//        predict(imu_msg);

//        std_msgs::Header header = imu_msg->header;
//        header.frame_id = "world";
//        if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
//            pubLatestOdometry(tmp_P, tmp_Q, tmp_V, header);
    }
}

//对齐数据 改成自定义的数据结构 包含点云、激光里程计、预留事件相机帧
std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, LidarInfo>>
getMeasurements()
{
    std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, LidarInfo>> measurements;

    while (true)
    {
        if (imu_buf.empty() || odometryBuf.empty())
            return measurements;

        if (!(imu_buf.back()->header.stamp.toSec() > odometryBuf.front()->header.stamp.toSec()))
        {
            //ROS_WARN("wait for imu, only should happen at the beginning");
            return measurements;
        }

        if (!(imu_buf.front()->header.stamp.toSec() < odometryBuf.front()->header.stamp.toSec()))
        {
            ROS_WARN("throw odom, only should happen at the beginning");
            if(odometryBuf.size() != 0)
            	odometryBuf.pop();
            if(surfBuf.size() != 0)
            	surfBuf.pop();
            if(edgeBuf.size() != 0)
            	edgeBuf.pop();
            continue;
        }

        nav_msgs::Odometry::ConstPtr odom_msg;
        if(odometryBuf.size() != 0) {
        	odom_msg = odometryBuf.front();
            odometryBuf.pop();
        }
        sensor_msgs::PointCloud2ConstPtr surf_msg;
        if(surfBuf.size() != 0) {
        	surf_msg = surfBuf.front();
            surfBuf.pop();
        }
        sensor_msgs::PointCloud2ConstPtr edge_msg;
        if(edgeBuf.size() != 0) {
        	edge_msg = edgeBuf.front();
            edgeBuf.pop();
        }
		LidarInfo lidar_info;
        lidar_info.laser_odometry_ = odom_msg;
        lidar_info.surf_cloud_ = surf_msg;
        lidar_info.edge_cloud_ = edge_msg;

        std::vector<sensor_msgs::ImuConstPtr> IMUs;
        while (imu_buf.front()->header.stamp.toSec() < odom_msg->header.stamp.toSec())
        {
            IMUs.emplace_back(imu_buf.front());
            imu_buf.pop();
        }
        IMUs.emplace_back(imu_buf.front());
        if (IMUs.empty())
            ROS_WARN("no imu between two image");

        measurements.emplace_back(IMUs, lidar_info);
    }
    return measurements;
}

void process_lio()
{
	while (true) {
		std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, LidarInfo>> measurements;
        std::unique_lock<std::mutex> lk(mBuf);
        con.wait(lk, [&]
                 {
            return (measurements = getMeasurements()).size() != 0;
                 });
        lk.unlock();
        //vins中和restart有关 这里用不着
//        m_estimator.lock();
		for (auto &measurement : measurements)
        {
            auto lidar_info = measurement.second;
            double dx = 0, dy = 0, dz = 0, rx = 0, ry = 0, rz = 0;
            for (auto &imu_msg : measurement.first)
            {
                double imu_time = imu_msg->header.stamp.toSec();
                double lidar_info_time = lidar_info.laser_odometry_->header.stamp.toSec();
                if (imu_time <= lidar_info_time)
                {
                    if (curr_time_ < 0)
                        curr_time_ = imu_time;
                    double dt = imu_time - curr_time_;
                    ROS_ASSERT(dt >= 0);
                    curr_time_ = imu_time;
                    dx = imu_msg->linear_acceleration.x;
                    dy = imu_msg->linear_acceleration.y;
                    dz = imu_msg->linear_acceleration.z;
                    rx = imu_msg->angular_velocity.x;
                    ry = imu_msg->angular_velocity.y;
                    rz = imu_msg->angular_velocity.z;
                    processIMU(dt, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz), imu_msg->header);
                    //printf("imu: dt:%f a: %f %f %f w: %f %f %f\n",dt, dx, dy, dz, rx, ry, rz);
                }
                else
                {
                    double dt_1 = lidar_info_time - curr_time_;
                    double dt_2 = imu_time - lidar_info_time;
                    curr_time_ = lidar_info_time;
                    ROS_ASSERT(dt_1 >= 0);
                    ROS_ASSERT(dt_2 >= 0);
                    ROS_ASSERT(dt_1 + dt_2 > 0);
                    double w1 = dt_2 / (dt_1 + dt_2);
                    double w2 = dt_1 / (dt_1 + dt_2);
                    dx = w1 * dx + w2 * imu_msg->linear_acceleration.x;
                    dy = w1 * dy + w2 * imu_msg->linear_acceleration.y;
                    dz = w1 * dz + w2 * imu_msg->linear_acceleration.z;
                    rx = w1 * rx + w2 * imu_msg->angular_velocity.x;
                    ry = w1 * ry + w2 * imu_msg->angular_velocity.y;
                    rz = w1 * rz + w2 * imu_msg->angular_velocity.z;
                    processIMU(dt_1, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz), imu_msg->header);
                    //printf("dimu: dt:%f a: %f %f %f w: %f %f %f\n",dt_1, dx, dy, dz, rx, ry, rz);
                }
            }

            TicToc t_s;
            processLidarInfo(lidar_info, lidar_info.laser_odometry_->header);

            double whole_t = t_s.toc();
//            printStatistics(estimator, whole_t);
//            std_msgs::Header header = img_msg->header;
//            header.frame_id = "world";
//
//            pubOdometry(estimator, header);
//            pubKeyPoses(estimator, header);
//            pubCameraPose(estimator, header);
//            pubPointCloud(estimator, header);
//            pubTF(estimator, header);
//            pubKeyframe(estimator);
//            if (relo_msg != NULL)
//                pubRelocalization(estimator);
            //ROS_ERROR("end: %f, at %f", img_msg->header.stamp.toSec(), ros::Time::now().toSec());
        }
//        m_estimator.unlock();
        //todo 这里lio-mapping额外加了一个thread_mutex锁 好像没有用
        mBuf.lock();
        m_state.lock();
//        if (solver_flag == NON_LINEAR)
//            update();
        m_state.unlock();
        mBuf.unlock();
	}
} // process_lio

void ClearState() {
  // TODO: CirclarBuffer should have clear method
  for (size_t i = 0; i < estimator_config_.window_size + 1;
       ++i) {
    Rs_[i].setIdentity();
    Ps_[i].setZero();
    Vs_[i].setZero();
    Bas_[i].setZero();
    Bgs_[i].setZero();
    dt_buf_[i].clear();
    linear_acceleration_buf_[i].clear();
    angular_velocity_buf_[i].clear();

    surf_stack_[i].reset();
    corner_stack_[i].reset();
    full_stack_[i].reset();
    size_surf_stack_[i] = 0;
    size_corner_stack_[i] = 0;
    init_local_map_ = false;

    if (pre_integrations_[i] != nullptr) {
      pre_integrations_[i].reset();
    }
  }

  for (size_t i = 0; i < estimator_config_.opt_window_size + 1;
       ++i) {
//    opt_point_coeff_mask_[i] = false;
//    opt_cube_centers_[i];
//    opt_valid_idx_[i];
    opt_point_coeff_map_[i].clear();
    opt_corner_stack_[i].reset();
    opt_surf_stack_[i].reset();
  }

  stage_flag_ = NOT_INITED;
  first_imu_ = false;
  cir_buf_count_ = 0;

  //acc_last_初值给多少 这里无所谓 processlaserodom会reset
  tmp_pre_integration_.reset();
  tmp_pre_integration_ = std::make_shared<leio::IntegrationBase>(leio::IntegrationBase(acc_last_,
                                                                           gyr_last_,
                                                                           Bas_[cir_buf_count_],
                                                                           Bgs_[cir_buf_count_],
                                                                           estimator_config_.pim_config));

  // TODO: make shared?
  last_marginalization_info = nullptr;

  R_WI_.setIdentity();
  Q_WI_ = R_WI_;

  // WARNING: g_norm should be set before clear
  g_norm_ = tmp_pre_integration_->config_.g_norm;

  convergence_flag_ = false;
}

int main(int argc, char **argv)
{
	ros::init(argc, argv, "laserPGO");
//    ROS_FATAL("11111111111111111111111111111111111111111111111111111111111");
	ros::NodeHandle nh;

    //todo 全局变量初始化
    down_size_filter_corner_.setLeafSize(estimator_config_.corner_filter_size, estimator_config_.corner_filter_size, estimator_config_.corner_filter_size);
  	down_size_filter_surf_.setLeafSize(estimator_config_.surf_filter_size, estimator_config_.surf_filter_size, estimator_config_.surf_filter_size);
  	down_size_filter_map_.setLeafSize(estimator_config_.map_filter_size, estimator_config_.map_filter_size, estimator_config_.map_filter_size);
    //imu参数
    estimator_config_.pim_config.acc_n = imuAccNoise;
    estimator_config_.pim_config.gyr_n = imuGyrNoise;
    estimator_config_.pim_config.acc_w = imuAccBiasN;
    estimator_config_.pim_config.gyr_w = imuGyrBiasN;
    estimator_config_.pim_config.g_norm = imuGravity;
    laser_cloud_corner_array_.resize(laser_cloud_num_);
    laser_cloud_surf_array_.resize(laser_cloud_num_);
    laser_cloud_corner_downsampled_array_.resize(laser_cloud_num_);
    laser_cloud_surf_downsampled_array_.resize(laser_cloud_num_);
    for (size_t i = 0; i < laser_cloud_num_; i++) {
      laser_cloud_corner_array_[i].reset(new PointCloud());
      laser_cloud_surf_array_[i].reset(new PointCloud());
      laser_cloud_corner_downsampled_array_[i].reset(new PointCloud());
      laser_cloud_surf_downsampled_array_[i].reset(new PointCloud());
    }

    //lio-mapping中构建estimator的部分
    para_pose_ = new double *[estimator_config_.opt_window_size + 1];
    para_speed_bias_ = new double *[estimator_config_.opt_window_size + 1];
    for (int i = 0; i < estimator_config_.opt_window_size + 1;
         ++i) {
      para_pose_[i] = new double[leio::SIZE_POSE];
      para_speed_bias_[i] = new double[leio::SIZE_SPEED_BIAS];
    }
    ClearState();

    // --------------------------------- 订阅后端数据 ---------------------------------
//	 ros::Subscriber subCenters = nh.subscribe<sensor_msgs::PointCloud2>("/Center_BA", 100, centerHandler);
	ros::Subscriber subSurf = nh.subscribe<sensor_msgs::PointCloud2>("/ground_BA", 100, SurfHandler);
    ros::Subscriber subEdge = nh.subscribe<sensor_msgs::PointCloud2>("/Edge_BA", 100, EdgeHandler);

    ros::Subscriber subLaserOdometry = nh.subscribe<nav_msgs::Odometry>("/BALM_mapped_to_init", 100, laserOdometryHandler);
//	ros::Subscriber subGPS = nh.subscribe<sensor_msgs::NavSatFix>("/gps/fix", 100, gpsHandler);

    //订阅IMU数据
    ros::Subscriber subImu = nh.subscribe<sensor_msgs::Imu>("imu_raw", 2000, imuHandler, ros::TransportHints().tcpNoDelay());

    // ------------------------------------------------------------------
//	pubOdomAftPGO = nh.advertise<nav_msgs::Odometry>("/aft_pgo_odom", 100);
//	pubPathAftPGO = nh.advertise<nav_msgs::Path>("/aft_pgo_path", 100);

    //执行后端优化的线程
    std::thread measurement_process{process_lio};
 	ros::spin();
	measurement_process.join();
	return 0;
}
