#include <fstream>
#include <math.h>
#include <vector>
#include <mutex>
#include <queue>
#include <thread>
#include <iostream>
#include <string>
#include <optional>

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
#include "utility/integration_base.h"

#include <pcl/io/pcd_io.h>
#include <image_transport/image_transport.h>
#include "utility/Scancontext.h"
#include "lidarFactor.hpp"

#include "CSF/CSF.h"
#include "feature_manager/FeatureManager.hpp"
#include "utils/Twist.hpp"
#include "utils/CircularBuffer.hpp"
#include "utils/geometry_utils.hpp"
#include "utils/math_utils.hpp"
#include "factor/PoseLocalParameterization.hpp"
#include "factor/MarginalizationFactor.hpp"
#include "factor/PivotPointPlaneFactor.hpp"

//todo 全局变量、类型定义 能直接初始化在定义时初始化 不能的放在main函数中初始化
//消息buffer + 消息队列锁
std::queue<sensor_msgs::ImuConstPtr> imu_buf;
double last_imu_t = 0;   //imuHandler使用
std::queue<nav_msgs::Odometry::ConstPtr> odometryBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> surfBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> edgeBuf;
std::mutex mBuf;
std::mutex m_state; //没用上
std::condition_variable con;
//LIO estimator

//自定义的雷达数据结构 包含点云 里程计信息 用于getmeasurements()对齐
class LidarInfo
{
	public:
    	LidarInfo(){};
        sensor_msgs::PointCloud2 surf_cloud_;
    	sensor_msgs::PointCloud2 edge_cloud_;
		nav_msgs::Odometry laser_odometry_;
}





using std::cout;
using std::endl;
using namespace std;
using namespace geometryutils;
typedef pcl::PointXYZI PointT;
typedef typename pcl::PointCloud<PointT> PointCloud;
typedef typename pcl::PointCloud<PointT>::Ptr PointCloudPtr;
typedef typename pcl::PointCloud<PointT>::ConstPtr PointCloudConstPtr;
typedef Twist<double> Transform;

//局部地图点云
PointCloudPtr local_surf_points_ptr_, local_surf_points_filtered_ptr_;
PointCloudPtr local_corner_points_ptr_, local_corner_points_filtered_ptr_;
//第一个局部地图构建标识
bool init_local_map_ = false;
pcl::VoxelGrid<pcl::PointXYZI> down_size_filter_corner_;   ///< voxel filter for down sizing corner clouds
pcl::VoxelGrid<pcl::PointXYZI> down_size_filter_surf_;     ///< voxel filter for down sizing surface clouds
pcl::VoxelGrid<pcl::PointXYZI> down_size_filter_map_;      ///< voxel filter for down sizing accumulated map
float corner_filter_size = 0.2;
float surf_filter_size = 0.4;
float map_filter_size = 0.6;

//ceres优化用的double数组
double **para_pose_;
double **para_speed_bias_;
//todo 外参不估计 和外参相关的内容需要适配 主要是ceres求残差部分
//double para_ex_pose_[SIZE_POSE];
double g_norm_;
bool gravity_fixed_ = false;
  Vector3d P_pivot_;
  Matrix3d R_pivot_;

PointCloudPtr laser_cloud_corner_last_;   ///< last corner points cloud
PointCloudPtr laser_cloud_surf_last_;     ///< last surface points cloud
PointCloudPtr full_cloud_;      ///< last full resolution cloud

//lio-mapping中mapping部分用到的
float scan_period_;
float time_factor_;
long map_frame_count_;
const int num_stack_frames_ = 1;
long frame_count_ = num_stack_frames_ - 1;        ///< number of processed frames
const int num_map_frames_;
int extrinsic_stage_ = 1;

Transform transform_sum_;
Transform transform_tobe_mapped_;
Transform transform_bef_mapped_;
Transform transform_aft_mapped_;

Transform transform_tobe_mapped_bef_;
Transform transform_es_;

ros::Time time_laser_cloud_corner_last_;   ///< time of current last corner cloud
ros::Time time_laser_cloud_surf_last_;     ///< time of current last surface cloud
ros::Time time_laser_full_cloud_;      ///< time of current full resolution cloud
ros::Time time_laser_odometry_;          ///< time of current laser odometry

bool new_laser_cloud_corner_last_;  ///< flag if a new last corner cloud has been received
bool new_laser_cloud_surf_last_;    ///< flag if a new last surface cloud has been received
bool new_laser_full_cloud_;     ///< flag if a new full resolution cloud has been received
bool new_laser_odometry_;         ///< flag if a new laser odometry has been received

bool is_ros_setup_ = false;
bool compact_data_ = false;
bool imu_inited_ = false;

//end lio-mapping mapping

//滑窗
//todo 滑窗相关的buffer和vins命名稍有不同 需要全换成lio_mapping的
CircularBuffer<PairTimeLaserTransform> all_laser_transforms_{window_size + 1};
CircularBuffer<Vector3d> Ps_{window_size + 1};
CircularBuffer<Matrix3d> Rs_{window_size + 1};
CircularBuffer<Vector3d> Vs_{window_size + 1};
CircularBuffer<Vector3d> Bas_{window_size + 1};
CircularBuffer<Vector3d> Bgs_{window_size + 1};

CircularBuffer<std_msgs::Header> Headers_{window_size + 1};
//用于点云去畸变
CircularBuffer<StampedTransform> imu_stampedtransforms{100};

CircularBuffer<vector<double> > dt_buf_{window_size + 1};
CircularBuffer<vector<Vector3d> > linear_acceleration_buf_{window_size + 1};
CircularBuffer<vector<Vector3d> > angular_velocity_buf_{window_size + 1};

CircularBuffer<shared_ptr<IntegrationBase> > pre_integrations_{window_size + 1};
CircularBuffer<PointCloudPtr> surf_stack_{window_size + 1};
CircularBuffer<PointCloudPtr> corner_stack_{window_size + 1};
CircularBuffer<PointCloudPtr> full_stack_{window_size + 1};

///> optimization buffers
CircularBuffer<bool> opt_point_coeff_mask_{opt_window_size + 1};
CircularBuffer<ScorePointCoeffMap> opt_point_coeff_map_{opt_window_size + 1};
CircularBuffer<CubeCenter> opt_cube_centers_{opt_window_size + 1};
CircularBuffer<Transform> opt_transforms_{opt_window_size + 1};
CircularBuffer<vector<size_t> > opt_valid_idx_{opt_window_size + 1};
CircularBuffer<PointCloudPtr> opt_corner_stack_{opt_window_size + 1};
CircularBuffer<PointCloudPtr> opt_surf_stack_{opt_window_size + 1};

CircularBuffer<Eigen::Matrix<double, 6, 6>> opt_matP_{opt_window_size + 1};
///< optimization buffers
//todo 需要追踪一下cir_buf_count_
size_t cir_buf_count_ = 0;
size_t laser_odom_recv_count_ = 0;

//todo 参数设定 有些可能用不上 再删除
//todo factor可能会和真的factor冲突
//边缘化因子
bool marginalization_factor = true;
//imu预积分因子
bool imu_factor = true;
bool run_optimization = true;
bool update_laser_imu = true;
bool gravity_fix = true;
bool plane_projection_factor = true;

bool point_distance_factor = false;
bool prior_factor = false;
bool marginalization_factor = true;
bool pcl_viewer = false;
bool convergence_flag_ = false;
//边缘化所需
MarginalizationInfo *last_marginalization_info;
vector<double *> last_marginalization_parameter_blocks;
vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d> > marg_coeffi, marg_coeffj;
vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > marg_pointi, marg_pointj;
vector<double> marg_score;



bool enable_deskew = true; ///< if disable, deskew from PointOdometry will be used
bool cutoff_deskew = false;
bool keep_features = false;

// IMU参数
float imuAccNoise = 3.9939570888238808e-03;          // 加速度噪声标准差
float imuGyrNoise = 1.5636343949698187e-03;          // 角速度噪声标准差
float imuAccBiasN = 6.4356659353532566e-05;          //
float imuGyrBiasN = 3.5640318696367613e-05;
float imuGravity = 9.80511;           // 重力加速度
float imuRPYWeight = 0.01;
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
//不估计外参 这个永远是固定的 vins中和相机数量一致 这里就一个 这里要给lidar to imu
Matrix3d ric = extRot.transpose();
Vector3d tic = -ric * extTrans;
Transform transform_lb_{Eigen::Quaterniond(extRPY), extTrans}; ///< Base to laser transform
Eigen::Matrix3d R_WI_; ///< R_WI is the rotation from the inertial frame into Lidar's world frame
Eigen::Quaterniond Q_WI_; ///< Q_WI is the rotation from the inertial frame into Lidar's world frame

//todo 变量命名vins替换成lio-mapping
//getMeasurements版本用这个
const int window_size = 10;
const int opt_window_size = 5;
int init_window_factor = 3;
int estimate_extrinsic = 2;

bool init_imu = 1;
bool init_odom = 0;
double latest_time;
double current_time = -1;
//frame_count到达window_size后就不再增加
int frame_count = 0;
//key为gtsam使用的变量索引 会持续增加
int key = 0;
Eigen::Vector3d tmp_P;
Eigen::Quaterniond tmp_Q;
Eigen::Vector3d tmp_V;
Eigen::Vector3d tmp_Ba;
Eigen::Vector3d tmp_Bg;
//imu中值积分的时候前一刻的imu信息 临时存储变量
//Eigen::Vector3d acc_0;
//Eigen::Vector3d gyr_0;
Eigen::Vector3d acc_last_, gyr_last_;
Eigen::Vector3d g_vec_;


bool first_imu = false;
//imu预积分滑窗
//todo 把vins替换成lio-mapping
IntegrationBase *pre_integrations[(window_size + 1)];
//IntegrationBase *tmp_pre_integration;
std::shared_ptr<IntegrationBase> tmp_pre_integration_;
vector<double> dt_buf[(window_size + 1)];
vector<Vector3d> linear_acceleration_buf[(window_size + 1)];
vector<Vector3d> angular_velocity_buf[(window_size + 1)];
//todo 这个要改用lio-mapping的
    Vector3d Ps[(window_size + 1)];
    Vector3d Vs[(window_size + 1)];
    Matrix3d Rs[(window_size + 1)];
    Vector3d Bas[(window_size + 1)];
    Vector3d Bgs[(window_size + 1)];
std_msgs::Header Headers[(window_size + 1)];
  CircularBuffer<PointCloudPtr> surf_stack_{window_size + 1};
  CircularBuffer<PointCloudPtr> corner_stack_{window_size + 1};
  CircularBuffer<PointCloudPtr> full_stack_{window_size + 1};
//lidar帧数据结构
class LidarFrame
{
    public:
        LidarFrame(){};
        LidarFrame(double _t):t{_t},is_key_frame{true} {};
//        map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>> > > points;
        double t;
        Matrix3d R;
        Vector3d T;
        IntegrationBase *pre_integration;
        bool is_key_frame;
};
map<double, LidarFrame> all_lidar_frame;
enum SolverFlag
{
    INITIAL,
    NON_LINEAR
};



struct LaserTransform {
  LaserTransform() {};
  LaserTransform(double laser_time, Transform laser_transform) : time{laser_time}, transform{laser_transform} {};

  double time;
  Transform transform;
  shared_ptr<IntegrationBase> pre_integration;

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

};

struct StampedTransform {
  double time;
  Transform transform;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

enum EstimatorStageFlag {
  NOT_INITED,
  INITED,
};
EstimatorStageFlag stage_flag_ = NOT_INITED;
  bool first_imu_ = false;
  double initial_time_ = -1;

//todo vins换lio-mapping
SolverFlag solver_flag = INITIAL;
double initial_timestamp = 0;
Eigen::Vector3d G{0.0, 0.0, 9.8};
int ESTIMATE_EXTRINSIC = 0;
Matrix3d back_R0, last_R, last_R0;
Vector3d back_P0, last_P, last_P0;

//todo 优化器使用的double数组 这个适配成lio-mapping
//外参不用估计
double para_Pose[window_size + 1][SIZE_POSE];
double para_SpeedBias[window_size + 1][SIZE_SPEEDBIAS];



bool failure_occur = 0;
vector<Vector3d> key_poses;



double timeLaserOdometry = 0.0;
double timeCenter= 0.0;
double timeSurf= 0.0;
double timeEdge= 0.0;



nav_msgs::Path globalPath;

//激光里程计回调函数
void laserOdometryHandler(const nav_msgs::Odometry::ConstPtr &_laserOdometry)
{
    if (!init_odom)
    {
      //其实可以不用跳
        //skip the first detected feature, which doesn't contain optical flow speed
        init_odom = 1;
        return;
    }
    mBuf.lock();
    odometryBuf.push(_laserOdometry);
    mBuf.unlock();
    con.notify_one();
}



//对imu做积分 把最新状态赋值给最新lidar帧
void processIMU(double dt, const Vector3d &linear_acceleration, const Vector3d &angular_velocity)
{
    if (!first_imu)
    {
        first_imu = true;
        acc_0 = linear_acceleration;
        gyr_0 = angular_velocity;
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

void solveGyroscopeBias(map<double, ImageFrame> &all_lidar_frame, Vector3d* Bgs)
{
    Matrix3d A;
    Vector3d b;
    Vector3d delta_bg;
    A.setZero();
    b.setZero();
    map<double, ImageFrame>::iterator frame_i;
    map<double, ImageFrame>::iterator frame_j;
    for (frame_i = all_lidar_frame.begin(); next(frame_i) != all_lidar_frame.end(); frame_i++)
    {
        frame_j = next(frame_i);
        MatrixXd tmp_A(3, 3);
        tmp_A.setZero();
        VectorXd tmp_b(3);
        tmp_b.setZero();
        Eigen::Quaterniond q_ij(frame_i->second.R.transpose() * frame_j->second.R);

        tmp_A = frame_j->second.pre_integration->jacobian.template block<3, 3>(O_R, O_BG);
        tmp_b = 2 * (frame_j->second.pre_integration->delta_q.inverse() * q_ij).vec();
        A += tmp_A.transpose() * tmp_A;
        b += tmp_A.transpose() * tmp_b;

    }
    delta_bg = A.ldlt().solve(b);
    ROS_WARN_STREAM("gyroscope bias initial calibration " << delta_bg.transpose());

    for (int i = 0; i <= window_size; i++)
        Bgs[i] += delta_bg;

    for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end( ); frame_i++)
    {
        frame_j = next(frame_i);
        frame_j->second.pre_integration->repropagate(Vector3d::Zero(), Bgs[0]);
    }
}

MatrixXd TangentBasis(Vector3d &g0)
{
    Vector3d b, c;
    Vector3d a = g0.normalized();
    Vector3d tmp(0, 0, 1);
    if(a == tmp)
        tmp << 1, 0, 0;
    b = (tmp - a * (a.transpose() * tmp)).normalized();
    c = a.cross(b);
    MatrixXd bc(3, 2);
    bc.block<3, 1>(0, 0) = b;
    bc.block<3, 1>(0, 1) = c;
    return bc;
}

/**
 * @brief 得到了一个初始的重力向量，引入重力大小作为先验，再进行几次迭代优化，求解最终的变量
 *
 * @param[in] all_image_frame
 * @param[in] g
 * @param[in] x
 */
void RefineGravity(map<double, LidarFrame> &all_lidar_frame, Vector3d &g, VectorXd &x)
{
    // 参考论文
    Vector3d g0 = g.normalized() * G.norm();
    Vector3d lx, ly;
    //VectorXd x;
    int all_frame_count = all_lidar_frame.size();
    int n_state = all_frame_count * 3 + 2 + 1;

    MatrixXd A{n_state, n_state};
    A.setZero();
    VectorXd b{n_state};
    b.setZero();

    map<double, LidarFrame>::iterator frame_i;
    map<double, LidarFrame>::iterator frame_j;
    for(int k = 0; k < 4; k++)
    {
        MatrixXd lxly(3, 2);
        lxly = TangentBasis(g0);
        int i = 0;
        for (frame_i = all_lidar_frame.begin(); next(frame_i) != all_lidar_frame.end(); frame_i++, i++)
        {
            frame_j = next(frame_i);

            MatrixXd tmp_A(6, 9);
            tmp_A.setZero();
            VectorXd tmp_b(6);
            tmp_b.setZero();

            double dt = frame_j->second.pre_integration->sum_dt;


            tmp_A.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();
            tmp_A.block<3, 2>(0, 6) = frame_i->second.R.transpose() * dt * dt / 2 * Matrix3d::Identity() * lxly;
            tmp_A.block<3, 1>(0, 8) = frame_i->second.R.transpose() * (frame_j->second.T - frame_i->second.T) / 100.0;
            tmp_b.block<3, 1>(0, 0) = frame_j->second.pre_integration->delta_p + frame_i->second.R.transpose() * frame_j->second.R * tic - tic - frame_i->second.R.transpose() * dt * dt / 2 * g0;

            tmp_A.block<3, 3>(3, 0) = -Matrix3d::Identity();
            tmp_A.block<3, 3>(3, 3) = frame_i->second.R.transpose() * frame_j->second.R;
            tmp_A.block<3, 2>(3, 6) = frame_i->second.R.transpose() * dt * Matrix3d::Identity() * lxly;
            tmp_b.block<3, 1>(3, 0) = frame_j->second.pre_integration->delta_v - frame_i->second.R.transpose() * dt * Matrix3d::Identity() * g0;


            Matrix<double, 6, 6> cov_inv = Matrix<double, 6, 6>::Zero();
            //cov.block<6, 6>(0, 0) = IMU_cov[i + 1];
            //MatrixXd cov_inv = cov.inverse();
            cov_inv.setIdentity();

            MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A;
            VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b;

            A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
            b.segment<6>(i * 3) += r_b.head<6>();

            A.bottomRightCorner<3, 3>() += r_A.bottomRightCorner<3, 3>();
            b.tail<3>() += r_b.tail<3>();

            A.block<6, 3>(i * 3, n_state - 3) += r_A.topRightCorner<6, 3>();
            A.block<3, 6>(n_state - 3, i * 3) += r_A.bottomLeftCorner<3, 6>();
        }
            A = A * 1000.0;
            b = b * 1000.0;
            x = A.ldlt().solve(b);
            VectorXd dg = x.segment<2>(n_state - 3);
            g0 = (g0 + lxly * dg).normalized() * G.norm();
            //double s = x(n_state - 1);
    }
    g = g0;
}

/**
 * @brief 求解各帧的速度，枢纽帧的重力方向，以及尺度
 *
 * @param[in] all_image_frame
 * @param[in] g
 * @param[in] x
 * @return true
 * @return false
 */
//todo 这个需要确定能否直接用
bool LinearAlignment(map<double, LidarFrame> &all_lidar_frame, Vector3d &g, VectorXd &x)
{
    // 这一部分内容对照论文进行理解
    int all_frame_count = all_lidar_frame.size();
    //todo 这个对不对
    int n_state = all_frame_count * 3 + 3 + 1;

    MatrixXd A{n_state, n_state};
    A.setZero();
    VectorXd b{n_state};
    b.setZero();

    map<double, LidarFrame>::iterator frame_i;
    map<double, LidarFrame>::iterator frame_j;
    int i = 0;
    for (frame_i = all_lidar_frame.begin(); next(frame_i) != all_lidar_frame.end(); frame_i++, i++)
    {
        frame_j = next(frame_i);

        MatrixXd tmp_A(6, 10);
        tmp_A.setZero();
        VectorXd tmp_b(6);
        tmp_b.setZero();

        double dt = frame_j->second.pre_integration->sum_dt;

        tmp_A.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();
        tmp_A.block<3, 3>(0, 6) = frame_i->second.R.transpose() * dt * dt / 2 * Matrix3d::Identity();
        tmp_A.block<3, 1>(0, 9) = frame_i->second.R.transpose() * (frame_j->second.T - frame_i->second.T) / 100.0;
        tmp_b.block<3, 1>(0, 0) = frame_j->second.pre_integration->delta_p + frame_i->second.R.transpose() * frame_j->second.R * tic - tic;
        //cout << "delta_p   " << frame_j->second.pre_integration->delta_p.transpose() << endl;
        tmp_A.block<3, 3>(3, 0) = -Matrix3d::Identity();
        tmp_A.block<3, 3>(3, 3) = frame_i->second.R.transpose() * frame_j->second.R;
        tmp_A.block<3, 3>(3, 6) = frame_i->second.R.transpose() * dt * Matrix3d::Identity();
        tmp_b.block<3, 1>(3, 0) = frame_j->second.pre_integration->delta_v;
        //cout << "delta_v   " << frame_j->second.pre_integration->delta_v.transpose() << endl;

        Matrix<double, 6, 6> cov_inv = Matrix<double, 6, 6>::Zero();
        //cov.block<6, 6>(0, 0) = IMU_cov[i + 1];
        //MatrixXd cov_inv = cov.inverse();
        cov_inv.setIdentity();

        MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A;
        VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b;

        A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
        b.segment<6>(i * 3) += r_b.head<6>();

        A.bottomRightCorner<4, 4>() += r_A.bottomRightCorner<4, 4>();
        b.tail<4>() += r_b.tail<4>();

        A.block<6, 4>(i * 3, n_state - 4) += r_A.topRightCorner<6, 4>();
        A.block<4, 6>(n_state - 4, i * 3) += r_A.bottomLeftCorner<4, 6>();
    }
    // 增强数值稳定性
    A = A * 1000.0;
    b = b * 1000.0;
    x = A.ldlt().solve(b);
    double s = x(n_state - 1) / 100.0;
    ROS_DEBUG("estimated scale: %f", s);
    g = x.segment<3>(n_state - 4);
    ROS_DEBUG_STREAM(" result g     " << g.norm() << " " << g.transpose());
    // 做一些检查
    if(fabs(g.norm() - G.norm()) > 1.0 || s < 0)
    {
        return false;
    }
    // 重力修复
    RefineGravity(all_image_frame, g, x);
    // 得到真实尺度
    s = (x.tail<1>())(0) / 100.0;
    (x.tail<1>())(0) = s;
    ROS_DEBUG_STREAM(" refine     " << g.norm() << " " << g.transpose());
    if(s < 0.0 )
        return false;
    else
        return true;
}

bool LidarIMUAlignment(map<double, LidarFrame> &all_lidar_frame, Vector3d* Bgs, Vector3d &g, VectorXd &x)
{
    solveGyroscopeBias(all_lidar_frame, Bgs);
    if(LinearAlignment(all_lidar_frame, g, x))
        return true;
    else
        return false;
}

bool lidarInitialAlign()
{
    TicToc t_g;
    VectorXd x;
    //solve scale
    //vins中求尺度 lio已经有尺度了 但是vins中这一步给出了零偏值
    bool result = LidarIMUAlignment(all_lidar_frame, Bgs, g, x);
    if(!result)
    {
        ROS_DEBUG("solve g failed!");
        return false;
    }

    // change state
    for (int i = 0; i <= frame_count; i++)
    {
        Matrix3d Ri = all_lidar_frame[Headers[i].stamp.toSec()].R;
        Vector3d Pi = all_lidar_frame[Headers[i].stamp.toSec()].T;
        Ps[i] = Pi;
        Rs[i] = Ri;
        all_lidar_frame[Headers[i].stamp.toSec()].is_key_frame = true;
    }

    double s = (x.tail<1>())(0);
    for (int i = 0; i <= window_size; i++)
    {
        pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]);
    }
    for (int i = frame_count; i >= 0; i--)
        Ps[i] = s * Ps[i] - Rs[i] * tic - (s * Ps[0] - Rs[0] * tic);
    int kv = -1;
    map<double, LidarFrame>::iterator frame_i;
    for (frame_i = all_lidar_frame.begin(); frame_i != all_lidar_frame.end(); frame_i++)
    {
        if(frame_i->second.is_key_frame)
        {
            kv++;
            Vs[kv] = frame_i->second.R * x.segment<3>(kv * 3);
        }
    }

    Matrix3d R0 = Utility::g2R(g);
    double yaw = Utility::R2ypr(R0 * Rs[0]).x();
    R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
    g = R0 * g;
    //Matrix3d rot_diff = R0 * Rs[0].transpose();
    Matrix3d rot_diff = R0;
    for (int i = 0; i <= frame_count; i++)
    {
        Ps[i] = rot_diff * Ps[i];
        Rs[i] = rot_diff * Rs[i];
        Vs[i] = rot_diff * Vs[i];
    }
    ROS_DEBUG_STREAM("g0     " << g.transpose());
    ROS_DEBUG_STREAM("my R0  " << Utility::R2ypr(Rs[0]).transpose());

    return true;
}

//初始化
bool initialStructure()
{
    TicToc t_sfm;
    //check imu observibility
    {
        map<double, LidarFrame>::iterator frame_it;
        Vector3d sum_g;
        for (frame_it = all_lidar_frame.begin(), frame_it++; frame_it != all_lidar_frame.end(); frame_it++)
        {
            double dt = frame_it->second.pre_integration->sum_dt;
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
            sum_g += tmp_g;
        }
        Vector3d aver_g;
        aver_g = sum_g * 1.0 / ((int)all_lidar_frame.size() - 1);
        double var = 0;
        for (frame_it = all_lidar_frame.begin(), frame_it++; frame_it != all_lidar_frame.end(); frame_it++)
        {
            double dt = frame_it->second.pre_integration->sum_dt;
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
            var += (tmp_g - aver_g).transpose() * (tmp_g - aver_g);
            //cout << "frame g " << tmp_g.transpose() << endl;
        }
        var = sqrt(var / ((int)all_lidar_frame.size() - 1));
        //ROS_WARN("IMU variation %f!", var);
        //IMU平均加速度不够也不中断 提示一下
        if(var < 0.25)
        {
            ROS_INFO("IMU excitation not enouth!");
            //return false;
        }
    }
    // global sfm
    //solve pnp for all frame
	//vins中这两步的目的就是把all_image_frame中滑窗内的帧的初始位姿求出来，这里all_lidar_frame已经有初始位姿了
    if (lidarInitialAlign())
        return true;
    else
    {
        ROS_INFO("misalign lidar structure with IMU");
        return false;
    }

}

void vector2double() {
    for (int i = 0; i <= window_size; i++)
    {
        para_Pose[i][0] = Ps[i].x();
        para_Pose[i][1] = Ps[i].y();
        para_Pose[i][2] = Ps[i].z();
        Quaterniond q{Rs[i]};
        para_Pose[i][3] = q.x();
        para_Pose[i][4] = q.y();
        para_Pose[i][5] = q.z();
        para_Pose[i][6] = q.w();

        para_SpeedBias[i][0] = Vs[i].x();
        para_SpeedBias[i][1] = Vs[i].y();
        para_SpeedBias[i][2] = Vs[i].z();

        para_SpeedBias[i][3] = Bas[i].x();
        para_SpeedBias[i][4] = Bas[i].y();
        para_SpeedBias[i][5] = Bas[i].z();

        para_SpeedBias[i][6] = Bgs[i].x();
        para_SpeedBias[i][7] = Bgs[i].y();
        para_SpeedBias[i][8] = Bgs[i].z();
    }
}

void double2vector()
{
    Vector3d origin_R0 = Utility::R2ypr(Rs[0]);
    Vector3d origin_P0 = Ps[0];

    if (failure_occur)
    {
        origin_R0 = Utility::R2ypr(last_R0);
        origin_P0 = last_P0;
        failure_occur = 0;
    }
    //todo 可能要改 这里是保持滑窗首帧yaw角不变
    Vector3d origin_R00 = Utility::R2ypr(Quaterniond(para_Pose[0][6],
                                                      para_Pose[0][3],
                                                      para_Pose[0][4],
                                                      para_Pose[0][5]).toRotationMatrix());
    double y_diff = origin_R0.x() - origin_R00.x();
    Matrix3d rot_diff = Utility::ypr2R(Vector3d(y_diff, 0, 0));
    if (abs(abs(origin_R0.y()) - 90) < 1.0 || abs(abs(origin_R00.y()) - 90) < 1.0)
    {
        ROS_DEBUG("euler singular point!");
        rot_diff = Rs[0] * Quaterniond(para_Pose[0][6],
                                       para_Pose[0][3],
                                       para_Pose[0][4],
                                       para_Pose[0][5]).toRotationMatrix().transpose();
    }

    for (int i = 0; i <= window_size; i++)
    {

        Rs[i] = rot_diff * Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();

        Ps[i] = rot_diff * Vector3d(para_Pose[i][0] - para_Pose[0][0],
                                para_Pose[i][1] - para_Pose[0][1],
                                para_Pose[i][2] - para_Pose[0][2]) + origin_P0;

        Vs[i] = rot_diff * Vector3d(para_SpeedBias[i][0],
                                    para_SpeedBias[i][1],
                                    para_SpeedBias[i][2]);

        Bas[i] = Vector3d(para_SpeedBias[i][3],
                          para_SpeedBias[i][4],
                          para_SpeedBias[i][5]);

        Bgs[i] = Vector3d(para_SpeedBias[i][6],
                          para_SpeedBias[i][7],
                          para_SpeedBias[i][8]);
    }

    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        dep(i) = para_Feature[i][0];
    f_manager.setDepth(dep);

    // relative info between two loop frame
    if(relocalization_info)
    {
        Matrix3d relo_r;
        Vector3d relo_t;
        relo_r = rot_diff * Quaterniond(relo_Pose[6], relo_Pose[3], relo_Pose[4], relo_Pose[5]).normalized().toRotationMatrix();
        relo_t = rot_diff * Vector3d(relo_Pose[0] - para_Pose[0][0],
                                     relo_Pose[1] - para_Pose[0][1],
                                     relo_Pose[2] - para_Pose[0][2]) + origin_P0;
        double drift_correct_yaw;
        drift_correct_yaw = Utility::R2ypr(prev_relo_r).x() - Utility::R2ypr(relo_r).x();
        drift_correct_r = Utility::ypr2R(Vector3d(drift_correct_yaw, 0, 0));
        drift_correct_t = prev_relo_t - drift_correct_r * relo_t;
        relo_relative_t = relo_r.transpose() * (Ps[relo_frame_local_index] - relo_t);
        relo_relative_q = relo_r.transpose() * Rs[relo_frame_local_index];
        relo_relative_yaw = Utility::normalizeAngle(Utility::R2ypr(Rs[relo_frame_local_index]).x() - Utility::R2ypr(relo_r).x());
        //cout << "vins relo " << endl;
        //cout << "vins relative_t " << relo_relative_t.transpose() << endl;
        //cout << "vins relative_yaw " <<relo_relative_yaw << endl;
        relocalization_info = 0;

    }
}

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
  if (!keep_features) {
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

  const PointCloudPtr &origin_surf_points = surf_stack;
  const Transform &transform_to_local = local_transform;
  size_t surf_points_size = origin_surf_points->points.size();

#ifdef USE_CORNER
  const PointCloudPtr &origin_corner_points = corner_stack;
  size_t corner_points_size = origin_corner_points->points.size();
#endif

//    DLOG(INFO) << "transform_to_local: " << transform_to_local;

  for (int i = 0; i < surf_points_size; i++) {
    point_ori = origin_surf_points->points[i];
    PointAssociateToMap(point_ori, point_sel, transform_to_local);

    int num_neighbors = 5;
    kdtree_surf_from_map->nearestKSearch(point_sel, num_neighbors, point_search_idx, point_search_sq_dis);

    if (point_search_sq_dis[num_neighbors - 1] < min_match_sq_dis_) {
      for (int j = 0; j < num_neighbors; j++) {
        mat_A0(j, 0) = local_surf_points_filtered_ptr->points[point_search_idx[j]].x;
        mat_A0(j, 1) = local_surf_points_filtered_ptr->points[point_search_idx[j]].y;
        mat_A0(j, 2) = local_surf_points_filtered_ptr->points[point_search_idx[j]].z;
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

//todo lio滑窗构建局部地图 填充feature_frames
void BuildLocalMap(vector<FeaturePerFrame> &feature_frames) {
	feature_frames.clear();

	TicToc t_build_map;

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

	vector<Transform> local_transforms;
	int pivot_idx = window_size - opt_window_size;

	Twist<double> transform_lb = transform_lb_.cast<double>();

	Eigen::Vector3d Ps_pivot = Ps[pivot_idx];
	Eigen::Matrix3d Rs_pivot = Rs[pivot_idx];

	Quaterniond rot_pivot(Rs_pivot * transform_lb.rot.inverse());
	Eigen::Vector3d pos_pivot = Ps_pivot - rot_pivot * transform_lb.pos;

	Twist<double> transform_pivot = Twist<double>(rot_pivot, pos_pivot);

	{
    	if (!init_local_map_) {
    		PointCloud transformed_cloud_surf, tmp_cloud_surf;
#ifdef USE_CORNER
    		PointCloud transformed_cloud_corner, tmp_cloud_corner;
#endif
    		for (int i = 0; i <= pivot_idx; ++i) {
    			Eigen::Vector3d Ps_i = Ps[i];
    			Eigen::Matrix3d Rs_i = Rs[i];

    			Quaterniond rot_li(Rs_i * transform_lb.rot.inverse());
    			Eigen::Vector3d pos_li = Ps_i - rot_li * transform_lb.pos;

    			Twist<double> transform_li = Twist<double>(rot_li, pos_li);
    			Eigen::Affine3f transform_pivot_i = (transform_pivot.inverse() * transform_li).cast<float>().transform();
    			//todo surf_stack_等的消息的填入
                pcl::transformPointCloud(*(surf_stack_[i]), transformed_cloud_surf, transform_pivot_i);
    			tmp_cloud_surf += transformed_cloud_surf;

#ifdef USE_CORNER
    			pcl::transformPointCloud(*(corner_stack_[i]), transformed_cloud_corner, transform_pivot_i);
    			tmp_cloud_corner += transformed_cloud_corner;
#endif
			}

    		*(surf_stack_[pivot_idx]) = tmp_cloud_surf;
#ifdef USE_CORNER
    		*(corner_stack_[pivot_idx]) = tmp_cloud_corner;
#endif
    		init_local_map_ = true;
    	}

    	for (int i = 0; i < window_size + 1; ++i) {

      		Eigen::Vector3d Ps_i = Ps[i];
      		Eigen::Matrix3d Rs_i = Rs[i];

      		Quaterniond rot_li(Rs_i * transform_lb.rot.inverse());
      		Eigen::Vector3d pos_li = Ps_i - rot_li * transform_lb.pos;

    		Twist<double> transform_li = Twist<double>(rot_li, pos_li);
      		Eigen::Affine3f transform_pivot_i = (transform_pivot.inverse() * transform_li).cast<float>().transform();

      		Transform local_transform = transform_pivot_i;
      		local_transforms.push_back(local_transform);

      		if (i < pivot_idx) {
        		continue;
      		}

      		PointCloud transformed_cloud_surf, transformed_cloud_corner;

      		// NOTE: exclude the latest one
      		if (i != window_size) {
        		if (i == pivot_idx) {
          			*local_surf_points_ptr_ += *(surf_stack_[i]);
//	        	down_size_filter_surf_.setInputCloud(local_surf_points_ptr_);
//	          down_size_filter_surf_.filter(transformed_cloud_surf);
//	          *local_surf_points_ptr_ = transformed_cloud_surf;
#ifdef USE_CORNER
          			*local_corner_points_ptr_ += *(corner_stack_[i]);
#endif
          			continue;
        		}

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

  	ROS_DEBUG_STREAM("t_build_map cost: " << t_build_map.Toc() << " ms");
  	DLOG(INFO) << "t_build_map cost: " << t_build_map.Toc() << " ms";

  	pcl::KdTreeFLANN<PointT>::Ptr kdtree_surf_from_map(new pcl::KdTreeFLANN<PointT>());
  	kdtree_surf_from_map->setInputCloud(local_surf_points_filtered_ptr_);

#ifdef USE_CORNER
  	pcl::KdTreeFLANN<PointT>::Ptr kdtree_corner_from_map(new pcl::KdTreeFLANN<PointT>());
  	kdtree_corner_from_map->setInputCloud(local_corner_points_filtered_ptr_);
#endif

	for (int idx = 0; idx < window_size + 1; ++idx) {

    	FeaturePerFrame feature_per_frame;
    	vector<unique_ptr<Feature>> features;
//    vector<unique_ptr<Feature>> &features = feature_per_frame.features;

    	TicToc t_features;

    	if (idx > pivot_idx) {
      		if (idx != window_size || !imu_factor) {
#ifdef USE_CORNER
        	CalculateFeatures(kdtree_surf_from_map, local_surf_points_filtered_ptr_, surf_stack_[idx],
                         kdtree_corner_from_map, local_corner_points_filtered_ptr_, corner_stack_[idx],
                          local_transforms[idx], features);
#else
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
        CalculateLaserOdom(kdtree_surf_from_map, local_surf_points_filtered_ptr_, surf_stack_[idx],
                           local_transforms[idx], features);
#endif

        DLOG(INFO) << "local_transforms[idx] aft" << local_transforms[idx];
      }
    } else {
      // NOTE: empty features
    }

    feature_per_frame.id = idx;
//    feature_per_frame.features = std::move(features);
    feature_per_frame.features.assign(make_move_iterator(features.begin()), make_move_iterator(features.end()));
    feature_frames.push_back(std::move(feature_per_frame));

    ROS_DEBUG_STREAM("feature cost: " << t_features.Toc() << " ms");
  }

}

void VectorToDouble() {
  int i, opt_i, pivot_idx = int(window_size - opt_window_size);
  P_pivot_ = Ps_[pivot_idx];
  R_pivot_ = Rs_[pivot_idx];
  for (i = 0, opt_i = pivot_idx; i < opt_window_size + 1; ++i, ++opt_i) {
    para_pose_[i][0] = Ps_[opt_i].x();
    para_pose_[i][1] = Ps_[opt_i].y();
    para_pose_[i][2] = Ps_[opt_i].z();
    Quaterniond q{Rs_[opt_i]};
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

//  {
//    /// base to lidar
//    para_ex_pose_[0] = transform_lb_.pos.x();
//    para_ex_pose_[1] = transform_lb_.pos.y();
//    para_ex_pose_[2] = transform_lb_.pos.z();
//    para_ex_pose_[3] = transform_lb_.rot.x();
//    para_ex_pose_[4] = transform_lb_.rot.y();
//    para_ex_pose_[5] = transform_lb_.rot.z();
//    para_ex_pose_[6] = transform_lb_.rot.w();
//  }
}

void Estimator::DoubleToVector() {
// FIXME: do we need to optimize the first state?
// WARNING: not just yaw angle rot_diff; if it is compared with global features, there should be no need for rot_diff

//  Quaterniond origin_R0{Rs_[0]};
  int pivot_idx = int(window_size - opt_window_size);
  Vector3d origin_P0 = Ps_[pivot_idx];
  Vector3d origin_R0 = R2ypr(Rs_[pivot_idx]);

  Vector3d origin_R00 = R2ypr(Quaterniond(para_pose_[0][6],
                                          para_pose_[0][3],
                                          para_pose_[0][4],
                                          para_pose_[0][5]).normalized().toRotationMatrix());
  // Z-axix R00 to R0, regard para_pose's R as rotate along the Z-axis first
  double y_diff = origin_R0.x() - origin_R00.x();

  //TODO
  Matrix3d rot_diff = ypr2R(Vector3d(y_diff, 0, 0));
  if (abs(abs(origin_R0.y()) - 90) < 1.0 || abs(abs(origin_R00.y()) - 90) < 1.0) {
    ROS_DEBUG("euler singular point!");
    rot_diff = Rs_[pivot_idx] * Quaterniond(para_pose_[0][6],
                                            para_pose_[0][3],
                                            para_pose_[0][4],
                                            para_pose_[0][5]).normalized().toRotationMatrix().transpose();
  }

//  DLOG(INFO) << "origin_P0" << origin_P0.transpose();

  {
    Eigen::Vector3d Ps_pivot = Ps_[pivot_idx];
    Eigen::Matrix3d Rs_pivot = Rs_[pivot_idx];
    Twist<double> trans_pivot{Eigen::Quaterniond{Rs_pivot}, Ps_pivot};

    Matrix3d R_opt_pivot = rot_diff * Quaterniond(para_pose_[0][6],
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
  for (i = 0, opt_i = pivot_idx; i < opt_window_size + 1; ++i, ++opt_i) {
//    DLOG(INFO) << "para aft: " << Vector3d(para_pose_[i][0], para_pose_[i][1], para_pose_[i][2]).transpose();

    Rs_[opt_i] = rot_diff * Quaterniond(para_pose_[i][6],
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
//  {
//    transform_lb_.pos = Vector3d(para_ex_pose_[0],
//                                 para_ex_pose_[1],
//                                 para_ex_pose_[2]).template cast<float>();
//    transform_lb_.rot = Quaterniond(para_ex_pose_[6],
//                                    para_ex_pose_[3],
//                                    para_ex_pose_[4],
//                                    para_ex_pose_[5]).template cast<float>();
//  }
}

//todo 优化核心部分 ceres实现滑窗优化
void SolveOptimization()
{
    if (cir_buf_count_ < window_size && imu_factor) {
    	LOG(ERROR) << "enter optimization before enough count: " << cir_buf_count_ << " < "
               	<< window_size;
    	return;
  	}

    TicToc tic_toc_opt;

    bool turn_off = true;
    ceres::Problem problem;
  	ceres::LossFunction *loss_function;
    //  loss_function = new ceres::HuberLoss(0.5);
  	loss_function = new ceres::CauchyLoss(1.0);
   	// NOTE: update from laser transform
  	if (update_laser_imu) {
    	DLOG(INFO) << "======= bef opt =======";
    	if (!imu_factor) {
      		Twist<double>
          		incre = (transform_lb_.inverse() * all_laser_transforms_[cir_buf_count_ - 1].second.transform.inverse()
          		* all_laser_transforms_[cir_buf_count_].second.transform * transform_lb_).cast<double>();
      		Ps_[cir_buf_count_] = Rs_[cir_buf_count_ - 1] * incre.pos + Ps_[cir_buf_count_ - 1];
      		Rs_[cir_buf_count_] = Rs_[cir_buf_count_ - 1] * incre.rot;
    	}
  	}

  	vector<FeaturePerFrame> feature_frames;
  	BuildLocalMap(feature_frames);
  	vector<double *> para_ids;
  	//region Add pose and speed bias parameters
  	for (int i = 0; i < opt_window_size + 1;
    	   ++i) {
    	ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
    	problem.AddParameterBlock(para_pose_[i], SIZE_POSE, local_parameterization);
    	problem.AddParameterBlock(para_speed_bias_[i], SIZE_SPEED_BIAS);
    	para_ids.push_back(para_pose_[i]);
    	para_ids.push_back(para_speed_bias_[i]);
  	}
  //endregion

    VectorToDouble();
    //边缘化因子
	vector<ceres::internal::ResidualBlock *> res_ids_marg;
	ceres::internal::ResidualBlock *res_id_marg = NULL;

  //region Marginalization residual
  if (marginalization_factor) {
    if (last_marginalization_info) {
      // construct new marginlization_factor
      MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
      res_id_marg = problem.AddResidualBlock(marginalization_factor, NULL,
                                             last_marginalization_parameter_blocks);
      res_ids_marg.push_back(res_id_marg);
    }
  }
  //endregion

  //imu预积分因子
  vector<ceres::internal::ResidualBlock *> res_ids_pim;
  if (imu_factor) {
    for (int i = 0; i < opt_window_size;
         ++i) {
      int j = i + 1;
      int opt_i = int(window_size - opt_window_size + i);
      int opt_j = opt_i + 1;
      //todo 预积分前面处理需要合并一下
      if (pre_integrations_[opt_j]->sum_dt_ > 10.0) {
        continue;
      }

      ImuFactor *f = new ImuFactor(pre_integrations_[opt_j]);
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
  if (point_distance_factor) {
    for (int i = 0; i < opt_window_size + 1; ++i) {
      int opt_i = int(window_size - opt_window_size + i);

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
          //todo 这里需要改一下 不古迹外参
          PivotPointPlaneFactor *f = new PivotPointPlaneFactor(p_eigen,
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
    if (imu_factor) {
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
    if (point_distance_factor) {
      e_option.parameter_blocks = para_ids;
      e_option.residual_blocks = res_ids_proj;
      problem.Evaluate(e_option, &cost_ppp, NULL, NULL, NULL);
      DLOG(INFO) << "bef_proj: " << cost_ppp;
    }
    if (marginalization_factor) {
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

  ROS_DEBUG_STREAM("t_opt: " << t_opt.Toc() << " ms");
  DLOG(INFO) <<"t_opt: " << t_opt.Toc() << " ms";

  DoubleToVector();

  //边缘化
  //region Constraint Marginalization
  if (marginalization_factor && !turn_off) {

    TicToc t_whole_marginalization;

    MarginalizationInfo *marginalization_info = new MarginalizationInfo();

    VectorToDouble();

    if (last_marginalization_info) {
      vector<int> drop_set;
      for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++) {
        if (last_marginalization_parameter_blocks[i] == para_pose_[0] ||
            last_marginalization_parameter_blocks[i] == para_speed_bias_[0])
          drop_set.push_back(i);
      }
      // construct new marginlization_factor
      MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
      ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                     last_marginalization_parameter_blocks,
                                                                     drop_set);

      marginalization_info->AddResidualBlockInfo(residual_block_info);
    }

    if (imu_factor) {
      int pivot_idx = window_size - opt_window_size;
      if (pre_integrations_[pivot_idx + 1]->sum_dt_ < 10.0) {
        ImuFactor *imu_factor_ = new ImuFactor(pre_integrations_[pivot_idx + 1]);
        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(imu_factor_, NULL,
                                                                       vector<double *>{para_pose_[0],
                                                                                        para_speed_bias_[0],
                                                                                        para_pose_[1],
                                                                                        para_speed_bias_[1]},
                                                                       vector<int>{0, 1});
        marginalization_info->AddResidualBlockInfo(residual_block_info);
      }
    }

    if (point_distance_factor) {
      for (int i = 1; i < opt_window_size + 1; ++i) {
        int opt_i = int(window_size - opt_window_size + i);

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

          PivotPointPlaneFactor *pivot_point_plane_factor = new PivotPointPlaneFactor(p_eigen,
                                                                                      coeff_eigen);

          ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(pivot_point_plane_factor, loss_function,
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
    ROS_DEBUG("pre marginalization %f ms", t_pre_margin.Toc());
    ROS_DEBUG_STREAM("pre marginalization: " << t_pre_margin.Toc() << " ms");

    TicToc t_margin;
    marginalization_info->Marginalize();
    ROS_DEBUG("marginalization %f ms", t_margin.Toc());
    ROS_DEBUG_STREAM("marginalization: " << t_margin.Toc() << " ms");

    std::unordered_map<long, double *> addr_shift;
    for (int i = 1; i < opt_window_size + 1; ++i) {
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

    DLOG(INFO) << "whole marginalization costs: " << t_whole_marginalization.Toc();
    ROS_DEBUG_STREAM("whole marginalization costs: " << t_whole_marginalization.Toc() << " ms");
  }
  //endregion

//  // NOTE: update to laser transform
//  if (update_laser_imu) {
//    DLOG(INFO) << "======= aft opt =======";
//    Twist<double> transform_lb = transform_lb_.cast<double>();
//    //tod
//    Transform &opt_l0_transform = opt_transforms_[0];
//    int opt_0 = int(window_size - opt_window_size + 0);
//    Quaterniond rot_l0(Rs_[opt_0] * transform_lb.rot.conjugate().normalized());
//    Eigen::Vector3d pos_l0 = Ps_[opt_0] - rot_l0 * transform_lb.pos;
//    opt_l0_transform = Twist<double>{rot_l0, pos_l0}.cast<float>(); // for updating the map
//
//    vector<Transform> imu_poses, lidar_poses;
//
//    for (int i = 0; i < opt_window_size + 1; ++i) {
//      int opt_i = int(window_size - opt_window_size + i);
//
//      Quaterniond rot_li(Rs_[opt_i] * transform_lb.rot.conjugate().normalized());
//      Eigen::Vector3d pos_li = Ps_[opt_i] - rot_li * transform_lb.pos;
//      Twist<double> transform_li = Twist<double>(rot_li, pos_li);
//
//      Twist<double> transform_bi = Twist<double>(Eigen::Quaterniond(Rs_[opt_i]), Ps_[opt_i]);
//      imu_poses.push_back(transform_bi.cast<float>());
//      lidar_poses.push_back(transform_li.cast<float>());
//
//    }
//
//    DLOG(INFO) << "velocity: " << Vs_.last().norm();
//    DLOG(INFO) << "transform_lb_: " << transform_lb_;
//
//    ROS_DEBUG_STREAM("lb in world: " << (rot_l0.normalized() * transform_lb.pos).transpose());
//
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
//
//      int pivot_idx = estimator_config_.window_size - estimator_config_.opt_window_size;
//
//      Eigen::Vector3d Ps_pivot = Ps_[pivot_idx];
//      Eigen::Matrix3d Rs_pivot = Rs_[pivot_idx];
//
//      Quaterniond rot_pivot(Rs_pivot * transform_lb.rot.inverse());
//      Eigen::Vector3d pos_pivot = Ps_pivot - rot_pivot * transform_lb.pos;
//      PublishCloudMsg(pub_local_surf_points_,
//                      *surf_stack_[pivot_idx + 1],
//                      Headers_[pivot_idx + 1].stamp,
//                      "/laser_local");
//
//      PublishCloudMsg(pub_local_corner_points_,
//                      *corner_stack_[pivot_idx + 1],
//                      Headers_[pivot_idx + 1].stamp,
//                      "/laser_local");
//
//      PublishCloudMsg(pub_local_full_points_,
//                      *full_stack_[pivot_idx + 1],
//                      Headers_[pivot_idx + 1].stamp,
//                      "/laser_local");
//
//      PublishCloudMsg(pub_map_surf_points_,
//                      *local_surf_points_filtered_ptr_,
//                      Headers_.last().stamp,
//                      "/laser_local");
//
//#ifdef USE_CORNER
//      PublishCloudMsg(pub_map_corner_points_,
//                      *local_corner_points_filtered_ptr_,
//                      Headers_.last().stamp,
//                      "/laser_local");
//#endif
//
//      laser_local_trans_.setOrigin(tf::Vector3{pos_pivot.x(), pos_pivot.y(), pos_pivot.z()});
//      laser_local_trans_.setRotation(tf::Quaternion{rot_pivot.x(), rot_pivot.y(), rot_pivot.z(), rot_pivot.w()});
//      laser_local_trans_.stamp_ = Headers_.last().stamp;
//      tf_broadcaster_est_.sendTransform(laser_local_trans_);
//
//      Eigen::Vector3d Ps_last = Ps_.last();
//      Eigen::Matrix3d Rs_last = Rs_.last();
//
//      Quaterniond rot_last(Rs_last * transform_lb.rot.inverse());
//      Eigen::Vector3d pos_last = Ps_last - rot_last * transform_lb.pos;
//
//      Quaterniond rot_predict = (rot_pivot.inverse() * rot_last).normalized();
//      Eigen::Vector3d pos_predict = rot_pivot.inverse() * (Ps_last - Ps_pivot);
//
//      PublishCloudMsg(pub_predict_surf_points_, *(surf_stack_.last()), Headers_.last().stamp, "/laser_predict");
//      PublishCloudMsg(pub_predict_full_points_, *(full_stack_.last()), Headers_.last().stamp, "/laser_predict");
//
//      {
//        // NOTE: full stack into end of the scan
////        PointCloudPtr tmp_points_ptr = boost::make_shared<PointCloud>(PointCloud());
////        *tmp_points_ptr = *(full_stack_.last());
////        TransformToEnd(tmp_points_ptr, transform_es_, 10);
////        PublishCloudMsg(pub_predict_corrected_full_points_,
////                        *tmp_points_ptr,
////                        Headers_.last().stamp,
////                        "/laser_predict");
//
//        TransformToEnd(full_stack_.last(), transform_es_, 10, true);
//        PublishCloudMsg(pub_predict_corrected_full_points_,
//                        *(full_stack_.last()),
//                        Headers_.last().stamp,
//                        "/laser_predict");
//      }
//
//#ifdef USE_CORNER
//      PublishCloudMsg(pub_predict_corner_points_, *(corner_stack_.last()), Headers_.last().stamp, "/laser_predict");
//#endif
//      laser_predict_trans_.setOrigin(tf::Vector3{pos_predict.x(), pos_predict.y(), pos_predict.z()});
//      laser_predict_trans_.setRotation(tf::Quaternion{rot_predict.x(), rot_predict.y(), rot_predict.z(),
//                                                      rot_predict.w()});
//      laser_predict_trans_.stamp_ = Headers_.last().stamp;
//      tf_broadcaster_est_.sendTransform(laser_predict_trans_);
//    }
//
//  }

  DLOG(INFO) << "tic_toc_opt: " << tic_toc_opt.Toc() << " ms";
  ROS_DEBUG_STREAM("tic_toc_opt: " << tic_toc_opt.Toc() << " ms");
}

void solveOdometry()
{
    if (frame_count < window_size)
        return;
    if (solver_flag == NON_LINEAR)
    {
        TicToc t_tri;
        optimization();
    }
}

//todo 滑窗
void SlideWindow() { // NOTE: this function is only for the states and the local map

  {
    if (init_local_map_) {
      int pivot_idx = window_size - opt_window_size;

      Twist<double> transform_lb = transform_lb_.cast<double>();

      Eigen::Vector3d Ps_pivot = Ps_[pivot_idx];
      Eigen::Matrix3d Rs_pivot = Rs_[pivot_idx];

      Quaterniond rot_pivot(Rs_pivot * transform_lb.rot.inverse());
      Eigen::Vector3d pos_pivot = Ps_pivot - rot_pivot * transform_lb.pos;

      Twist<double> transform_pivot = Twist<double>(rot_pivot, pos_pivot);

      PointCloudPtr transformed_cloud_surf_ptr(new PointCloud);
      PointCloudPtr transformed_cloud_corner_ptr(new PointCloud);
      PointCloud filtered_surf_points;
      PointCloud filtered_corner_points;

      int i = pivot_idx + 1; // the index of the next pivot
      Eigen::Vector3d Ps_i = Ps_[i];
      Eigen::Matrix3d Rs_i = Rs_[i];

      Quaterniond rot_li(Rs_i * transform_lb.rot.inverse());
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

//todo 可能会导致失败
bool failureDetection()
{
    if (Bas[window_size].norm() > 2.5)
    {
        ROS_INFO(" big IMU acc bias estimation %f", Bas[window_size].norm());
        return true;
    }
    if (Bgs[window_size].norm() > 1.0)
    {
        ROS_INFO(" big IMU gyr bias estimation %f", Bgs[window_size].norm());
        return true;
    }
    Vector3d tmp_P = Ps[window_size];
    if ((tmp_P - last_P).norm() > 5)
    {
        ROS_INFO(" big translation");
        return true;
    }
    if (abs(tmp_P.z() - last_P.z()) > 1)
    {
        ROS_INFO(" big z translation");
        return true;
    }
    Matrix3d tmp_R = Rs[window_size];
    Matrix3d delta_R = tmp_R.transpose() * last_R;
    Quaterniond delta_Q(delta_R);
    double delta_angle;
    delta_angle = acos(delta_Q.w()) * 2.0 / 3.14 * 180.0;
    if (delta_angle > 50)
    {
        ROS_INFO(" big delta_angle ");
        //return true;
    }
    return false;
}

void clearState()
{
    for (int i = 0; i < window_size + 1; i++)
    {
        Rs[i].setIdentity();
        Ps[i].setZero();
        Vs[i].setZero();
        Bas[i].setZero();
        Bgs[i].setZero();
        dt_buf[i].clear();
        linear_acceleration_buf[i].clear();
        angular_velocity_buf[i].clear();

        if (pre_integrations[i] != nullptr)
            delete pre_integrations[i];
        pre_integrations[i] = nullptr;
    }

    for (auto &it : all_lidar_frame)
    {
        if (it.second.pre_integration != nullptr)
        {
            delete it.second.pre_integration;
            it.second.pre_integration = nullptr;
        }
    }

    solver_flag = INITIAL;
    first_imu = false,
    frame_count = 0;
    key = 0;
    //todo 需要对gtsam相关的内容进行管理

    solver_flag = INITIAL;
    initial_timestamp = 0;
    all_lidar_frame.clear();


    if (tmp_pre_integration != nullptr)
        delete tmp_pre_integration;
    if (last_marginalization_info != nullptr)
        delete last_marginalization_info;

    tmp_pre_integration = nullptr;
    last_marginalization_info = nullptr;

    failure_occur = 0;
    relocalization_info = 0;

    drift_correct_r = Matrix3d::Identity();
    drift_correct_t = Vector3d::Zero();
}

bool RunInitialization() {

  // NOTE: check IMU observibility, adapted from VINS-mono
  {
    PairTimeLaserTransform laser_trans_i, laser_trans_j;
    Vector3d sum_g;

    for (size_t i = 0; i < window_size;
         ++i) {
      laser_trans_j = all_laser_transforms_[i + 1];

      double dt = laser_trans_j.second.pre_integration->sum_dt_;
      Vector3d tmp_g = laser_trans_j.second.pre_integration->delta_v_ / dt;
      sum_g += tmp_g;
    }

    Vector3d aver_g;
    aver_g = sum_g * 1.0 / (window_size);
    double var = 0;

    for (size_t i = 0; i < window_size;
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
  //todo R_WI_ LIO版本的初始化要确认是否正确
  bool init_result
      = ImuInitializer::Initialization(all_laser_transforms_, Vs_, Bas_, Bgs_, g_vec_in_laser, transform_lb_, R_WI_);
//  init_result = false;

//  Q_WI_ = R_WI_;
//  g_vec_ = R_WI_ * Eigen::Vector3d(0.0, 0.0, -1.0) * g_norm_;
//  g_vec_ = Eigen::Vector3d(0.0, 0.0, -1.0) * g_norm_;

  // TODO: update states Ps_
  for (size_t i = 0; i < window_size + 1;
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

//处理完lidar_info消息后做LIO
void ProcessLaserOdom(const Transform &transform_in, const std_msgs::Header &header) {

  ROS_DEBUG(">>>>>>> new laser odom coming <<<<<<<");

  ++laser_odom_recv_count_;

  if (stage_flag_ != INITED
      && laser_odom_recv_count_ % init_window_factor != 0) { /// better for initialization
    return;
  }

  Headers_.push(header);

  // TODO: LaserFrame Object
  // LaserFrame laser_frame(laser, header.stamp.toSec());

  LaserTransform laser_transform(header.stamp.toSec(), transform_in);

  laser_transform.pre_integration = tmp_pre_integration_;
  pre_integrations_.push(tmp_pre_integration_);

  // reset tmp_pre_integration_
  tmp_pre_integration_.reset();
  tmp_pre_integration_ = std::make_shared<IntegrationBase>(IntegrationBase(acc_last_,
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

  //todo 这里往滑窗里放点云
  // TODO: avoid memory allocation?
  if (stage_flag_ != INITED || (!enable_deskew && !cutoff_deskew)) {
    surf_stack_.push(boost::make_shared<PointCloud>(*laser_cloud_surf_stack_downsampled_));
    size_surf_stack_.push(laser_cloud_surf_stack_downsampled_->size());

    corner_stack_.push(boost::make_shared<PointCloud>(*laser_cloud_corner_stack_downsampled_));
    size_corner_stack_.push(laser_cloud_corner_stack_downsampled_->size());
  }

  full_stack_.push(boost::make_shared<PointCloud>(*full_cloud_));

  opt_surf_stack_.push(surf_stack_.last());
  opt_corner_stack_.push(corner_stack_.last());

  //todo ?
  opt_matP_.push(matP_.cast<double>());
  ///< optimization buffers

  if (run_optimization) {
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
        if (cir_buf_count_ == window_size) {
          tic_toc_.Tic();

          if (!imu_factor) {
            init_result = true;
            // TODO: update states Ps_
            for (size_t i = 0; i < window_size + 1;
                 ++i) {
              const Transform &trans_li = all_laser_transforms_[i].second.transform;
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

          DLOG(INFO) << "initialization time: " << tic_toc_.Toc() << " ms";

          if (init_result) {
            stage_flag_ = INITED;
//            SetInitFlag(true);
			imu_inited_ = set_init;

            Q_WI_ = R_WI_;
//            wi_trans_.setRotation(tf::Quaternion{Q_WI_.x(), Q_WI_.y(), Q_WI_.z(), Q_WI_.w()});

            ROS_WARN_STREAM(">>>>>>> IMU initialized <<<<<<<");

//            if (enable_deskew || cutoff_deskew) {
//              ros::ServiceClient client = nh_.serviceClient<std_srvs::SetBool>("/enable_odom");
//              std_srvs::SetBool srv;
//              srv.request.data = 0;
//              if (client.call(srv)) {
//                DLOG(INFO) << "TURN OFF THE ORIGINAL LASER ODOM";
//              } else {
//                LOG(FATAL) << "FAILED TO CALL TURNING OFF THE ORIGINAL LASER ODOM";
//              }
//            }

            for (size_t i = 0; i < window_size + 1;
                 ++i) {
              Twist<double> transform_lb = transform_lb_.cast<double>();

              Quaterniond Rs_li(Rs_[i] * transform_lb.rot.inverse());
              Eigen::Vector3d Ps_li = Ps_[i] - Rs_li * transform_lb.pos;

              Twist<double> trans_li{Rs_li, Ps_li};

              DLOG(INFO) << "TEST trans_li " << i << ": " << trans_li;
              DLOG(INFO) << "TEST all_laser_transforms " << i << ": " << all_laser_transforms_[i].second.transform;
            }

            SolveOptimization();

            SlideWindow();

            for (size_t i = 0; i < window_size + 1;
                 ++i) {
              const Transform &trans_li = all_laser_transforms_[i].second.transform;
              Transform trans_bi = trans_li * transform_lb_;
              DLOG(INFO) << "TEST " << i << ": " << trans_bi.pos.transpose();
            }

            for (size_t i = 0; i < window_size + 1;
                 ++i) {
              Twist<double> transform_lb = transform_lb_.cast<double>();

              Quaterniond Rs_li(Rs_[i] * transform_lb.rot.inverse());
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
        if (opt_point_coeff_map_.size() == opt_window_size + 1) {

//        PublishResults();

        SlideWindow();

//        {
//          int pivot_idx = estimator_config_.window_size - estimator_config_.opt_window_size;
//          local_odom_.header.stamp = Headers_[pivot_idx + 1].stamp;
//          local_odom_.header.seq += 1;
//          Twist<double> transform_lb = transform_lb_.cast<double>();
//          Eigen::Vector3d Ps_pivot = Ps_[pivot_idx];
//          Eigen::Matrix3d Rs_pivot = Rs_[pivot_idx];
//          Quaterniond rot_pivot(Rs_pivot * transform_lb.rot.inverse());
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
//          Quaterniond rot_last(Rs_last * transform_lb.rot.inverse());
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

  {
    transform_sum_.pos.x() = lidar_info.laser_odometry_.pose.pose.position.x;
    transform_sum_.pos.y() = lidar_info.laser_odometry_.pose.pose.position.y;
    transform_sum_.pos.z() = lidar_info.laser_odometry_.pose.pose.position.z;
    transform_sum_.rot.x() = lidar_info.laser_odometry_.pose.pose.orientation.x;
    transform_sum_.rot.y() = lidar_info.laser_odometry_.pose.pose.orientation.y;
    transform_sum_.rot.z() = lidar_info.laser_odometry_.pose.pose.orientation.z;
    transform_sum_.rot.w() = lidar_info.laser_odometry_.pose.pose.orientation.w;
  }

  {
    laser_cloud_corner_last_->clear();
    laser_cloud_surf_last_->clear();
    full_cloud_->clear();

    // 将 ROS 消息转换为 PCL 点云
    pcl::fromROSMsg(lidar_info.surf_cloud_, *laser_cloud_surf_last_);
    pcl::fromROSMsg(lidar_info.edge_cloud_, *laser_cloud_corner_last_);
    //todo 没加full_cloud_
  }

  time_laser_cloud_corner_last_ = lidar_info.laser_odometry_.header.stamp;
  time_laser_cloud_surf_last_ = lidar_info.laser_odometry_.header.stamp;
  time_laser_full_cloud_ = lidar_info.laser_odometry_.header.stamp;
  time_laser_odometry_ = lidar_info.laser_odometry_.header.stamp;

  new_laser_cloud_corner_last_ = true;
  new_laser_cloud_surf_last_ = true;
  new_laser_full_cloud_ = true;
  new_laser_odometry_ = true;

  DLOG(INFO) << "decode lidar_info time: " << tic_toc_decoder.Toc() << " ms";
}

void setParameter()
{
//  // 固定不变 赋值
//    tic = TIC[i];
//    ric = RIC[i];
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

//todo scan to map不用 直接给里程计的值
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
    TransformAssociateToMap();
  }

  // NOTE: the stack points are the last corner or surf poitns
  size_t laser_cloud_corner_last_size = laser_cloud_corner_last_->points.size();
  for (int i = 0; i < laser_cloud_corner_last_size; i++) {
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

//  DLOG(INFO) << "center_before: " << center_cube_i << " " << center_cube_j << " " << center_cube_k;
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


  laser_cloud_valid_idx_.clear();
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
#if 0
    for (int i = 0; i < laser_cloud_corner_stack_ds_size; ++i) {
      PointAssociateToMap(laser_cloud_corner_stack_downsampled_->points[i], point_sel, transform_tobe_mapped_);

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
        laser_cloud_corner_array_[cube_idx]->push_back(point_sel);
      }
    }

    // store down sized surface stack points in corresponding cube clouds
    for (int i = 0; i < laser_cloud_surf_stack_ds_size; ++i) {
      PointAssociateToMap(laser_cloud_surf_stack_downsampled_->points[i], point_sel, transform_tobe_mapped_);

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

    // down size all valid (within field of view) feature cube clouds
    for (int i = 0; i < laser_cloud_valid_size; ++i) {
      size_t index = laser_cloud_valid_idx_[i];

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
#endif

    // publish result
    PublishResults();
  }

  DLOG(INFO) << "mapping: " << tic_toc_.Toc() << " ms";

}

// lio-mapping 版本
void processLidarInfo(const LidarInfo &lidar_info, const std_msgs::Header &header)
{
    //1. process lidar_info
  	LidarInfoHandler(lidar_info);

  	if (stage_flag_ == INITED) {
    	Transform trans_prev(Eigen::Quaterniond(Rs_[window_size - 1]).cast<float>(),
        	                 Ps_[window_size - 1].cast<float>());
    	Transform trans_curr(Eigen::Quaterniond(Rs_.last()).cast<float>(),
        	                 Ps_.last().cast<float>());

    	Transform d_trans = trans_prev.inverse() * trans_curr;
		//todo transform_sum_追踪
    	Transform transform_incre(transform_bef_mapped_.inverse() * transform_sum_.transform());

    	if (imu_factor) {
      	//    // WARNING: or using direct date?
      		transform_tobe_mapped_bef_ = transform_tobe_mapped_ * transform_lb_ * d_trans * transform_lb_.inverse();
      		transform_tobe_mapped_ = transform_tobe_mapped_bef_;
    	} else {
      		TransformAssociateToMap();
      		DLOG(INFO) << ">>>>> transform original tobe <<<<<: " << transform_tobe_mapped_;
    	}
  	}

  	if (stage_flag_ != INITED || !imu_factor) {
    	/// 2. process decoded data
    	Process();
  	} else {
  	}

    //todo process会修改以下内容 除了transform
  	DLOG(INFO) << "laser_cloud_surf_last_[" << header.stamp.toSec() << "]: "
    	        << laser_cloud_surf_last_->size();
  	DLOG(INFO) << "laser_cloud_corner_last_[" << header.stamp.toSec() << "]: "
    	        << laser_cloud_corner_last_->size();

	DLOG(INFO) << endl << "transform_aft_mapped_[" << header.stamp.toSec() << "]: " << transform_aft_mapped_;
  	DLOG(INFO) << "laser_cloud_surf_stack_downsampled_[" << header.stamp.toSec() << "]: "
    	        << laser_cloud_surf_stack_downsampled_->size();
  	DLOG(INFO) << "laser_cloud_corner_stack_downsampled_[" << header.stamp.toSec() << "]: "
    	        << laser_cloud_corner_stack_downsampled_->size();

    //todo 获取当前帧位姿 想直接用前面传过来的位姿
  	Transform transform_to_init_ = transform_aft_mapped_;
  	ProcessLaserOdom(transform_to_init_, header);

// NOTE: will be updated in PointMapping's OptimizeTransformTobeMapped
//  if (stage_flag_ == INITED && !estimator_config_.imu_factor) {
//    TransformUpdate();
//    DLOG(INFO) << ">>>>> transform sum <<<<<: " << transform_sum_;
//  }
}

//todo 这里是IMU积分
void predict(const sensor_msgs::ImuConstPtr &imu_msg)
{
    double t = imu_msg->header.stamp.toSec();
    if (init_imu)
    {
        latest_time = t;
        init_imu = 0;
        return;
    }
    double dt = t - latest_time;
    latest_time = t;

    double dx = imu_msg->linear_acceleration.x;
    double dy = imu_msg->linear_acceleration.y;
    double dz = imu_msg->linear_acceleration.z;
    Eigen::Vector3d linear_acceleration{dx, dy, dz};

    double rx = imu_msg->angular_velocity.x;
    double ry = imu_msg->angular_velocity.y;
    double rz = imu_msg->angular_velocity.z;
    Eigen::Vector3d angular_velocity{rx, ry, rz};

    Eigen::Vector3d un_acc_0 = tmp_Q * (acc_0 - tmp_Ba) - estimator.g;

    Eigen::Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - tmp_Bg;
    tmp_Q = tmp_Q * Utility::deltaQ(un_gyr * dt);

    Eigen::Vector3d un_acc_1 = tmp_Q * (linear_acceleration - tmp_Ba) - estimator.g;

    Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);

    //最新imu帧的位姿
    tmp_P = tmp_P + dt * tmp_V + 0.5 * dt * dt * un_acc;
    tmp_V = tmp_V + dt * un_acc;
	//上一imu的加速度 角速度 用于中值积分
    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;
}

void update()
{
    TicToc t_predict;
    latest_time = current_time;
    tmp_P = estimator.Ps[window_size];
    tmp_Q = estimator.Rs[window_size];
    tmp_V = estimator.Vs[window_size];
    tmp_Ba = estimator.Bas[window_size];
    tmp_Bg = estimator.Bgs[window_size];
    acc_0 = estimator.acc_0;
    gyr_0 = estimator.gyr_0;

    queue<sensor_msgs::ImuConstPtr> tmp_imu_buf = imu_buf;
    for (sensor_msgs::ImuConstPtr tmp_imu_msg; !tmp_imu_buf.empty(); tmp_imu_buf.pop())
        predict(tmp_imu_buf.front());
}

//todo 根据需求修改
void SurfHandler(const sensor_msgs::PointCloud2ConstPtr &_laserSurf)
{
    mBuf.lock();
    surfBuf.push(_laserOdometry);
    mBuf.unlock();
    con.notify_one();
}
//todo 根据需求修改
void EdgeHandler(const sensor_msgs::PointCloud2ConstPtr &_laserEdge)
{
    mBuf.lock();
    edgeBuf.push(_laserOdometry);
    mBuf.unlock();
    con.notify_one();
}

//todo 根据需求修改
void pubPath_lio( void )
{
    std::cout << "mark pubPath_lio" << std::endl;
    // pub odom and path
    nav_msgs::Odometry odomAftPGO;
    nav_msgs::Path pathAftPGO;
    pathAftPGO.header.frame_id = "camera_init";
    //todo 如果很卡再把两个锁分开用
//    mBuf.lock();
    //[0, 9]
    for (int node_idx=0; node_idx < windowSize; node_idx++) // -1 is just delayed visualization (because sometimes mutexed while adding(push_back) a new one)
    {
        // 经过isam回环修正后的位姿
        const Frame& pose_set = frameWindow[node_idx];
        nav_msgs::Odometry odomAftPGOthis;
        odomAftPGOthis.header.frame_id = "camera_init";
        odomAftPGOthis.child_frame_id = "/aft_pgo";
        // 修正前和修正后的位姿的时间是相同的
        odomAftPGOthis.header.stamp = ros::Time().fromSec(frameTimes[node_idx]);
		// 获取位置并赋值
		odomAftPGOthis.pose.pose.position.x = pose_set.pose.translation().x();
		odomAftPGOthis.pose.pose.position.y = pose_set.pose.translation().y();
		odomAftPGOthis.pose.pose.position.z = pose_set.pose.translation().z();

		// 获取姿态并赋值
		gtsam::Quaternion quaternion = pose_set.pose.rotation().toQuaternion();
		odomAftPGOthis.pose.pose.orientation.x = quaternion.x();
		odomAftPGOthis.pose.pose.orientation.y = quaternion.y();
		odomAftPGOthis.pose.pose.orientation.z = quaternion.z();
		odomAftPGOthis.pose.pose.orientation.w = quaternion.w();

        odomAftPGO = odomAftPGOthis;

        geometry_msgs::PoseStamped poseStampAftPGO;
        poseStampAftPGO.header = odomAftPGOthis.header;
        poseStampAftPGO.pose = odomAftPGOthis.pose.pose;

        pathAftPGO.header.stamp = odomAftPGOthis.header.stamp;
        pathAftPGO.header.frame_id = "camera_init";
        pathAftPGO.poses.push_back(poseStampAftPGO);
    }
//    mBuf.unlock();

    //保存轨迹，path_save是文件目录,txt文件提前建好 tum格式 time x y z
//    std::ofstream pose1("/media/ctx/0BE20E8D0BE20E8D/dataset/kitti_dataset/result/loamgba/result_pose_09_30_0027_07_loop.txt", std::ios::app);
//    pose1.setf(std::ios::scientific, std::ios::floatfield);
//    //kitti数据集转换tum格式的数据是18位
//    pose1.precision(9);
//    //第一个激光帧时间 static变量 只赋值一次
//    static double timeStart = odomAftPGO.header.stamp.toSec();
//    auto T1 =ros::Time().fromSec(timeStart) ;
//    // tf::Quaternion quat;
//    // tf::createQuaternionMsgFromRollPitchYaw(double r, double p, double y);//返回四元数
//    pose1<< odomAftPGO.header.stamp -T1<< " "
//        << -odomAftPGO.pose.pose.position.x << " "
//        << -odomAftPGO.pose.pose.position.z << " "
//        << -odomAftPGO.pose.pose.position.y<< " "
//        << odomAftPGO.pose.pose.orientation.x << " "
//        << odomAftPGO.pose.pose.orientation.y << " "
//        << odomAftPGO.pose.pose.orientation.z << " "
//        << odomAftPGO.pose.pose.orientation.w << std::endl;
//    pose1.close();


    pubOdomAftPGO.publish(odomAftPGO); // 滑窗内最后一帧位姿
    pubPathAftPGO.publish(pathAftPGO); // 轨迹

    //cout << "pathAftPGO.poses = " << pathAftPGO.poses.size() << endl;
    globalPath = pathAftPGO;
    if(follow)
    {
        static tf::TransformBroadcaster br;
        tf::Transform transform;
        tf::Quaternion q;
        transform.setOrigin(tf::Vector3(odomAftPGO.pose.pose.position.x, odomAftPGO.pose.pose.position.y, odomAftPGO.pose.pose.position.z));
        q.setW(odomAftPGO.pose.pose.orientation.w);
        q.setX(odomAftPGO.pose.pose.orientation.x);
        q.setY(odomAftPGO.pose.pose.orientation.y);
        q.setZ(odomAftPGO.pose.pose.orientation.z);
        transform.setRotation(q);
        br.sendTransform(tf::StampedTransform(transform, odomAftPGO.header.stamp, "camera_init", "/aft_pgo"));

    }

} // pubPath_lio


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
        //vins中和restart有关 这里用不着 需要确认这个锁的控制范围
//        m_estimator.lock();
		for (auto &measurement : measurements)
        {
            auto lidar_info = measurement.second;
            double dx = 0, dy = 0, dz = 0, rx = 0, ry = 0, rz = 0;
            for (auto &imu_msg : measurement.first)
            {
                double t = imu_msg->header.stamp.toSec();
                double lidar_info_t = lidar_info.odom_msg->header.stamp.toSec();
                if (t <= odom_t)
                {
                  //todo vins中有update()会更新current_time
                    if (current_time < 0)
                        current_time = t;
                    double dt = t - current_time;
                    ROS_ASSERT(dt >= 0);
                    current_time = t;
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
                    double dt_1 = odom_t - current_time;
                    double dt_2 = t - lidar_info_t;
                    current_time = lidar_info_t;
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
            //todo 目前改到这里 要传参数吗？
            processLidarInfo(lidar_info, lidar_info.odom_msg->header);

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

int main(int argc, char **argv)
{
	ros::init(argc, argv, "laserPGO");
	ros::NodeHandle nh;

    //todo 全局变量初始化
    down_size_filter_corner_.setLeafSize(corner_filter_size, corner_filter_size, corner_filter_size);
  	down_size_filter_surf_.setLeafSize(surf_filter_size, surf_filter_size, surf_filter_size);
  	down_size_filter_map_.setLeafSize(map_filter_size, map_filter_size, map_filter_size);

    // --------------------------------- 订阅后端数据 ---------------------------------
//	 ros::Subscriber subCenters = nh.subscribe<sensor_msgs::PointCloud2>("/Center_BA", 100, centerHandler);
	ros::Subscriber subSurf = nh.subscribe<sensor_msgs::PointCloud2>("/ground_BA", 100, SurfHandler);
    ros::Subscriber subEdge = nh.subscribe<sensor_msgs::PointCloud2>("/Edge_BA", 100, EdgeHandler);

    ros::Subscriber subLaserOdometry = nh.subscribe<nav_msgs::Odometry>("/BALM_mapped_to_init", 100, laserOdometryHandler);
//	ros::Subscriber subGPS = nh.subscribe<sensor_msgs::NavSatFix>("/gps/fix", 100, gpsHandler);

    //订阅IMU数据
    ros::Subscriber subImu = nh.subscribe<sensor_msgs::Imu>("imu_raw", 2000, imuHandler, ros::TransportHints().tcpNoDelay());

    // ------------------------------------------------------------------
	pubOdomAftPGO = nh.advertise<nav_msgs::Odometry>("/aft_pgo_odom", 100);
	pubPathAftPGO = nh.advertise<nav_msgs::Path>("/aft_pgo_path", 100);

    //执行后端优化的线程
    std::thread measurement_process{process_lio};
 	ros::spin();

	return 0;
}
