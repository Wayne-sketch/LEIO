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

#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot2.h>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/geometry/Point3.h>

#include "utility/common.h"
#include "utility/tic_toc.h"
#include "utility/integration_base.h"

#include <pcl/io/pcd_io.h>
#include <image_transport/image_transport.h>
#include "utility/Scancontext.h"
#include "lidarFactor.hpp"

using namespace gtsam;

#include "CSF/CSF.h"

bool useCSF;
bool follow;
bool viewMap;
bool useIsam;
bool needToOptimize = false;

//getMeasurements版本用这个
std::condition_variable con;
const int WINDOW_SIZE = 10;
//imuHandler使用
double last_imu_t = 0;
bool init_imu = 1;
bool init_odom = 0;
double latest_time;
int sum_of_wait = 0;
double current_time = -1;
//todo 初始化？
int frame_count;
Eigen::Vector3d tmp_P;
Eigen::Quaterniond tmp_Q;
Eigen::Vector3d tmp_V;
Eigen::Vector3d tmp_Ba;
Eigen::Vector3d tmp_Bg;
//imu中值积分的时候前一刻的imu信息 临时存储变量
Eigen::Vector3d acc_0;
Eigen::Vector3d gyr_0;
std::mutex m_state;
bool first_imu = false;
//imu预积分滑窗
IntegrationBase *pre_integrations[(WINDOW_SIZE + 1)];
IntegrationBase *tmp_pre_integration;
vector<double> dt_buf[(WINDOW_SIZE + 1)];
vector<Vector3d> linear_acceleration_buf[(WINDOW_SIZE + 1)];
vector<Vector3d> angular_velocity_buf[(WINDOW_SIZE + 1)];
    Vector3d Ps[(WINDOW_SIZE + 1)];
    Vector3d Vs[(WINDOW_SIZE + 1)];
    Matrix3d Rs[(WINDOW_SIZE + 1)];
    Vector3d Bas[(WINDOW_SIZE + 1)];
    Vector3d Bgs[(WINDOW_SIZE + 1)];
std_msgs::Header Headers[(WINDOW_SIZE + 1)];
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
//todo 不估计外参 这个永远是固定的 vins中和相机数量一致 这里就一个
Matrix3d ric;
Vector3d tic;
enum SolverFlag
{
    INITIAL,
    NON_LINEAR
};
//todo 可以这样用吗
enum SIZE_PARAMETERIZATION
{
    SIZE_POSE = 7,
    SIZE_SPEEDBIAS = 9,
    SIZE_FEATURE = 1
};
enum StateOrder
{
    O_P = 0,
    O_R = 3,
    O_V = 6,
    O_BA = 9,
    O_BG = 12
};
SolverFlag solver_flag = INITIAL;
double initial_timestamp = 0;
Eigen::Vector3d G{0.0, 0.0, 9.8};
//todo 要给RIC等赋值
int ESTIMATE_EXTRINSIC = 0;
Matrix3d back_R0, last_R, last_R0;
Vector3d back_P0, last_P, last_P0;

//优化器使用的double数组
double para_Pose[WINDOW_SIZE + 1][SIZE_POSE];
double para_SpeedBias[WINDOW_SIZE + 1][SIZE_SPEEDBIAS];
double para_Feature[NUM_OF_F][SIZE_FEATURE];
//外参不用估计
//double para_Ex_Pose[NUM_OF_CAM][SIZE_POSE];
double para_Retrive_Pose[SIZE_POSE];
double para_Td[1][1];
double para_Tr[1][1];
bool failure_occur = 0;
vector<Vector3d> key_poses;

double FilterGroundLeaf;

using std::cout;
using std::endl;

using namespace std;


double keyframeMeterGap;
double keyframeDegGap, keyframeRadGap;
double translationAccumulated = 1000000.0; // large value means must add the first given frame.
double rotaionAccumulated = 1000000.0; // large value means must add the first given frame.

bool isNowKeyFrame = false; 

Pose6D odom_pose_prev {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}; // init 
Pose6D odom_pose_curr {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}; // init pose is zero 

std::queue<nav_msgs::Odometry::ConstPtr> odometryBuf;
nav_msgs::Odometry::ConstPtr lastOdom;
std::queue<sensor_msgs::PointCloud2ConstPtr> surfBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> edgeBuf;

std::queue<sensor_msgs::NavSatFix::ConstPtr> gpsBuf;
std::queue<std::pair<int, int> > scLoopICPBuf;

//消息队列锁
std::mutex mBuf;
std::mutex mKF;

double timeLaserOdometry = 0.0;
double timeCenter= 0.0;
double timeSurf= 0.0;
double timeEdge= 0.0;

// std::vector< pcl::PointCloud<PointType>::Ptr > keyframeLaserClouds; 
// std::vector< std::pair< pcl::PointCloud<PointType>::Ptr, pcl::PointCloud<PointType>::Ptr> > keyframeLaserClouds_;
std::vector< std::tuple<pcl::PointCloud<PointType>::Ptr, pcl::PointCloud<PointType>::Ptr, pcl::PointCloud<PointType>::Ptr> > keyframeLaserClouds_;

//LIO用自定义的数据结构
double frameTime = 0;
std::vector<Frame> frameWindow;  // 存储滑窗内的位姿 速度 零偏
std::vector<double> frameTimes;
// windowSize - 1 个IMU预积分器，[0, windowSize - 2]，储滑窗内相邻帧的IMU预积分 imuMeasurementsWindow[i]存储的是poseWindow[i]到poseWindow[i + 1]的预积分
std::vector<std::shared_ptr<gtsam::PreintegratedImuMeasurements>> imuMeasurementsWindow;
bool systemInitialized = false;
int windowSize = 10;
//key永远指向最新到来的lidar帧
int key = 0;
// ISAM2优化器
//用这个把以前的替换掉
gtsam::ISAM2 optimizer;
gtsam::NonlinearFactorGraph graphFactors;
gtsam::Values graphValues;
//imu因子图优化过程中的状态变量
//在添加IMU预积分因子的时候作为临时存储变量
gtsam::Pose3 prevPose_;
gtsam::Vector3 prevVel_;
gtsam::NavState prevState_;
gtsam::imuBias::ConstantBias prevBias_;
//imu回调函数填入信息
std::queue<sensor_msgs::ImuConstPtr> imu_buf;
std::condition_variable cond_var;
// 定义IMU预积分器 用于IMU预积分器初始化的变量
//为了优化后通过新的零偏更新预积分值 需要使用windowSize个IMU预积分器 并把相邻两帧之间的imu信息保存下来 每次零偏更新的时候重置一下预积分器然后重新积分一次
//processImu把信息对齐保存下来 对齐后的IMU信息 滑窗的时候怎么做更快
std::vector<std::vector<sensor_msgs::Imu>> imuBufAligned;

// IMU参数
float imuAccNoise = 3.9939570888238808e-03;          // 加速度噪声标准差
float imuGyrNoise = 1.5636343949698187e-03;          // 角速度噪声标准差
float imuAccBiasN = 6.4356659353532566e-05;          //
float imuGyrBiasN = 3.5640318696367613e-05;
float imuGravity = 9.80511;           // 重力加速度
float imuRPYWeight = 0.01;
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
// 初始化的imu预积分的噪声协方差
std::shared_ptr<gtsam::PreintegrationParams> initialImuPreintegratorParam = gtsam::PreintegrationParams::MakeSharedU(imuGravity);
gtsam::imuBias::ConstantBias priorImuBias((gtsam::Vector(6) << 0, 0, 0, 0, 0, 0).finished());; // assume zero initial bias
//初始化首个LiDAR位姿先验协方差 完全固定的先验
gtsam::noiseModel::Diagonal::shared_ptr priorPoseNoise;
//初始化速度先验协方差
gtsam::noiseModel::Diagonal::shared_ptr priorVelNoise; //rad, rad, rad, m, m, m
//初始化IMU零偏先验协方差
gtsam::noiseModel::Diagonal::shared_ptr priorBiasNoise;
//
gtsam::Vector noiseModelBetweenBias;
// imu-lidar位姿变换
gtsam::Pose3 imu2Lidar = gtsam::Pose3(gtsam::Rot3(1, 0, 0, 0), gtsam::Point3(-extTrans.x(), -extTrans.y(), -extTrans.z()));
gtsam::Pose3 lidar2Imu = gtsam::Pose3(gtsam::Rot3(1, 0, 0, 0), gtsam::Point3(extTrans.x(), extTrans.y(), extTrans.z()));

//原来的数据结构
std::vector<Pose6D> keyframePoses;
std::vector<Pose6D> keyframePosesUpdated;
std::vector<double> keyframeTimes;
int recentIdxUpdated = 0;
bool gtSAMgraphMade = false;
gtsam::NonlinearFactorGraph gtSAMgraph;
gtsam::Values initialEstimate;
gtsam::ISAM2 *isam;
gtsam::Values isamCurrentEstimate;

noiseModel::Diagonal::shared_ptr priorNoise;
noiseModel::Diagonal::shared_ptr odomNoise;
noiseModel::Base::shared_ptr robustLoopNoise;
noiseModel::Base::shared_ptr robustGPSNoise;
// pcl::VoxelGridLargeScaleLarge
pcl::VoxelGridLargeScale<PointType> downSizeFilterScancontext;
SCManager scManager;
double scDistThres, scMaximumRadius;

pcl::VoxelGridLargeScale<PointType> downSizeFilterICP;
std::mutex mtxICP;
//isam相关操作锁
std::mutex mtxPosegraph;
std::mutex mtxRecentPose;

pcl::PointCloud<PointType>::Ptr laserCloudMapPGO(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr offGroundMap(new pcl::PointCloud<PointType>());

pcl::VoxelGridLargeScale<PointType> downSizeFilterMapPGO;
bool laserCloudMapPGORedraw = true;

//bool useGPS = true;
bool useGPS = false;
sensor_msgs::NavSatFix::ConstPtr currGPS;
bool hasGPSforThisKF = false;
bool gpsOffsetInitialized = false; 
double gpsAltitudeInitOffset = 0.0;
double recentOptimizedX = 0.0;
double recentOptimizedY = 0.0;

ros::Publisher pubMapAftPGO, pubOdomAftPGO, pubPathAftPGO;
ros::Publisher pubLoopScanLocal, pubLoopSubmapLocal;
ros::Publisher pubOdomRepubVerifier;

std::string save_directory;
std::string map_save_directory;

std::string pgKITTIformat, pgScansDirectory, pgSCDsDirectory;
std::string odomKITTIformat;
std::fstream pgG2oSaveStream, pgTimeSaveStream;

std::vector<std::string> edges_str; // used in writeEdge

pcl::KdTreeFLANN<PointType>::Ptr kdtreeCenterloopMap(new pcl::KdTreeFLANN<PointType>());
std::vector<int> pointSearchInd;
std::vector<float> pointSearchSqDis;

nav_msgs::Path globalPath;
// 一秒20次
double vizMapFreq; // 0.1 means run onces every 10s
double processIsamFreq;
double vizPathFreq;
double loopClosureFreq; // can change

std::string padZeros(int val, int num_digits = 6) 
{
  std::ostringstream out;
  out << std::internal << std::setfill('0') << std::setw(num_digits) << val;
  return out.str();
}

std::string getVertexStr(const int _node_idx, const gtsam::Pose3& _Pose)
{
    gtsam::Point3 t = _Pose.translation();
    gtsam::Rot3 R = _Pose.rotation();

    std::string curVertexInfo 
    {
        "VERTEX_SE3:QUAT " + std::to_string(_node_idx) + " "
        + std::to_string(t.x()) + " " + std::to_string(t.y()) + " " + std::to_string(t.z())  + " " 
        + std::to_string(R.toQuaternion().x()) + " " + std::to_string(R.toQuaternion().y()) + " " 
        + std::to_string(R.toQuaternion().z()) + " " + std::to_string(R.toQuaternion().w()) };

    // pgVertexSaveStream << curVertexInfo << std::endl;
    // vertices_str.emplace_back(curVertexInfo);
    return curVertexInfo;
}

void writeEdge(const std::pair<int, int> _node_idx_pair, const gtsam::Pose3& _relPose, std::vector<std::string>& edges_str)
{
    gtsam::Point3 t = _relPose.translation();
    gtsam::Rot3 R = _relPose.rotation();

    std::string curEdgeInfo 
    {
        "EDGE_SE3:QUAT " + std::to_string(_node_idx_pair.first) + " " + std::to_string(_node_idx_pair.second) + " "
        + std::to_string(t.x()) + " " + std::to_string(t.y()) + " " + std::to_string(t.z())  + " "
        + std::to_string(R.toQuaternion().x()) + " " + std::to_string(R.toQuaternion().y()) + " " 
        + std::to_string(R.toQuaternion().z()) + " " + std::to_string(R.toQuaternion().w()) };

    // pgEdgeSaveStream << curEdgeInfo << std::endl;
    edges_str.emplace_back(curEdgeInfo);
}

void saveSCD(std::string fileName, Eigen::MatrixXd matrix, std::string delimiter = " ")
{
    // delimiter: ", " or " " etc.

    int precision = 3; // or Eigen::FullPrecision, but SCD does not require such accruate precisions so 3 is enough.
    const static Eigen::IOFormat the_format(precision, Eigen::DontAlignCols, delimiter, "\n");
 
    std::ofstream file(fileName);
    if (file.is_open())
    {
        file << matrix.format(the_format);
        file.close();
    }
}

gtsam::Pose3 Pose6DtoGTSAMPose3(const Pose6D& p)
{
    return gtsam::Pose3( gtsam::Rot3::RzRyRx(p.roll, p.pitch, p.yaw), gtsam::Point3(p.x, p.y, p.z) );
} // Pose6DtoGTSAMPose3

void saveGTSAMgraphG2oFormat(const gtsam::Values& _estimates)
{
    // save pose graph (runs when programe is closing)
    // cout << "****************************************************" << endl; 
    cout << "Saving the posegraph ..." << endl; // giseop

    pgG2oSaveStream = std::fstream(save_directory + "singlesession_posegraph.g2o", std::fstream::out);

    int pose_idx = 0;
    for(const auto& _pose6d: keyframePoses) {
        gtsam::Pose3 pose = Pose6DtoGTSAMPose3(_pose6d);    
        pgG2oSaveStream << getVertexStr(pose_idx, pose) << endl;
        pose_idx++;
    }
    for(auto& _line: edges_str)
        pgG2oSaveStream << _line << std::endl;

    pgG2oSaveStream.close();
}

void saveOdometryVerticesKITTIformat(std::string _filename)
{
    // ref from gtsam's original code "dataset.cpp"
    std::fstream stream(_filename.c_str(), std::fstream::out);
    for(const auto& _pose6d: keyframePoses) 
    {
        gtsam::Pose3 pose = Pose6DtoGTSAMPose3(_pose6d);
        Point3 t = pose.translation();
        Rot3 R = pose.rotation();

        auto col1 = R.column(1); // Point3
        auto col2 = R.column(2); // Point3
        auto col3 = R.column(3); // Point3

        stream << col1.x() << " " << col2.x() << " " << col3.x() << " " << t.x() << " "
               << col1.y() << " " << col2.y() << " " << col3.y() << " " << t.y() << " "
               << col1.z() << " " << col2.z() << " " << col3.z() << " " << t.z() << std::endl;
    }
}

void saveOptimizedVerticesKITTIformat(gtsam::Values _estimates, std::string _filename)
{
    using namespace gtsam;

    // ref from gtsam's original code "dataset.cpp"
    std::fstream stream(_filename.c_str(), std::fstream::out);

    for(const auto& key_value: _estimates)
    {
        auto p = dynamic_cast<const GenericValue<Pose3>*>(&key_value.value);
        if (!p) continue;
        const Pose3& pose = p->value();
        Point3 t = pose.translation();
        Rot3 R = pose.rotation();

        // 绕x轴顺时针旋转90°，绕y轴顺时针旋转90°
        // r11 r21 r31
        auto col1 = R.column(1); // Point3
        auto col2 = R.column(2); // Point3
        auto col3 = R.column(3); // Point3

        stream << col1.x() << " " << col2.x() << " " << col3.x() << " " << t.x() << " "
               << col1.y() << " " << col2.y() << " " << col3.y() << " " << t.y() << " "
               << col1.z() << " " << col2.z() << " " << col3.z() << " " << t.z() << std::endl;
    }
}

void laserOdometryHandler(const nav_msgs::Odometry::ConstPtr &_laserOdometry)
{
    if (!init_odom)
    {
      //todo 其实可以不用跳
        //skip the first detected feature, which doesn't contain optical flow speed
        init_odom = 1;
        return;
    }
    mBuf.lock();
    odometryBuf.push(_laserOdometry);
    mBuf.unlock();
    con.notify_one();
}

//对齐数据
std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, nav_msgs::Odometry::ConstPtr>>
getMeasurements()
{
    std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, nav_msgs::Odometry::ConstPtr>> measurements;

    while (true)
    {
        if (imu_buf.empty() || odometryBuf.empty())
            return measurements;

        if (!(imu_buf.back()->header.stamp.toSec() > odometryBuf.front()->header.stamp.toSec()))
        {
            //ROS_WARN("wait for imu, only should happen at the beginning");
            sum_of_wait++;
            return measurements;
        }

        if (!(imu_buf.front()->header.stamp.toSec() < odometryBuf.front()->header.stamp.toSec()))
        {
            ROS_WARN("throw odom, only should happen at the beginning");
            odometryBuf.pop();
            continue;
        }
        nav_msgs::Odometry::ConstPtr odom_msg = odometryBuf.front();
        odometryBuf.pop();

        std::vector<sensor_msgs::ImuConstPtr> IMUs;
        while (imu_buf.front()->header.stamp.toSec() < odom_msg->header.stamp.toSec())
        {
            IMUs.emplace_back(imu_buf.front());
            imu_buf.pop();
        }
        IMUs.emplace_back(imu_buf.front());
        if (IMUs.empty())
            ROS_WARN("no imu between two image");
        measurements.emplace_back(IMUs, odom_msg);
    }
    return measurements;
}

//对imu做积分 把最新状态赋值给最新lidar帧
void processIMU(double dt, const Vector3d &linear_acceleration, const Vector3d &angular_velocity)
{
    if (!first_imu)
    {
        first_imu = true;
        acc_0 = linear_acceleration;
        gyr_0 = angular_velocity;
    }

    //todo frame_count指向？
    if (!pre_integrations[frame_count])
    {
        pre_integrations[frame_count] = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};
    }
    if (frame_count != 0)
    {
        pre_integrations[frame_count]->push_back(dt, linear_acceleration, angular_velocity);
        //if(solver_flag != NON_LINEAR)
            tmp_pre_integration->push_back(dt, linear_acceleration, angular_velocity);

        dt_buf[frame_count].push_back(dt);
        linear_acceleration_buf[frame_count].push_back(linear_acceleration);
        angular_velocity_buf[frame_count].push_back(angular_velocity);

        int j = frame_count;
        Vector3d un_acc_0 = Rs[j] * (acc_0 - Bas[j]) - g;
        Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - Bgs[j];
        Rs[j] *= Utility::deltaQ(un_gyr * dt).toRotationMatrix();
        Vector3d un_acc_1 = Rs[j] * (linear_acceleration - Bas[j]) - g;
        Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
        Ps[j] += dt * Vs[j] + 0.5 * dt * dt * un_acc;
        Vs[j] += dt * un_acc;
    }
    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;
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

    for (int i = 0; i <= WINDOW_SIZE; i++)
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
            tmp_b.block<3, 1>(0, 0) = frame_j->second.pre_integration->delta_p + frame_i->second.R.transpose() * frame_j->second.R * TIC[0] - TIC[0] - frame_i->second.R.transpose() * dt * dt / 2 * g0;

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
        tmp_b.block<3, 1>(0, 0) = frame_j->second.pre_integration->delta_p + frame_i->second.R.transpose() * frame_j->second.R * TIC[0] - TIC[0];
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
    bool result = VisualIMUAlignment(all_image_frame, Bgs, g, x);
    if(!result)
    {
        ROS_DEBUG("solve g failed!");
        return false;
    }

    // change state
    for (int i = 0; i <= frame_count; i++)
    {
        Matrix3d Ri = all_image_frame[Headers[i].stamp.toSec()].R;
        Vector3d Pi = all_image_frame[Headers[i].stamp.toSec()].T;
        Ps[i] = Pi;
        Rs[i] = Ri;
        all_image_frame[Headers[i].stamp.toSec()].is_key_frame = true;
    }

    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < dep.size(); i++)
        dep[i] = -1;
    f_manager.clearDepth(dep);

    //triangulat on cam pose , no tic
    Vector3d TIC_TMP[NUM_OF_CAM];
    for(int i = 0; i < NUM_OF_CAM; i++)
        TIC_TMP[i].setZero();
    ric[0] = RIC[0];
    f_manager.setRic(ric);
    f_manager.triangulate(Ps, &(TIC_TMP[0]), &(RIC[0]));

    double s = (x.tail<1>())(0);
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]);
    }
    for (int i = frame_count; i >= 0; i--)
        Ps[i] = s * Ps[i] - Rs[i] * TIC[0] - (s * Ps[0] - Rs[0] * TIC[0]);
    int kv = -1;
    map<double, ImageFrame>::iterator frame_i;
    for (frame_i = all_image_frame.begin(); frame_i != all_image_frame.end(); frame_i++)
    {
        if(frame_i->second.is_key_frame)
        {
            kv++;
            Vs[kv] = frame_i->second.R * x.segment<3>(kv * 3);
        }
    }
    for (auto &it_per_id : f_manager.feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
        it_per_id.estimated_depth *= s;
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

//todo 从滑窗vector中取信息 转化成gtsam所需的格式 比如gtsam::Pose3等
void vector2gtsam() {
    for (int i = 0; i <= WINDOW_SIZE; i++)
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

//todo 取出gtsam优化结果
void gtsam2vector()
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

    for (int i = 0; i <= WINDOW_SIZE; i++)
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

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = Vector3d(para_Ex_Pose[i][0],
                          para_Ex_Pose[i][1],
                          para_Ex_Pose[i][2]);
        ric[i] = Quaterniond(para_Ex_Pose[i][6],
                             para_Ex_Pose[i][3],
                             para_Ex_Pose[i][4],
                             para_Ex_Pose[i][5]).toRotationMatrix();
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

//todo 优化核心部分 改成gtsam求解
void optimization()
{
	//todo vins每次新建一个ceres::Problem 改成每次新建一个isam2优化器 还有配套的gtsam::Graph gtsam::Values
	gtsam::ISAM2 optimizer;
	gtsam::NonlinearFactorGraph graphFactors;
	gtsam::Values graphValues;

    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Pose[i], SIZE_POSE, local_parameterization);
        problem.AddParameterBlock(para_SpeedBias[i], SIZE_SPEEDBIAS);
    }

    TicToc t_whole, t_prepare;
    //ins中这里把滑窗内数据取出来 放入优化器 vector2double();
    //todo 改成从滑窗内取数据 放入gtsam优化器 isam2
	vector2gtsam();

	//todo边缘化的处理

    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        int j = i + 1;
        if (pre_integrations[j]->sum_dt > 10.0)
            continue;
        IMUFactor* imu_factor = new IMUFactor(pre_integrations[j]);
        problem.AddResidualBlock(imu_factor, NULL, para_Pose[i], para_SpeedBias[i], para_Pose[j], para_SpeedBias[j]);
    }

	//todo添加帧间相对位姿约束

    ROS_DEBUG("prepare for gtsam: %f", t_prepare.toc());

	//todo 回环约束
	//todo 优化求解


    gtsam2vector();

    //todo 边缘化操作 LIO认为每一帧lidar帧都是关键帧 所以永远是margin_old
    TicToc t_whole_marginalization;
    if (1) //mrgin_old
    {
		//todo 边缘化 优化一次

        //todo 处理滑窗逻辑
        std::unordered_map<long, double *> addr_shift;
        for (int i = 1; i <= WINDOW_SIZE; i++)
        {
            addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
            addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
        }
        vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);

        if (last_marginalization_info)
            delete last_marginalization_info;
        last_marginalization_info = marginalization_info;
        last_marginalization_parameter_blocks = parameter_blocks;

    }
    ROS_DEBUG("whole marginalization costs: %f", t_whole_marginalization.toc());

    ROS_DEBUG("whole time for gtsam: %f", t_whole.toc());
}

void solveOdometry()
{
    if (frame_count < WINDOW_SIZE)
        return;
    if (solver_flag == NON_LINEAR)
    {
        TicToc t_tri;
        optimization();
    }
}

//todo 滑窗
void slideWindow()
{
    TicToc t_margin;
    if (1) //margin_old
    {
        double t_0 = Headers[0].stamp.toSec();
        back_R0 = Rs[0];
        back_P0 = Ps[0];
        if (frame_count == WINDOW_SIZE)
        {
            for (int i = 0; i < WINDOW_SIZE; i++)
            {
                Rs[i].swap(Rs[i + 1]);

                std::swap(pre_integrations[i], pre_integrations[i + 1]);

                dt_buf[i].swap(dt_buf[i + 1]);
                linear_acceleration_buf[i].swap(linear_acceleration_buf[i + 1]);
                angular_velocity_buf[i].swap(angular_velocity_buf[i + 1]);

                Headers[i] = Headers[i + 1];
                Ps[i].swap(Ps[i + 1]);
                Vs[i].swap(Vs[i + 1]);
                Bas[i].swap(Bas[i + 1]);
                Bgs[i].swap(Bgs[i + 1]);
            }
            Headers[WINDOW_SIZE] = Headers[WINDOW_SIZE - 1];
            Ps[WINDOW_SIZE] = Ps[WINDOW_SIZE - 1];
            Vs[WINDOW_SIZE] = Vs[WINDOW_SIZE - 1];
            Rs[WINDOW_SIZE] = Rs[WINDOW_SIZE - 1];
            Bas[WINDOW_SIZE] = Bas[WINDOW_SIZE - 1];
            Bgs[WINDOW_SIZE] = Bgs[WINDOW_SIZE - 1];

            delete pre_integrations[WINDOW_SIZE];
            pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};

            dt_buf[WINDOW_SIZE].clear();
            linear_acceleration_buf[WINDOW_SIZE].clear();
            angular_velocity_buf[WINDOW_SIZE].clear();

            if (true || solver_flag == INITIAL)
            {
                map<double, LidarFrame>::iterator it_0;
                it_0 = all_lidar_frame.find(t_0);
                delete it_0->second.pre_integration;
                it_0->second.pre_integration = nullptr;

                for (map<double, LidarFrame>::iterator it = all_lidar_frame.begin(); it != it_0; ++it)
                {
                    if (it->second.pre_integration)
                        delete it->second.pre_integration;
                    it->second.pre_integration = NULL;
                }

                all_lidar_frame.erase(all_lidar_frame.begin(), it_0);
                all_lidar_frame.erase(t_0);

            }
//            slideWindowOld(); //vins中和视觉特征管理有关 不要了
        }
    }
}

//todo 可能会导致失败
bool failureDetection()
{
    if (Bas[WINDOW_SIZE].norm() > 2.5)
    {
        ROS_INFO(" big IMU acc bias estimation %f", Bas[WINDOW_SIZE].norm());
        return true;
    }
    if (Bgs[WINDOW_SIZE].norm() > 1.0)
    {
        ROS_INFO(" big IMU gyr bias estimation %f", Bgs[WINDOW_SIZE].norm());
        return true;
    }
    Vector3d tmp_P = Ps[WINDOW_SIZE];
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
    Matrix3d tmp_R = Rs[WINDOW_SIZE];
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
    for (int i = 0; i < WINDOW_SIZE + 1; i++)
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

    //固定外参 不变
//    tic[i] = Vector3d::Zero();
//    ric[i] = Matrix3d::Identity();

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

void setParameter()
{
  //todo 固定不变 赋值
    tic = TIC[i];
    ric = RIC[i];
}

//处理laserOdom lio优化
void processOdom(const nav_msgs::Odometry::ConstPtr &odom_msg, const std_msgs::Header &header)
{
    ROS_DEBUG("new odom coming ------------------------------------------");
	//todo 暂时不涉及关键帧问题 每一帧都用
    Headers[frame_count] = header;

    //new 这里做了改动 vins中先添加frame 然后初始化阶段赋值，这里直接给位姿
    //todo 零偏怎么给 initialStructure会估计零偏
    LidarFrame lidarframe(header.stamp.toSec());
    odometry_msg->pose.pose.position.x
        odometry_msg->pose.pose.orientation.w
    //提取平移部分
    lidarframe.T.x() = odometry_msg->pose.pose.position.x;
    lidarframe.T.y() = odometry_msg->pose.pose.position.y;
    lidarframe.T.z() = odometry_msg->pose.pose.position.z;
    // 提取旋转部分 (四元数 -> 旋转矩阵)
    Eigen::Quaterniond quaternion(
        odometry_msg->pose.pose.orientation.w,
        odometry_msg->pose.pose.orientation.x,
        odometry_msg->pose.pose.orientation.y,
        odometry_msg->pose.pose.orientation.z
    );
    lidarframe.R = quaternion.toRotationMatrix();
    lidarframe.pre_integration = tmp_pre_integration;
    all_lidar_frame.insert(make_pair(header.stamp.toSec(), lidarframe));
    //这里把tmp_pre_integration用了，重新开始预积分
    tmp_pre_integration = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};

    //todo 不估计外参 注意后面是否用了ric RIC ESTIMATE_EXTRINSIC=0
    if (solver_flag == INITIAL)
    {
        if (frame_count == WINDOW_SIZE)
        {
            bool result = false;
            if( ESTIMATE_EXTRINSIC != 2 && (header.stamp.toSec() - initial_timestamp) > 0.1)
            {
               result = initialStructure();
               initial_timestamp = header.stamp.toSec();
            }
            if(result)
            {
                solver_flag = NON_LINEAR;
                //todo 改到这里
                solveOdometry();
                slideWindow();
                ROS_INFO("Initialization finish!");
                last_R = Rs[WINDOW_SIZE];
                last_P = Ps[WINDOW_SIZE];
                last_R0 = Rs[0];
                last_P0 = Ps[0];

            }
            else
                slideWindow();
        }
        else
            frame_count++;
    }
    else
    {
        TicToc t_solve;
        solveOdometry();
        ROS_DEBUG("solver costs: %fms", t_solve.toc());

        if (failureDetection())
        {
            ROS_WARN("failure detection!");
            failure_occur = 1;
            clearState();
            setParameter();
            ROS_WARN("system reboot!");
            return;
        }

        TicToc t_margin;
        slideWindow();
        ROS_DEBUG("marginalization costs: %fms", t_margin.toc());
        // prepare output of VINS
        key_poses.clear();
        for (int i = 0; i <= WINDOW_SIZE; i++)
            key_poses.push_back(Ps[i]);

        last_R = Rs[WINDOW_SIZE];
        last_P = Ps[WINDOW_SIZE];
        last_R0 = Rs[0];
        last_P0 = Ps[0];
    }
}

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

    tmp_P = tmp_P + dt * tmp_V + 0.5 * dt * dt * un_acc;
    tmp_V = tmp_V + dt * un_acc;

    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;
}

void update()
{
    TicToc t_predict;
    latest_time = current_time;
    tmp_P = estimator.Ps[WINDOW_SIZE];
    tmp_Q = estimator.Rs[WINDOW_SIZE];
    tmp_V = estimator.Vs[WINDOW_SIZE];
    tmp_Ba = estimator.Bas[WINDOW_SIZE];
    tmp_Bg = estimator.Bgs[WINDOW_SIZE];
    acc_0 = estimator.acc_0;
    gyr_0 = estimator.gyr_0;

    queue<sensor_msgs::ImuConstPtr> tmp_imu_buf = imu_buf;
    for (sensor_msgs::ImuConstPtr tmp_imu_msg; !tmp_imu_buf.empty(); tmp_imu_buf.pop())
        predict(tmp_imu_buf.front());
}

// 尾部压入消息
// 后端位姿
//todo 这里单独写成一个线程 process_lio
void process_lio()
{
	//todo 下面的要加进来
	while (true) {
		std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, nav_msgs::Odometry::ConstPtr>> measurements;
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
            auto odom_msg = measurement.second;
            double dx = 0, dy = 0, dz = 0, rx = 0, ry = 0, rz = 0;
            for (auto &imu_msg : measurement.first)
            {
                double t = imu_msg->header.stamp.toSec();
                double odom_t = odom_msg->header.stamp.toSec();
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
                    processIMU(dt, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));
                    //printf("imu: dt:%f a: %f %f %f w: %f %f %f\n",dt, dx, dy, dz, rx, ry, rz);

                }
                else
                {
                    double dt_1 = odom_t - current_time;
                    double dt_2 = t - img_t;
                    current_time = img_t;
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
                    processIMU(dt_1, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));
                    //printf("dimu: dt:%f a: %f %f %f w: %f %f %f\n",dt_1, dx, dy, dz, rx, ry, rz);
                }
            }

            TicToc t_s;
            //todo 目前改到这里 要传参数吗？
            processOdom(odom_msg, odom_msg->header);

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
        mBuf.lock();
        m_state.lock();
        if (solver_flag == NON_LINEAR)
            update();
        m_state.unlock();
        mBuf.unlock();
	}

        //todo 下面都不要了 但是要复用一下gtsam的内容
    std::lock_guard<std::mutex> lock(mBuf);
    odometryBuf.push(_laserOdometry);
    // 确保 imuBuf 中有足够数据覆盖两帧 odometry 的时间范围
    if (imuBuf.empty() || imuBuf.front()->header.stamp.toSec() > currentOdomTime) {
        // IMU 数据太晚 丢掉一帧odom
        odometryBuf.pop();
        return;
    }

    // 提取 IMU 数据，确保覆盖两帧 odometry 的时间间隔
    std::vector<sensor_msgs::Imu::ConstPtr> alignedImuData;
    while (!imuBuf.empty() && imuBuf.front()->header.stamp.toSec() < currentOdomTime) {
        alignedImuData.push_back(imuBuf.front());
        imuBuf.pop_front();
    }

    if (imuBuf.back().header.stamp.toSec() < odometryBuf.front()->header.stamp.toSec()) {
    	return ;
    }

    std::vector<sensor_msgs::Imu> imuDataBetweenFrames;
    // 对齐两个LiDAR帧之间的IMU数据 把imuBuf分到imuBufAligned
    auto currOdom = odometryBuf.front();
    frameTime = currOdom -> header.stamp.toSec();
    //第一帧处理
    if (lastOdom == nullptr) {
        //把当前帧存下来
        Frame firstFrame;
        // 滑窗内的位姿存储
        gtsam::Pose3 lidarPose = gtsam::Pose3(gtsam::Rot3::Quaternion(currOdom->pose.pose.orientation.w,
            	                                 					currOdom->pose.pose.orientation.x,
                	                             					currOdom->pose.pose.orientation.y,
                    	                         					currOdom->pose.pose.orientation.z),
                        	    				gtsam::Point3(currOdom->pose.pose.position.x,
                            	       					currOdom->pose.pose.position.y,
                                	   					currOdom->pose.pose.position.z));
        firstFrame.pose = lidarPose;
        //第一帧速度和零偏都给0 后续的利用IMU预积分器推导
        firstFrame.velocity = gtsam::Vector3(0, 0, 0);
        firstFrame.bias = gtsam::imuBias::ConstantBias();
        frameWindow.push_back(firstFrame);
        frameTimes.push_back(frameTime);
        //等待下一个激光帧
        lastOdom = currOdom;
        odometryBuf.pop();
            //todo 第一lidar帧处理时 把前面的imuBuf删掉
        while (!imuBuf.empty()) {
			if (imuBuf.front().header.stamp.toSec() < frameTime) {
            	imuBuf.pop();
			}
        }
            //这里不用key++，key=0指向第一帧
//            continue;
//              std::cout << "frame放进来了" <<std::endl;
        return ;
    }
        //非第一帧
        //IMU预积分
        //todo 有时候startTime和endTime是相同的
        double startTime = lastOdom->header.stamp.toSec();
        double endTime = currOdom->header.stamp.toSec();
        //todo 这里能解决问题 还没找到问题原因
        if (startTime == endTime) {
   			odometryBuf.pop();
      		lastOdom = currOdom;
            cond_var.notify_all();
    		return;
		}
        //todo 这里需要处理首尾部分的IMU数据，和lidar对齐，参考LIO-SAM
//        std::cout << "imuBuf.size(): " << imuBuf.size() << std::endl;
        while (!imuBuf.empty()) {
            auto imuData = imuBuf.front();
            double imuTime = imuData.header.stamp.toSec();
            //这里进来了
//            std::cout << "进来了" << std::endl;
//            std::cout << std::fixed << std::setprecision(10);  // 设置精度为 10 位小数
//            std::cout << "imuTime: " << imuTime << std::endl;
//            std:;cout << "startTime: " << startTime << endl;
//              std::cout << "endTime: " << endTime << endl;

            if (imuTime >= startTime && imuTime <= endTime) {
//              	std::cout << "放入imu信息"	<< std::endl;
                imuDataBetweenFrames.push_back(imuData);
//                std::cout << "inside imuDataBetweenFrames.size(): " << imuDataBetweenFrames.size() << std::endl;
                imuBuf.pop();
            } else if (imuTime > endTime) {
                break;
            } else {
              //todo 这里是否有可能填入不完全
              imuBuf.pop();
            }
        }

//		std::cout << "imuDataBetweenFrames.size(): " << imuDataBetweenFrames.size() << std::endl;
        if (!imuDataBetweenFrames.empty()) {
          // 将对齐后的IMU数据存入imuBufAligned
          imuBufAligned.push_back(imuDataBetweenFrames);
          // 创建预积分器
          auto preintegrator = std::make_shared<gtsam::PreintegratedImuMeasurements>(initialImuPreintegratorParam, priorImuBias);
          double lastImuTime = -1;
          // 进行预积分
          for (const auto& imuData : imuDataBetweenFrames) {
            double imuTime = imuData.header.stamp.toSec();
            // 提取IMU加速度和角速度
            // 假设IMU数据中包含必要的信息进行预积分
            gtsam::Vector3 accel(imuData.linear_acceleration.x,
                                 imuData.linear_acceleration.y,
                                 imuData.linear_acceleration.z);
            gtsam::Vector3 gyro(imuData.angular_velocity.x,
                                imuData.angular_velocity.y,
                                imuData.angular_velocity.z);
            double dt = (lastImuTime < 0) ? (1.0 / 500.0) : (imuTime - lastImuTime);
            // 将IMU数据添加到预积分器
            preintegrator->integrateMeasurement(accel, gyro, dt);
            lastImuTime = imuTime;
          }
          // 将预积分器存入imuMeasurementsWindow
          imuMeasurementsWindow.push_back(preintegrator);
        }

       	if (imuDataBetweenFrames.empty()) {
        	cond_var.notify_all();
			return ;
		}
        Frame thisFrame;
        // 滑窗内的位姿存储
        gtsam::Pose3 lidarPose = gtsam::Pose3(gtsam::Rot3::Quaternion(currOdom->pose.pose.orientation.w,
                                             currOdom->pose.pose.orientation.x,
                                             currOdom->pose.pose.orientation.y,
                                             currOdom->pose.pose.orientation.z),
                            Point3(currOdom->pose.pose.position.x,
                                   currOdom->pose.pose.position.y,
                                   currOdom->pose.pose.position.z));
        thisFrame.pose = lidarPose;
        //todo 应该给多少 这里随便给，在process_lio中会利用IMU预积分器推导 check一下
        thisFrame.velocity = gtsam::Vector3(0, 0, 0);
        thisFrame.bias = gtsam::imuBias::ConstantBias();
        frameWindow.push_back(thisFrame);
        frameTimes.push_back(frameTime);
//        std::cout << "frameWindow.size(): " << frameWindow.size() << std::endl;
        //至此 当前帧 上一帧到当前帧的IMU预积分都放到滑窗中了 维护一下滑窗大小 windowSize(10) + 最新一帧
        // 如果滑窗内的数据超过了指定大小，移除最旧的一帧
        if (frameWindow.size() > windowSize + 1 && frameTimes.size() > windowSize + 1 && imuMeasurementsWindow.size() > windowSize && imuBufAligned.size() > windowSize) {
            frameWindow.erase(frameWindow.begin());
            frameTimes.erase(frameTimes.begin());
            imuMeasurementsWindow.erase(imuMeasurementsWindow.begin());
            imuBufAligned.erase(imuBufAligned.begin());
//            slideWindow();
        }
        //上一帧更新为当前帧
        lastOdom = currOdom;
        //已经把lidar帧放进来了，再pop
        odometryBuf.pop();
        //指向最新帧的索引
        key++;
        needToOptimize = true;
        cond_var.notify_all();

        std::unique_lock<std::mutex> lock(mBuf);
//        std::cout << "process_lio mark 1" << std::endl;
        //滑窗满了才做优化 算上最新帧共11帧
//        std::cout << "frameWindow.size():" << frameWindow.size() << std::endl;
        cond_var.wait(lock, [] { return frameWindow.size() == windowSize + 1; });
        if (needToOptimize == false) {
          std::cout << "要跳过 此时key=" << key <<std::endl;
          cond_var.notify_all();
          continue;
        }
//        std::cout << "process_lio mark 0" << std::endl;
        //获取当前激光帧位姿
        float p_x;
        float p_y;
        float p_z;
        float r_x;
        float r_y;
        float r_z;
        float r_w;
//        std::cout << "process_lio mark 1" << std::endl;
        //LIO系统初始化
        if (systemInitialized == false) {
          //重置isam2优化器，清空大因子图
          resetOPtimization();
          //todo IMU信息对齐是否采用LIO-SAM模式
          //初始化第一帧作为先验，后续每帧不再是先验
          //添加里程计位姿先验因子 把第一帧的lidar位姿作为固定先验 以lidar位姿为核心
          p_x = frameWindow[0].pose.translation().x();
          p_y = frameWindow[0].pose.translation().y();
          p_z = frameWindow[0].pose.translation().z();
          r_x = frameWindow[0].pose.rotation().toQuaternion().x();
          r_y = frameWindow[0].pose.rotation().toQuaternion().y();
          r_z = frameWindow[0].pose.rotation().toQuaternion().z();
          r_w = frameWindow[0].pose.rotation().toQuaternion().w();

          gtsam::Pose3 lidarPose0 = gtsam::Pose3(gtsam::Rot3::Quaternion(r_w, r_x, r_y, r_z), gtsam::Point3(p_x, p_y, p_z));
          gtsam::PriorFactor<gtsam::Pose3> priorPose(gtsam::Symbol('x', 0), lidarPose0, priorPoseNoise);
          graphFactors.add(priorPose);
          //添加速度先验因子 初始化速度定为0
          gtsam::PriorFactor<gtsam::Vector3> priorVel(gtsam::Symbol('v', 0), gtsam::Vector3(0, 0, 0), priorVelNoise);
          graphFactors.add(priorVel);
          //添加imu偏置先验因子
          gtsam::PriorFactor<gtsam::imuBias::ConstantBias> priorBias(gtsam::Symbol('b', 0), gtsam::imuBias::ConstantBias(), priorBiasNoise);
          graphFactors.add(priorBias);
          //变量节点赋初值
          std::cout << "初始化要添加x v b" << 0 << std::endl;
          graphValues.insert(gtsam::Symbol('x', 0), lidarPose0);
          graphValues.insert(gtsam::Symbol('v', 0), gtsam::Vector3(0, 0, 0));
          graphValues.insert(gtsam::Symbol('b', 0), gtsam::imuBias::ConstantBias());

          //把滑窗内剩余帧也加入到因子图中 [1 - 10]
          for (int i = 1; i <= windowSize; i++) {
                    std::cout << "process_lio 初始化 遍历" << i << std::endl;
            //添加相对位姿
            p_x = frameWindow[i - 1].pose.translation().x();
            p_y = frameWindow[i - 1].pose.translation().y();
            p_z = frameWindow[i - 1].pose.translation().z();
            r_x = frameWindow[i - 1].pose.rotation().toQuaternion().x();
            r_y = frameWindow[i - 1].pose.rotation().toQuaternion().y();
            r_z = frameWindow[i - 1].pose.rotation().toQuaternion().z();
            r_w = frameWindow[i - 1].pose.rotation().toQuaternion().w();
            gtsam::Pose3 lidarPose_prev = gtsam::Pose3(gtsam::Rot3::Quaternion(r_w, r_x, r_y, r_z), gtsam::Point3(p_x, p_y, p_z));
            std::cout << "process_lio 遍历 mark 1" << std::endl;

            p_x = frameWindow[i].pose.translation().x();
            p_y = frameWindow[i].pose.translation().y();
            p_z = frameWindow[i].pose.translation().z();
            r_x = frameWindow[i].pose.rotation().toQuaternion().x();
            r_y = frameWindow[i].pose.rotation().toQuaternion().y();
            r_z = frameWindow[i].pose.rotation().toQuaternion().z();
            r_w = frameWindow[i].pose.rotation().toQuaternion().w();
            gtsam::Pose3 lidarPose_curr = gtsam::Pose3(gtsam::Rot3::Quaternion(r_w, r_x, r_y, r_z), gtsam::Point3(p_x, p_y, p_z));
            gtsam::Pose3 relPose = lidarPose_prev.between(lidarPose_curr);
            std::cout << "process_lio 遍历 mark 2" << std::endl;
            // 添加后端里程计因子，当前帧和上一帧的帧间约束
            graphFactors.add(gtsam::BetweenFactor<gtsam::Pose3>(gtsam::Symbol('x', i - 1), gtsam::Symbol('x', i), relPose, odomNoise));
//            graphValues.insert(gtsam::Symbol('x', i), lidarPose_curr);
            std::cout << "process_lio 遍历 mark 3" << std::endl;

//           	if (imuMeasurementsWindow[i - 1] == nullptr) std::cout << "nullptr!" <<std::endl;
            //todo 这里有问题 大概率和imuMeasurements有关
            std::cout << "imuMeasurementsWindow.size():" << imuMeasurementsWindow.size() << std::endl;
            //添加IMU预积分因子
            //上一帧位姿，上一帧速度，下一帧位姿，下一帧速度，上一帧帧偏置，预积分量
            gtsam::ImuFactor imu_factor(gtsam::Symbol('x', i - 1), gtsam::Symbol('v', i - 1), gtsam::Symbol('x', i), gtsam::Symbol('v', i), gtsam::Symbol('b', i - 1), (*imuMeasurementsWindow[i - 1]));
            graphFactors.add(imu_factor);
                        std::cout << "process_lio 遍历 mark 4" << std::endl;

            //添加imu偏置因子，上一帧偏置，下一帧偏置，观测值为0（零偏尽量不动），噪声协方差（和IMU预积分器的积分时间相关）；deltaTij()是积分段的时间
            //噪声处理一下
            graphFactors.add(gtsam::BetweenFactor<gtsam::imuBias::ConstantBias>(gtsam::Symbol('b', i - 1), gtsam::Symbol('b', i), gtsam::imuBias::ConstantBias(),
                         gtsam::noiseModel::Diagonal::Sigmas(sqrt(imuMeasurementsWindow[i - 1]->deltaTij()) * noiseModelBetweenBias)));
                        std::cout << "process_lio 遍历 mark 5" << std::endl;

            //用前一帧的状态、偏置，施加imu预计分量，得到当前帧的状态
            // 提取前一帧位姿、速度
            prevPose_  = frameWindow[i - 1].pose;
            prevVel_   = frameWindow[i - 1].velocity;
            // 获取前一帧imu偏置
            prevBias_ = frameWindow[i - 1].bias;
            // 获取前一帧状态
            prevState_ = gtsam::NavState(prevPose_, prevVel_);
            gtsam::NavState propState_ = imuMeasurementsWindow[i - 1]->predict(prevState_, prevBias_);
            // 变量节点赋初值
//            if (!graphValues.exists(gtsam::Symbol('x', i))) {
                      std::cout << "初始化要添加x v b" << i << std::endl;
    			graphValues.insert(gtsam::Symbol('x', i), lidarPose_curr);
//			}
//            if (!graphValues.exists(gtsam::Symbol('v', i))) {
    			graphValues.insert(gtsam::Symbol('v', i), propState_.v());
//			}
            //todo 这里要不要用propState_的bias
//            if (!graphValues.exists(gtsam::Symbol('b', i))) {
    			graphValues.insert(gtsam::Symbol('b', i), prevBias_);
//			}
            //todo 及时更新每帧的速度和零偏 位姿还没有更新
            frameWindow[i].velocity = propState_.v();
                        std::cout << "process_lio 遍历 mark 6" << std::endl;

            //零偏不用更新，第一次滑窗优化初始值都是0
//            frameWindow[i].bias = propState_.b();
          }
          systemInitialized = true;
//          slideWindow();
            std::cout << "process_lio 遍历 mark 7" << std::endl;
        }//end if
        else {
            //todo 先不考虑100帧重置isam优化器
            //创建因子图
            graphFactors.resize(0);
            graphValues.clear();
            //先把滑窗内第一帧加到value中 但不作为先验约束
            p_x = frameWindow[0].pose.translation().x();
            p_y = frameWindow[0].pose.translation().y();
            p_z = frameWindow[0].pose.translation().z();
            r_x = frameWindow[0].pose.rotation().toQuaternion().x();
            r_y = frameWindow[0].pose.rotation().toQuaternion().y();
            r_z = frameWindow[0].pose.rotation().toQuaternion().z();
            r_w = frameWindow[0].pose.rotation().toQuaternion().w();
            gtsam::Pose3 lidarPose0 = gtsam::Pose3(gtsam::Rot3::Quaternion(r_w, r_x, r_y, r_z), gtsam::Point3(p_x, p_y, p_z));
            //todo 可能需要作为先验约束
//            std::cout << "优化阶段要添加x" << key - windowSize + 1 << std::endl;
//            graphValues.insert(gtsam::Symbol('x', key - windowSize + 1), lidarPose0);
            //todo 需要添加零偏和速度

            //添加相对位姿因子
            //todo 这里确认一下gtsam维护因子图的索引机制 参考LIO-SAM 每100帧重启一次gtsam优化器，前面的信息可以像LIO-SAM一样作为新的先验，也可以边缘化一下，边缘化是最正确合理的方式 gtsam或者自动边缘化，或者计算边缘化协方差后手动删除因子
            //这里的索引需要倒推，确认滑窗大小的一致性，算上最新帧有10帧，windowSize=10
            //把滑窗内所有帧加入到因子图中
            for (int i = 1; i <= windowSize; i++) {
              //添加相对位姿
              p_x = frameWindow[i - 1].pose.translation().x();
              p_y = frameWindow[i - 1].pose.translation().y();
              p_z = frameWindow[i - 1].pose.translation().z();
              r_x = frameWindow[i - 1].pose.rotation().toQuaternion().x();
              r_y = frameWindow[i - 1].pose.rotation().toQuaternion().y();
              r_z = frameWindow[i - 1].pose.rotation().toQuaternion().z();
              r_w = frameWindow[i - 1].pose.rotation().toQuaternion().w();
              gtsam::Pose3 lidarPose_prev = gtsam::Pose3(gtsam::Rot3::Quaternion(r_w, r_x, r_y, r_z), gtsam::Point3(p_x, p_y, p_z));

              p_x = frameWindow[i].pose.translation().x();
              p_y = frameWindow[i].pose.translation().y();
              p_z = frameWindow[i].pose.translation().z();
              r_x = frameWindow[i].pose.rotation().toQuaternion().x();
              r_y = frameWindow[i].pose.rotation().toQuaternion().y();
              r_z = frameWindow[i].pose.rotation().toQuaternion().z();
              r_w = frameWindow[i].pose.rotation().toQuaternion().w();
              gtsam::Pose3 lidarPose_curr = gtsam::Pose3(gtsam::Rot3::Quaternion(r_w, r_x, r_y, r_z), gtsam::Point3(p_x, p_y, p_z));
              gtsam::Pose3 relPose = lidarPose_prev.between(lidarPose_curr);
              // 添加后端里程计因子，当前帧和上一帧的帧间约束
              gtSAMgraph.add(gtsam::BetweenFactor<gtsam::Pose3>(gtsam::Symbol('x', key - windowSize + i), gtsam::Symbol('x', key - windowSize + i + 1), relPose, odomNoise));

              //添加IMU预积分因子
              //上一帧位姿，上一帧速度，下一帧位姿，下一帧速度，上一帧帧偏置，预积分量
              gtsam::ImuFactor imu_factor(gtsam::Symbol('x', key - windowSize + i), gtsam::Symbol('v', key - windowSize + i), gtsam::Symbol('x', key - windowSize + i + 1), gtsam::Symbol('v', key - windowSize + i + 1), gtsam::Symbol('b', key - windowSize + i), (*imuMeasurementsWindow[i - 1]));
              graphFactors.add(imu_factor);
              //添加imu偏置因子，上一帧偏置，下一帧偏置，观测值(应该给多少)，噪声协方差；deltaTij()是积分段的时间
              //todo 噪声处理一下
              graphFactors.add(gtsam::BetweenFactor<gtsam::imuBias::ConstantBias>(gtsam::Symbol('b', key - windowSize + i), gtsam::Symbol('b', key - windowSize + i + 1), gtsam::imuBias::ConstantBias(),
                         gtsam::noiseModel::Diagonal::Sigmas(sqrt(imuMeasurementsWindow[i - 1]->deltaTij()) * noiseModelBetweenBias)));
              // 提取前一帧位姿、速度
              prevPose_  = frameWindow[i - 1].pose;
              prevVel_   = frameWindow[i - 1].velocity;
              // 获取前一帧imu偏置
              prevBias_ = frameWindow[i - 1].bias;
              // 获取前一帧状态
              prevState_ = gtsam::NavState(prevPose_, prevVel_);
              gtsam::NavState propState_ = imuMeasurementsWindow[i - 1]->predict(prevState_, prevBias_);
              // 变量节点赋初值
              //todo 第二次调用出问题了
              if (i == windowSize) {
              	std::cout << "优化阶段要添加x v b" << key - windowSize + i + 1 << "此时key =" << key << std::endl;
              	graphValues.insert(gtsam::Symbol('x', key - windowSize + i + 1), lidarPose_curr);
              	graphValues.insert(gtsam::Symbol('v', key - windowSize + i + 1), propState_.v());
              	graphValues.insert(gtsam::Symbol('b', key - windowSize + i + 1), prevBias_);
              }
            }//end 添加因子
        }//else end
        std::cout << "process_lio mark2" << std::endl;
        //todo 会卡在这里 查看一下有无重复的键
          //优化
          optimizer.update(graphFactors, graphValues);
          optimizer.update();
          graphFactors.resize(0);
          graphValues.clear();
          std::cout << "process_lio mark3" << std::endl;
          //先把结果提取出来 放入frameWindow中 滑窗内第一帧永远是固定先验 所以只提取后面10帧
          gtsam::Values result = optimizer.calculateEstimate();
          for (int i = 1; i <= windowSize; i++) {
              frameWindow[i].pose = result.at<gtsam::Pose3>(gtsam::Symbol('x', key - windowSize + i));
              frameWindow[i].velocity = result.at<gtsam::Vector3>(gtsam::Symbol('v', key - windowSize + i));
              frameWindow[i].bias = result.at<gtsam::imuBias::ConstantBias>(gtsam::Symbol('b', key - windowSize + i));
          }
          std::cout << "process_lio mark4" << std::endl;

          //优化之后重新计算预积分
          //重置优化之后的偏置 [0, 9]
          for (int i = 0; i <= windowSize - 1; i++) {
              //重置优化之后的偏置
              imuMeasurementsWindow[i]->resetIntegrationAndSetBias(frameWindow[i].bias);
              //重新积分
              imuRepropagate((*imuMeasurementsWindow[i]), i);
          }
          std::cout << "process_lio mark5" << std::endl;
		  pubPath_lio();
          std::cout << "process_lio mark" << std::endl;
          needToOptimize = false;
        cond_var.notify_all();
//    }
} // laserOdometryHandler



// 后端质心
void SurfHandler(const sensor_msgs::PointCloud2ConstPtr &_laserSurf)
{
  {
    std::lock_guard<std::mutex> lock(mBuf);
    surfBuf.push(_laserSurf);
//    std::cout << "loop模块接收balm质心: " << odometryBuf.size() << std::endl;
  }
  cond_var.notify_all();  // 唤醒等待的线程
}
void EdgeHandler(const sensor_msgs::PointCloud2ConstPtr &_laserEdge)
{
  {
    std::lock_guard<std::mutex> lock(mBuf);
    edgeBuf.push(_laserEdge);
//    std::cout << "odometryBuf.size(): " << odometryBuf.size() << std::endl;
  }
  cond_var.notify_all();  // 唤醒等待的线程
}


void gpsHandler(const sensor_msgs::NavSatFix::ConstPtr &_gps)
{
    if(useGPS) 
    {
        mBuf.lock();
        gpsBuf.push(_gps);
        mBuf.unlock();
    }
} // gpsHandler

void initNoises( void )
{
    gtsam::Vector priorNoiseVector6(6);
    priorNoiseVector6 << 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12;
    priorNoise = noiseModel::Diagonal::Variances(priorNoiseVector6);

    gtsam::Vector odomNoiseVector6(6);
    // odomNoiseVector6 << 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4;
    odomNoiseVector6 << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4;
    odomNoise = noiseModel::Diagonal::Variances(odomNoiseVector6);

    double loopNoiseScore = 0.5; // constant is ok...
    gtsam::Vector robustNoiseVector6(6); // gtsam::Pose3 factor has 6 elements (6D)
    robustNoiseVector6 << loopNoiseScore, loopNoiseScore, loopNoiseScore, loopNoiseScore, loopNoiseScore, loopNoiseScore;
    robustLoopNoise = gtsam::noiseModel::Robust::Create(
                    gtsam::noiseModel::mEstimator::Cauchy::Create(1), // optional: replacing Cauchy by DCS or GemanMcClure is okay but Cauchy is empirically good.
                    gtsam::noiseModel::Diagonal::Variances(robustNoiseVector6) );

    double bigNoiseTolerentToXY = 1000000000.0; // 1e9
    double gpsAltitudeNoiseScore = 250.0; // if height is misaligned after loop clsosing, use this value bigger
    gtsam::Vector robustNoiseVector3(3); // gps factor has 3 elements (xyz)
    robustNoiseVector3 << bigNoiseTolerentToXY, bigNoiseTolerentToXY, gpsAltitudeNoiseScore; // means only caring altitude here. (because LOAM-like-methods tends to be asymptotically flyging)
    robustGPSNoise = gtsam::noiseModel::Robust::Create(
                    gtsam::noiseModel::mEstimator::Cauchy::Create(1), // optional: replacing Cauchy by DCS or GemanMcClure is okay but Cauchy is empirically good.
                    gtsam::noiseModel::Diagonal::Variances(robustNoiseVector3) );

} // initNoises

Pose6D getOdom(nav_msgs::Odometry::ConstPtr _odom)
{
    auto tx = _odom->pose.pose.position.x;
    auto ty = _odom->pose.pose.position.y;
    auto tz = _odom->pose.pose.position.z;

    double roll, pitch, yaw;
    geometry_msgs::Quaternion quat = _odom->pose.pose.orientation;
    tf::Matrix3x3(tf::Quaternion(quat.x, quat.y, quat.z, quat.w)).getRPY(roll, pitch, yaw);

    return Pose6D{tx, ty, tz, roll, pitch, yaw}; 
} // getOdom

Pose6D diffTransformation(const Pose6D& _p1, const Pose6D& _p2)
{
    Eigen::Affine3f SE3_p1 = pcl::getTransformation(_p1.x, _p1.y, _p1.z, _p1.roll, _p1.pitch, _p1.yaw);
    Eigen::Affine3f SE3_p2 = pcl::getTransformation(_p2.x, _p2.y, _p2.z, _p2.roll, _p2.pitch, _p2.yaw);
    Eigen::Matrix4f SE3_delta0 = SE3_p1.matrix().inverse() * SE3_p2.matrix();
    Eigen::Affine3f SE3_delta; SE3_delta.matrix() = SE3_delta0;
    float dx, dy, dz, droll, dpitch, dyaw;
    pcl::getTranslationAndEulerAngles (SE3_delta, dx, dy, dz, droll, dpitch, dyaw);
    // std::cout << "delta : " << dx << ", " << dy << ", " << dz << ", " << droll << ", " << dpitch << ", " << dyaw << std::endl;

    return Pose6D{double(abs(dx)), double(abs(dy)), double(abs(dz)), double(abs(droll)), double(abs(dpitch)), double(abs(dyaw))};
} // SE3Diff

// 
pcl::PointCloud<PointType>::Ptr local2global(const pcl::PointCloud<PointType>::Ptr &cloudIn, const Pose6D& tf)
{
    pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

    int cloudSize = cloudIn->size();
    cloudOut->resize(cloudSize);

    Eigen::Affine3f transCur = pcl::getTransformation(tf.x, tf.y, tf.z, tf.roll, tf.pitch, tf.yaw);
    
    int numberOfCores = 16;
    #pragma omp parallel for num_threads(numberOfCores)
    for (int i = 0; i < cloudSize; ++i)
    {
        const auto &pointFrom = cloudIn->points[i];
        cloudOut->points[i].x = transCur(0,0) * pointFrom.x + transCur(0,1) * pointFrom.y + transCur(0,2) * pointFrom.z + transCur(0,3);
        cloudOut->points[i].y = transCur(1,0) * pointFrom.x + transCur(1,1) * pointFrom.y + transCur(1,2) * pointFrom.z + transCur(1,3);
        cloudOut->points[i].z = transCur(2,0) * pointFrom.x + transCur(2,1) * pointFrom.y + transCur(2,2) * pointFrom.z + transCur(2,3);
        cloudOut->points[i].intensity = pointFrom.intensity;
    }

    return cloudOut;
}

void pubPath( void )
{
    // pub odom and path 
    nav_msgs::Odometry odomAftPGO;
    nav_msgs::Path pathAftPGO;
    pathAftPGO.header.frame_id = "camera_init";
    mKF.lock(); 
    // for (int node_idx=0; node_idx < int(keyframePosesUpdated.size()) - 1; node_idx++) // -1 is just delayed visualization (because sometimes mutexed while adding(push_back) a new one)
    for (int node_idx=0; node_idx < recentIdxUpdated; node_idx++) // -1 is just delayed visualization (because sometimes mutexed while adding(push_back) a new one)
    {
        // 经过isam回环修正后的位姿
        const Pose6D& pose_est = keyframePosesUpdated.at(node_idx); // upodated poses
        // const gtsam::Pose3& pose_est = isamCurrentEstimate.at<gtsam::Pose3>(node_idx);
        nav_msgs::Odometry odomAftPGOthis;
        odomAftPGOthis.header.frame_id = "camera_init";
        odomAftPGOthis.child_frame_id = "/aft_pgo";
        // 修正前和修正后的位姿的时间是相同的
        odomAftPGOthis.header.stamp = ros::Time().fromSec(keyframeTimes.at(node_idx));
        odomAftPGOthis.pose.pose.position.x = pose_est.x;
        odomAftPGOthis.pose.pose.position.y = pose_est.y;
        odomAftPGOthis.pose.pose.position.z = pose_est.z;
        odomAftPGOthis.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(pose_est.roll, pose_est.pitch, pose_est.yaw);

        odomAftPGO = odomAftPGOthis;

        geometry_msgs::PoseStamped poseStampAftPGO;
        poseStampAftPGO.header = odomAftPGOthis.header;
        poseStampAftPGO.pose = odomAftPGOthis.pose.pose;

        pathAftPGO.header.stamp = odomAftPGOthis.header.stamp;
        pathAftPGO.header.frame_id = "camera_init";
        pathAftPGO.poses.push_back(poseStampAftPGO);
    }
    mKF.unlock();

    //保存轨迹，path_save是文件目录,txt文件提前建好 tum格式 time x y z
    std::ofstream pose1("/media/ctx/0BE20E8D0BE20E8D/dataset/kitti_dataset/result/loamgba/result_pose_09_30_0027_07_loop.txt", std::ios::app);
    pose1.setf(std::ios::scientific, std::ios::floatfield);
    //kitti数据集转换tum格式的数据是18位
    pose1.precision(9);
    //第一个激光帧时间 static变量 只赋值一次
    static double timeStart = odomAftPGO.header.stamp.toSec();
    auto T1 =ros::Time().fromSec(timeStart) ;
    // tf::Quaternion quat;
    // tf::createQuaternionMsgFromRollPitchYaw(double r, double p, double y);//返回四元数
    pose1<< odomAftPGO.header.stamp -T1<< " "
        << -odomAftPGO.pose.pose.position.x << " "
        << -odomAftPGO.pose.pose.position.z << " "
        << -odomAftPGO.pose.pose.position.y<< " "
        << odomAftPGO.pose.pose.orientation.x << " "
        << odomAftPGO.pose.pose.orientation.y << " "
        << odomAftPGO.pose.pose.orientation.z << " "
        << odomAftPGO.pose.pose.orientation.w << std::endl;
    pose1.close();


    pubOdomAftPGO.publish(odomAftPGO); // 最新当前帧的姿态
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

} // pubPath

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

void updatePoses(void)
{
    mKF.lock(); 
    for (int node_idx=0; node_idx < int(isamCurrentEstimate.size()); node_idx++)
    {
        // 引用
        Pose6D& p = keyframePosesUpdated[node_idx];
        // isam更新后的位姿
        p.x = isamCurrentEstimate.at<gtsam::Pose3>(node_idx).translation().x();
        p.y = isamCurrentEstimate.at<gtsam::Pose3>(node_idx).translation().y();
        p.z = isamCurrentEstimate.at<gtsam::Pose3>(node_idx).translation().z();
        p.roll = isamCurrentEstimate.at<gtsam::Pose3>(node_idx).rotation().roll();
        p.pitch = isamCurrentEstimate.at<gtsam::Pose3>(node_idx).rotation().pitch();
        p.yaw = isamCurrentEstimate.at<gtsam::Pose3>(node_idx).rotation().yaw();
    }
    mKF.unlock();

    mtxRecentPose.lock();
    const gtsam::Pose3& lastOptimizedPose = isamCurrentEstimate.at<gtsam::Pose3>(int(isamCurrentEstimate.size())-1);
    recentOptimizedX = lastOptimizedPose.translation().x();
    recentOptimizedY = lastOptimizedPose.translation().y();

    recentIdxUpdated = int(keyframePosesUpdated.size()) - 1;

    mtxRecentPose.unlock();

} // updatePoses

void runISAM2opt(void)
{
    // called when a variable added
    //todo-3 添加gtsam边缘化 在大矩阵上边缘化掉 scancontext添加回环约束也要在大滑窗范围内
    //todo-1 IMU加进来 构建滑窗 参考LIO-SAM
    //todo-2 滑窗内lidar帧间先用相对位姿约束 不行的话再用点面残差约束
    isam->update(gtSAMgraph, initialEstimate);
    isam->update();
    
    gtSAMgraph.resize(0);
    initialEstimate.clear();

    isamCurrentEstimate = isam->calculateEstimate();
    updatePoses();
}

pcl::PointCloud<PointType>::Ptr transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn, gtsam::Pose3 transformIn)
{
    pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

    PointType *pointFrom;

    int cloudSize = cloudIn->size();
    cloudOut->resize(cloudSize);

    Eigen::Affine3f transCur = pcl::getTransformation(
                                    transformIn.translation().x(), transformIn.translation().y(), transformIn.translation().z(), 
                                    transformIn.rotation().roll(), transformIn.rotation().pitch(), transformIn.rotation().yaw() );
    
    int numberOfCores = 8; // TODO move to yaml 
    #pragma omp parallel for num_threads(numberOfCores)
    for (int i = 0; i < cloudSize; ++i)
    {
        pointFrom = &cloudIn->points[i];
        cloudOut->points[i].x = transCur(0,0) * pointFrom->x + transCur(0,1) * pointFrom->y + transCur(0,2) * pointFrom->z + transCur(0,3);
        cloudOut->points[i].y = transCur(1,0) * pointFrom->x + transCur(1,1) * pointFrom->y + transCur(1,2) * pointFrom->z + transCur(1,3);
        cloudOut->points[i].z = transCur(2,0) * pointFrom->x + transCur(2,1) * pointFrom->y + transCur(2,2) * pointFrom->z + transCur(2,3);
        cloudOut->points[i].intensity = pointFrom->intensity;
    }
    return cloudOut;
} // transformPointCloud

void loopFindNearKeyframesCloud( pcl::PointCloud<PointType>::Ptr& nearKeyframes, const int& key, const int& submap_size, const int& root_idx)
{
    // extract and stacking near keyframes (in global coord)
    nearKeyframes->clear();
    for (int i = -submap_size; i <= submap_size; ++i) 
    {
        int keyNear = key + i; // see https://github.com/gisbi-kim/SC-A-LOAM/issues/7 ack. @QiMingZhenFan found the error and modified as below. 
        if (keyNear < 0 || keyNear >= int(keyframeLaserClouds_.size()) )
            continue;

        mKF.lock(); 
        // P_curr * T_curr2w = P_w
        // TODO: root_idx --> keyNear
        *nearKeyframes += * local2global(std::get<0>(keyframeLaserClouds_[keyNear]), keyframePosesUpdated[root_idx]);
        mKF.unlock(); 
    }

    if (nearKeyframes->empty())
        return;

    // downsample near keyframes
    pcl::PointCloud<PointType>::Ptr cloud_temp(new pcl::PointCloud<PointType>());
    downSizeFilterICP.setInputCloud(nearKeyframes);
    downSizeFilterICP.filter(*cloud_temp);
    *nearKeyframes = *cloud_temp;
} // loopFindNearKeyframesCloud

// scan2map优化后的位姿
std::optional<gtsam::Pose3> doICPVirtualRelative( int _loop_kf_idx, int _curr_kf_idx )
{
    // parse pointclouds
    int historyKeyframeSearchNum = 30; // enough. ex. [-25, 25] covers submap length of 50x1 = 50m if every kf gap is 1m
    pcl::PointCloud<PointType>::Ptr cureKeyframeCloud(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr targetKeyframeCloud(new pcl::PointCloud<PointType>());
    // 更新P_curr_w
    //
    loopFindNearKeyframesCloud(cureKeyframeCloud, _curr_kf_idx, 0, _loop_kf_idx); // use same root of loop kf idx 
    // 输出P_curr_w的前后五十帧
    loopFindNearKeyframesCloud(targetKeyframeCloud, _loop_kf_idx, historyKeyframeSearchNum, _loop_kf_idx); 

    /*
    sensor_msgs::PointCloud2 cureKeyframeCloudMsg;
    pcl::toROSMsg(*cureKeyframeCloud, cureKeyframeCloudMsg);
    cureKeyframeCloudMsg.header.frame_id = "camera_init";
    pubLoopScanLocal.publish(cureKeyframeCloudMsg);// 空的

    sensor_msgs::PointCloud2 targetKeyframeCloudMsg;
    pcl::toROSMsg(*targetKeyframeCloud, targetKeyframeCloudMsg);
    targetKeyframeCloudMsg.header.frame_id = "camera_init";
    pubLoopSubmapLocal.publish(targetKeyframeCloudMsg);
    */


    // -------------------------------------pcl-------------------------------------
    // ICP Settings
    pcl::IterativeClosestPoint<PointType, PointType> icp;
    icp.setMaxCorrespondenceDistance(150); // giseop , use a value can cover 2*historyKeyframeSearchNum range in meter 
    icp.setMaximumIterations(100);
    icp.setTransformationEpsilon(1e-6);
    icp.setEuclideanFitnessEpsilon(1e-6);
    icp.setRANSACIterations(0);

    // Align pointclouds
    // P_curr2w
    icp.setInputSource(cureKeyframeCloud);
    // P_local2w
    icp.setInputTarget(targetKeyframeCloud);
    pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
    icp.align(*unused_result);
 
    float loopFitnessScoreThreshold = 0.3; // user parameter but fixed low value is safe. 
    if (icp.hasConverged() == false || icp.getFitnessScore() > loopFitnessScoreThreshold) 
    {
        std::cout << "[SC loop] ICP fitness test failed (" << icp.getFitnessScore() << " > " << loopFitnessScoreThreshold << "). Reject this SC loop." << std::endl;
        return std::nullopt;
    } 
    else 
    {
        std::cout << "[SC loop] ICP fitness test passed (" << icp.getFitnessScore() << " < " << loopFitnessScoreThreshold << "). Add this SC loop." << std::endl;
    }

    // Get pose transformation
    float x, y, z, roll, pitch, yaw;
    Eigen::Affine3f correctionLidarFrame;
    // ICP迭代优化后的位姿
    correctionLidarFrame = icp.getFinalTransformation();
    // -------------------------------------pcl-------------------------------------

    pcl::getTranslationAndEulerAngles (correctionLidarFrame, x, y, z, roll, pitch, yaw);
    gtsam::Pose3 poseFrom = Pose3(Rot3::RzRyRx(roll, pitch, yaw), Point3(x, y, z));
    gtsam::Pose3 poseTo = Pose3(Rot3::RzRyRx(0.0, 0.0, 0.0), Point3(0.0, 0.0, 0.0));

    return poseFrom.between(poseTo);
} // doICPVirtualRelative

void process_pg()
{
    while(ros::ok())
    {
        //todo 加入IMU预积分部分
		while ( !odometryBuf.empty()  && !surfBuf.empty() && !edgeBuf.empty() )
        {
			mBuf.lock();
            while (!odometryBuf.empty() && odometryBuf.front()->header.stamp.toSec() < edgeBuf.front()->header.stamp.toSec())
            {
                // 先进先出
                odometryBuf.pop();
            }
            if (odometryBuf.empty())
            {
                mBuf.unlock();
                break;
            }

            // Time equal check
            timeLaserOdometry = odometryBuf.front()->header.stamp.toSec();
            timeSurf = surfBuf.front()->header.stamp.toSec();
            timeEdge = edgeBuf.front()->header.stamp.toSec();

            if(timeLaserOdometry != timeEdge || timeLaserOdometry != timeSurf || timeSurf != timeEdge || timeEdge != timeLaserOdometry )
            {
                cout << "unsync timestamp!!!!!!!!!!!!!!" << endl;
                printf("time corner %f surf %f odom %f \n", timeEdge, timeSurf,  timeLaserOdometry);

                sleep(1000);
            }



            pcl::PointCloud<PointType>::Ptr SurfFrame(new pcl::PointCloud<PointType>());
            pcl::fromROSMsg(*surfBuf.front(), *SurfFrame);
            surfBuf.pop();

            pcl::PointCloud<PointType>::Ptr EdgeFrame(new pcl::PointCloud<PointType>());
            pcl::fromROSMsg(*edgeBuf.front(), *EdgeFrame);
            edgeBuf.pop();



            // -------------------------------------------------------------------
            Pose6D pose_curr = getOdom(odometryBuf.front());
            odometryBuf.pop();

            // find nearest gps
            double eps = 0.1; 
            // find a gps topioc arrived within eps second 
            while (!gpsBuf.empty())
            {
                auto thisGPS = gpsBuf.front();
                auto thisGPSTime = thisGPS->header.stamp.toSec();
                if( abs(thisGPSTime - timeLaserOdometry) < eps )
                {
                    currGPS = thisGPS;
                    hasGPSforThisKF = true; 
                    break;
                }
                else 
                {
                    hasGPSforThisKF = false;
                }
                gpsBuf.pop();
            }
            mBuf.unlock(); 

            //
            // Early reject by counting local delta movement (for equi-spereated kf drop)
            // 当前帧变为上一帧
            odom_pose_prev = odom_pose_curr;
            odom_pose_curr = pose_curr;
            // δT_curr2last
            Pose6D dtf = diffTransformation(odom_pose_prev, odom_pose_curr); // dtf means delta_transform

            double delta_translation = sqrt(dtf.x*dtf.x + dtf.y*dtf.y + dtf.z*dtf.z); // note: absolute value. 
            translationAccumulated += delta_translation;
            rotaionAccumulated += (dtf.roll + dtf.pitch + dtf.yaw); // sum just naive approach.  
            // 旋转和平移大于一定阈值就认为是关键帧
            if( translationAccumulated > keyframeMeterGap || rotaionAccumulated > keyframeRadGap ) 
            {
                isNowKeyFrame = true;
                translationAccumulated = 0.0; // reset 
                rotaionAccumulated = 0.0; // reset 
            }
            else 
            {
                isNowKeyFrame = false;
            }
            // 不是关键帧就跳过当前帧
            if( ! isNowKeyFrame ) 
                continue; 

            if( !gpsOffsetInitialized )
            {
                if(hasGPSforThisKF)
                {
                    // if the very first frame 第一帧作为初始值
                    gpsAltitudeInitOffset = currGPS->altitude;
                    gpsOffsetInitialized = true;
                } 
            }

            //
            // Save data and Add consecutive node 
            // 
            pcl::PointCloud<PointType>::Ptr thisKeyFrame(new pcl::PointCloud<PointType>());
//            thisKeyFrameDS = thisKeyFrame;
//            downSizeFilterScancontext.setInputCloud(thisKeyFrame);
//            downSizeFilterScancontext.filter(*thisKeyFrameDS);

            mKF.lock();
            if(useCSF)
            {
                // --------------------------- CSF ---------------------------
                CSF csf;
                csf.params.iterations = 600;
                csf.params.time_step = 0.95;
                csf.params.cloth_resolution = 3;
                csf.params.bSloopSmooth = false;
                csf.setPointCloud(*SurfFrame);
                // pcl::io::savePCDFileBinary(map_save_directory, *SurfFrame);

                std::vector<int>  groundIndexes, offGroundIndexes;
                // 输出的是vector<int>类型的地面点和非地面点索引
                pcl::PointCloud<pcl::PointXYZI>::Ptr groundFrame(new pcl::PointCloud<pcl::PointXYZI>);
                pcl::PointCloud<pcl::PointXYZI>::Ptr offGroundFrame(new pcl::PointCloud<pcl::PointXYZI>);
                csf.do_filtering(groundIndexes, offGroundIndexes);
                pcl::copyPointCloud(*SurfFrame, groundIndexes, *groundFrame);
                pcl::copyPointCloud(*SurfFrame, offGroundIndexes, *offGroundFrame);
                //FilterGround.setInputCloud(groundFrame_);
                //FilterGround.filter(*groundFrame_);
                *SurfFrame = *groundFrame;

                *EdgeFrame += *offGroundFrame;// +cornerFrame
            }
            *thisKeyFrame = *EdgeFrame + *SurfFrame;
            // 存放关键帧的容器
            keyframeLaserClouds_.push_back(std::make_tuple(thisKeyFrame, EdgeFrame, SurfFrame));
            
            keyframePoses.push_back(pose_curr);
            // 后端里程计位姿
            keyframePosesUpdated.push_back(pose_curr); // init
            keyframeTimes.push_back(timeLaserOdometry);
            // 
            scManager.makeAndSaveScancontextAndKeys(*thisKeyFrame);

            laserCloudMapPGORedraw = true;
            mKF.unlock(); 

            const int prev_node_idx = keyframePoses.size() - 2; 
            const int curr_node_idx = keyframePoses.size() - 1; // becuase cpp starts with 0 (actually this index could be any number, but for simple implementation, we follow sequential indexing)
            if( ! gtSAMgraphMade /* prior node */) 
            {
                const int init_node_idx = 0; 
                // 起始位姿
                gtsam::Pose3 poseOrigin = Pose6DtoGTSAMPose3(keyframePoses.at(init_node_idx));
                // auto poseOrigin = gtsam::Pose3(gtsam::Rot3::RzRyRx(0.0, 0.0, 0.0), gtsam::Point3(0.0, 0.0, 0.0));

                mtxPosegraph.lock();
                {
                    // prior factor 添加先验因子
                    gtSAMgraph.add(gtsam::PriorFactor<gtsam::Pose3>(init_node_idx, poseOrigin, priorNoise));
                    initialEstimate.insert(init_node_idx, poseOrigin);
                    // runISAM2opt();          
                }   
                mtxPosegraph.unlock();

                gtSAMgraphMade = true; 

                cout << "posegraph prior node " << init_node_idx << " added" << endl;
            } 
            else /* consecutive node (and odom factor) after the prior added */ 
            { 
                // == keyframePoses.size() > 1 
                gtsam::Pose3 poseFrom = Pose6DtoGTSAMPose3(keyframePoses.at(prev_node_idx));
                gtsam::Pose3 poseTo = Pose6DtoGTSAMPose3(keyframePoses.at(curr_node_idx));

                mtxPosegraph.lock();
                {
                    // odom factor
                    // T_curr2last
                    gtsam::Pose3 relPose = poseFrom.between(poseTo);
                    // 添加后端里程计因子，当前帧和上一帧的帧间约束
                    gtSAMgraph.add(gtsam::BetweenFactor<gtsam::Pose3>(prev_node_idx, curr_node_idx, relPose, odomNoise));

                    // gps factor
                    if(hasGPSforThisKF)
                    {
                        double curr_altitude_offseted = currGPS->altitude - gpsAltitudeInitOffset;
                        mtxRecentPose.lock();
                        // in this example, only adjusting altitude (for x and y, very big noises are set) 只用到GPS的海拔高度
                        gtsam::Point3 gpsConstraint(recentOptimizedX, recentOptimizedY, curr_altitude_offseted);
                        mtxRecentPose.unlock();
                        // 添加GPS因子
                        gtSAMgraph.add(gtsam::GPSFactor(curr_node_idx, gpsConstraint, robustGPSNoise));
                        cout << "GPS factor added at node " << curr_node_idx << endl;
                    }
                    initialEstimate.insert(curr_node_idx, poseTo);
                    writeEdge({prev_node_idx, curr_node_idx}, relPose, edges_str); // giseop
                    // runISAM2opt();
                }
                mtxPosegraph.unlock();
                if(curr_node_idx % 100 == 0)
                    cout << "posegraph odom node " << curr_node_idx << " added." << endl;
            }
            // if want to print the current graph, use gtSAMgraph.print("\nFactor Graph:\n");

            // save utility 
            std::string curr_node_idx_str = padZeros(curr_node_idx);
            //pcl::io::savePCDFileBinary(pgScansDirectory + curr_node_idx_str + ".pcd", *thisKeyFrame); // scan 

            const auto& curr_scd = scManager.getConstRefRecentSCD();
            saveSCD(pgSCDsDirectory + curr_node_idx_str + ".scd", curr_scd);

            pgTimeSaveStream << timeCenter<< std::endl; // path 

        }

        // ps. 
        // scan context detector is running in another thread (in constant Hz, e.g., 1 Hz)
        // pub path and point cloud in another thread

        // wait (must required for running the while loop)
        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }
} // process_pg

void performSCLoopClosure(void)
{
    if( int(keyframePoses.size()) < scManager.NUM_EXCLUDE_RECENT) // do not try too early 
        return;

    auto detectResult = scManager.detectLoopClosureID(); // first: nn index, second: yaw diff 
    int SCclosestHistoryFrameID = detectResult.first;
    if( SCclosestHistoryFrameID != -1 ) 
    { 
        // 回环检测到的ID
        const int prev_node_idx = SCclosestHistoryFrameID;
        const int curr_node_idx = keyframePoses.size() - 1; // because cpp starts 0 and ends n-1
        cout << "Loop detected! - between " << prev_node_idx << " and " << curr_node_idx << "" << endl;

        mBuf.lock();
        // 回环ID
        scLoopICPBuf.push(std::pair<int, int>(prev_node_idx, curr_node_idx));
        // addding actual 6D constraints in the other thread, icp_calculation.
        mBuf.unlock();
    }
} // performSCLoopClosure

void process_lcd(void)
{
    // 10~20hz last = 30
    //float loopClosureFrequency = 20.0; // can change
    ros::Rate rate(loopClosureFreq);
    while (ros::ok())
    {
        rate.sleep();
        performSCLoopClosure();
        // performRSLoopClosure(); // TODO
    }
} // process_lcd

void process_icp(void)
{
    while(1)
    {
		while ( !scLoopICPBuf.empty() )
        {
            if( scLoopICPBuf.size() > 30 )
            {
                ROS_WARN("Too many loop clousre candidates to be ICPed is waiting ... Do process_lcd less frequently (adjust loopClosureFrequency)");
            }

            mBuf.lock(); 
            std::pair<int, int> loop_idx_pair = scLoopICPBuf.front();
            scLoopICPBuf.pop();
            mBuf.unlock();

            const int prev_node_idx = loop_idx_pair.first;
            const int curr_node_idx = loop_idx_pair.second;
            // TODO:使用pcl太耗时了 30ms
            clock_t start, end;
            double time;
            start = clock();
            // 返回当前帧到回环帧的位姿：curr_scan2loop_localmap的位姿 T_curr2loop_kf
            auto relative_pose_optional = doICPVirtualRelative(prev_node_idx, curr_node_idx);
            end = clock();
            time = ((double) (end - start)) / CLOCKS_PER_SEC;
            // 0.04s
            cout << "loop2local comsumming Time: " << time << "s" << endl;

            if(relative_pose_optional) 
            {
                gtsam::Pose3 relative_pose = relative_pose_optional.value();
                mtxPosegraph.lock();
                // 添加pcl回环因子
                gtSAMgraph.add(gtsam::BetweenFactor<gtsam::Pose3>(prev_node_idx, curr_node_idx, relative_pose, robustLoopNoise));
                writeEdge({prev_node_idx, curr_node_idx}, relative_pose, edges_str); // giseop
                // runISAM2opt();
                mtxPosegraph.unlock();
            }
        }

        // wait (must required for running the while loop)
        std::chrono::milliseconds dura(2);
        // 2ms
        std::this_thread::sleep_for(dura);
    }
} // process_icp

void process_viz_path(void)
{
    ros::Rate rate(vizPathFreq);
    while (ros::ok()) 
    {
        rate.sleep();
//        if(recentIdxUpdated > 1)
//        {
            pubPath_lio();
//        }
    }
}

void process_isam(void)
{
    //一秒优化一次 last = 30
    // float hz = 40;
    ros::Rate rate(processIsamFreq);
    while (ros::ok())
    {
        rate.sleep();
        //保证至少有先验信息
        if( gtSAMgraphMade )
        {
            mtxPosegraph.lock();
            // 全局图优化
            runISAM2opt();
            cout << "running isam2 optimization ..." << endl;
            mtxPosegraph.unlock();

//            saveOptimizedVerticesKITTIformat(isamCurrentEstimate, pgKITTIformat); // pose
//            saveOdometryVerticesKITTIformat(odomKITTIformat); // pose
//            saveGTSAMgraphG2oFormat(isamCurrentEstimate);
        }
    }
}


void process_viz_map(void)
{
    // 一秒20次
    float vizmapFrequency = vizMapFreq; // 0.1 means run onces every 10s
    ros::Rate rate(vizmapFrequency);
    while (ros::ok())
    {
        rate.sleep();
        if(recentIdxUpdated > 1)
        {
            // pubMap();
            int SKIP_FRAMES = 1; // sparse map visulalization to save computations
            int counter = 0;
            laserCloudMapPGO->clear();
            offGroundMap->clear();
            mKF.lock();
            // 全局位姿被优化后，遍历所有位姿，重新构建大地图并发布
            for (int node_idx=0; node_idx < recentIdxUpdated; node_idx++)
            {
                if(counter % SKIP_FRAMES == 0)
                {
                    // 找到位姿修正后的对应的那一帧
                    // P_w = P_curr * T_curr2w
                    *laserCloudMapPGO += *local2global(std::get<2>(keyframeLaserClouds_[node_idx]), keyframePosesUpdated[node_idx]);
                    *offGroundMap += *local2global(std::get<1>(keyframeLaserClouds_[node_idx]), keyframePosesUpdated[node_idx]);
                }
                counter++;
            }
            mKF.unlock();
            if(useCSF)
            {
                downSizeFilterMapPGO.setInputCloud(laserCloudMapPGO);
                downSizeFilterMapPGO.filter(*laserCloudMapPGO);
            }
            if(viewMap)
            {
                sensor_msgs::PointCloud2 laserCloudMapPGOMsg;
                pcl::toROSMsg((*laserCloudMapPGO+*offGroundMap), laserCloudMapPGOMsg);
                laserCloudMapPGOMsg.header.frame_id = "camera_init";
                // 发布回环修正后的降采样地图
                pubMapAftPGO.publish(laserCloudMapPGOMsg);
            }

        }
    }
    cout << "---------------------- SaveMap ----------------------" << endl;
    pcl::PointCloud<PointType>::Ptr globalMap(new pcl::PointCloud<PointType>());
    *globalMap = *laserCloudMapPGO + *offGroundMap;
    pcl::io::savePCDFileBinary(map_save_directory, *globalMap);
}


void saveMapTraj()
{
    double q1,q2,q3,q4,x,y,z;
    /*
    for(int i = 0; i < globalPath.poses.size(); i++)
    {
        q1 = globalPath.poses[i].pose.orientation.x;
        q2 = globalPath.poses[i].pose.orientation.y;
        q3 = globalPath.poses[i].pose.orientation.z;
        q4 = globalPath.poses[i].pose.orientation.w;
        x = globalPath.poses[i].pose.position.x;
        y = globalPath.poses[i].pose.position.y;
        z = globalPath.poses[i].pose.position.z;
    }
    cout << q1 << " " << q2 << " "  << q3 << " "  << q4 << " "  <<  x << " "  << y << " "  << z << endl;
    */
}

/**
 * imu原始测量数据转换到lidar系，加速度、角速度、RPY
*/
sensor_msgs::Imu imuConverter(const sensor_msgs::Imu& imu_in)
{
    sensor_msgs::Imu imu_out = imu_in;
    // 加速度，只跟xyz坐标系的旋转有关系
    Eigen::Vector3d acc(imu_in.linear_acceleration.x, imu_in.linear_acceleration.y, imu_in.linear_acceleration.z);
    acc = extRot * acc;
    imu_out.linear_acceleration.x = acc.x();
    imu_out.linear_acceleration.y = acc.y();
    imu_out.linear_acceleration.z = acc.z();
    // 角速度，只跟xyz坐标系的旋转有关系
    Eigen::Vector3d gyr(imu_in.angular_velocity.x, imu_in.angular_velocity.y, imu_in.angular_velocity.z);
    gyr = extRot * gyr;
    imu_out.angular_velocity.x = gyr.x();
    imu_out.angular_velocity.y = gyr.y();
    imu_out.angular_velocity.z = gyr.z();
    //todo 9轴IMU才会用到 可能不需要
    // RPY
    Eigen::Quaterniond q_from(imu_in.orientation.w, imu_in.orientation.x, imu_in.orientation.y, imu_in.orientation.z);
    // 为什么是右乘，可以动手画一下看看
    Eigen::Quaterniond q_final = q_from * extQRPY;
    imu_out.orientation.x = q_final.x();
    imu_out.orientation.y = q_final.y();
    imu_out.orientation.z = q_final.z();
    imu_out.orientation.w = q_final.w();

    if (sqrt(q_final.x()*q_final.x() + q_final.y()*q_final.y() + q_final.z()*q_final.z() + q_final.w()*q_final.w()) < 0.1)
    {
        ROS_ERROR("Invalid quaternion, please use a 9-axis IMU!");
        ros::shutdown();
    }
    return imu_out;
}

/**
 * 订阅imu原始数据
 * 1、用上一帧激光里程计时刻对应的状态、偏置，施加从该时刻开始到当前时刻的imu预积分量，得到当前时刻的状态，也就是imu里程计
 * 2、imu里程计位姿转到lidar系，发布里程计
*/
//向imuBuf中填入IMU信息
void imuHandler(const sensor_msgs::Imu::ConstPtr& imu_raw)
{
    std::lock_guard<std::mutex> lock(mBuf);
    // imu原始测量数据转换到lidar系，加速度、角速度、RPY
    sensor_msgs::Imu thisImu = imuConverter(*imu_raw);
    // 添加当前帧imu数据到队列
    // test 给零偏加固定偏移
    // thisImu.angular_velocity.x += 3;
    // thisImu.angular_velocity.y += 3;
    // thisImu.angular_velocity.z += 3;
    //这里先不做对齐
    //目前不用IMU的姿态角信息
    imuBuf.push(thisImu);
//    std::cout << "imu回调函数中 imuBuf.size(): " << imuBuf.size() << std::endl;
}

//todo
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

    last_imu_t = imu_msg->header.stamp.toSec();

    {
        std::lock_guard<std::mutex> lg(m_state);
        predict(imu_msg);
//        std_msgs::Header header = imu_msg->header;
//        header.frame_id = "world";
//        if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
//            pubLatestOdometry(tmp_P, tmp_Q, tmp_V, header);
    }
}

// 获取IMU数据并对齐到两个LiDAR帧之间
void processIMU() {
  //10hz处理
//    ros::Rate rate(5);
    while (ros::ok()) {
//      	rate.sleep();
      	//降低频率否则会出问题
        std::vector<sensor_msgs::Imu> imuDataBetweenFrames;
        //只进来一次
//        std::cout << "获取mBuf前mark" << std::endl;
        std::unique_lock<std::mutex> lock(mBuf);
        //这里目前只考虑lidar里程计和imu信息，最后在把回环加入进来时再考虑点云信息
        //todo 这里始终在0附近 进不去
        std::cout << "processimu odometryBuf.size(): " << odometryBuf.size() << std::endl;
        cond_var.wait(lock, [] { return !odometryBuf.empty() && !imuBuf.empty(); });
        // 对齐两个LiDAR帧之间的IMU数据 把imuBuf分到imuBufAligned
        auto currOdom = odometryBuf.front();
        odometryBuf.pop();
        frameTime = currOdom -> header.stamp.toSec();
        //第一帧处理
        if (lastOdom == nullptr) {
            //等待下一个激光帧
            lastOdom = currOdom;
            //把当前帧存下来
            Frame firstFrame;
        	// 滑窗内的位姿存储
        	gtsam::Pose3 lidarPose = gtsam::Pose3(gtsam::Rot3::Quaternion(currOdom->pose.pose.orientation.w,
            	                                 currOdom->pose.pose.orientation.x,
                	                             currOdom->pose.pose.orientation.y,
                    	                         currOdom->pose.pose.orientation.z),
                        	    Point3(currOdom->pose.pose.position.x,
                            	       currOdom->pose.pose.position.y,
                                	   currOdom->pose.pose.position.z));
        	firstFrame.pose = lidarPose;
        	//第一帧速度和零偏都给0 后续的利用IMU预积分器推导
        	firstFrame.velocity = gtsam::Vector3(0, 0, 0);
        	firstFrame.bias = gtsam::imuBias::ConstantBias();
        	frameWindow.push_back(firstFrame);
            frameTimes.push_back(frameTime);
        	// 如果滑窗内的数据超过了指定大小，移除最旧的一帧
        	if (frameWindow.size() > windowSize + 1) {
            	frameWindow.erase(frameWindow.begin());
        	}
            if (frameTimes.size() > windowSize + 1) {
            	frameTimes.erase(frameTimes.begin());
        	}
        	//维护imu预积分器和imuBufAligned的大小
        	if (imuMeasurementsWindow.size() > windowSize) {
          		imuMeasurementsWindow.erase(imuMeasurementsWindow.begin());
        	}
        	if (imuBufAligned.size() > windowSize) {
          		imuBufAligned.erase(imuBufAligned.begin());
        	}
            //这里不用key++，key=0指向第一帧
            continue;
        }
        //非第一帧
        //IMU预积分
        //todo 这里时间获取有问题 应该没问题 这里的时间戳总一样
        double startTime = lastOdom->header.stamp.toSec();
        double endTime = currOdom->header.stamp.toSec();
        //todo 这里需要处理首尾部分的IMU数据，和lidar对齐，参考LIO-SAM
        std::cout << "imuBuf.size(): " << imuBuf.size() << std::endl;
        while (!imuBuf.empty()) {
            auto imuData = imuBuf.front();
            double imuTime = imuData.header.stamp.toSec();
            //这里进来了
//            std::cout << "进来了" << std::endl;
//            std::cout << "imuTime: " << imuTime << std::endl;
//            std:;cout << "startTime: " << startTime << endl;
//              std::cout << "endTime: " << endTime << endl;
            if (imuTime >= startTime && imuTime <= endTime) {
              	//todo 这里没进来过
              	std::cout << "放入imu信息"	<< std::endl;
                imuDataBetweenFrames.push_back(imuData);
                std::cout << "inside imuDataBetweenFrames.size(): " << imuDataBetweenFrames.size() << std::endl;
                imuBuf.pop();
            } else if (imuTime > endTime) {
                break;
            } else {
              imuBuf.pop();
            }
        }
        //todo 这里始终为0 问题最大
		std::cout << "imuDataBetweenFrames.size(): " << imuDataBetweenFrames.size() << std::endl;
                //todo 这里可能进不去
        if (!imuDataBetweenFrames.empty()) {
          // 将对齐后的IMU数据存入imuBufAligned
          imuBufAligned.push_back(imuDataBetweenFrames);
          // 创建预积分器
          auto preintegrator = std::make_shared<gtsam::PreintegratedImuMeasurements>(initialImuPreintegratorParam, priorImuBias);
          double lastImuTime = -1;
          // 进行预积分
          for (const auto& imuData : imuDataBetweenFrames) {
            double imuTime = imuData.header.stamp.toSec();
            // 提取IMU加速度和角速度
            // 假设IMU数据中包含必要的信息进行预积分
            gtsam::Vector3 accel(imuData.linear_acceleration.x,
                                 imuData.linear_acceleration.y,
                                 imuData.linear_acceleration.z);
            gtsam::Vector3 gyro(imuData.angular_velocity.x,
                                imuData.angular_velocity.y,
                                imuData.angular_velocity.z);
            double dt = (lastImuTime < 0) ? (1.0 / 500.0) : (imuTime - lastImuTime);
            // 将IMU数据添加到预积分器
            preintegrator->integrateMeasurement(accel, gyro, dt);
            lastImuTime = imuTime;
          }
          // 将预积分器存入imuMeasurementsWindow
          imuMeasurementsWindow.push_back(preintegrator);
        }

        Frame thisFrame;
        // 滑窗内的位姿存储
        gtsam::Pose3 lidarPose = gtsam::Pose3(gtsam::Rot3::Quaternion(currOdom->pose.pose.orientation.w,
                                             currOdom->pose.pose.orientation.x,
                                             currOdom->pose.pose.orientation.y,
                                             currOdom->pose.pose.orientation.z),
                            Point3(currOdom->pose.pose.position.x,
                                   currOdom->pose.pose.position.y,
                                   currOdom->pose.pose.position.z));
        thisFrame.pose = lidarPose;
        //todo 应该给多少 这里随便给，在process_lio中会利用IMU预积分器推导 check一下
        thisFrame.velocity = gtsam::Vector3(0, 0, 0);
        thisFrame.bias = gtsam::imuBias::ConstantBias();
        frameWindow.push_back(thisFrame);
        frameTimes.push_back(frameTime);
//        std::cout << "frameWindow.size(): " << frameWindow.size() << std::endl;
        //至此 当前帧 上一帧到当前帧的IMU预积分都放到滑窗中了 维护一下滑窗大小 windowSize(10) + 最新一帧
        // 如果滑窗内的数据超过了指定大小，移除最旧的一帧
        if (frameWindow.size() > windowSize + 1) {
            frameWindow.erase(frameWindow.begin());
        }
        if (frameTimes.size() > windowSize + 1) {
            frameTimes.erase(frameTimes.begin());
        }
        //维护imu预积分器和imuBufAligned的大小
        if (imuMeasurementsWindow.size() > windowSize) {
          imuMeasurementsWindow.erase(imuMeasurementsWindow.begin());
        }
        if (imuBufAligned.size() > windowSize) {
          imuBufAligned.erase(imuBufAligned.begin());
        }
        //上一帧更新为当前帧
        lastOdom = currOdom;
        //指向最新帧的索引
        key++;
        cond_var.notify_all();
    }
}

void resetOPtimization() {
  gtsam::ISAM2Params optParameters;
  optParameters.relinearizeThreshold = 0.1;
  optParameters.relinearizeSkip = 1;
  optimizer = gtsam::ISAM2(optParameters);

  gtsam::NonlinearFactorGraph newGraphFactors;
  graphFactors = newGraphFactors;

  gtsam::Values newGraphValues;
  graphValues = newGraphValues;
}

//imu重新预积分
void imuRepropagate(gtsam::PreintegratedImuMeasurements &imuMeasurement, int i) {
       double lastImuTime_repropagate = -1;
       for (int j = 0; j < imuBufAligned[i].size(); j++) {
         sensor_msgs::Imu *thisImu = &imuBufAligned[i][j];
         //todo 确认一下
         double imuTime = ROS_TIME(thisImu);
         // 提取IMU加速度和角速度数据，进行预积分
         gtsam::Vector3 accel(thisImu->linear_acceleration.x,
                              thisImu->linear_acceleration.y,
                              thisImu->linear_acceleration.z);
         gtsam::Vector3 gyro(thisImu->angular_velocity.x,
                             thisImu->angular_velocity.y,
                             thisImu->angular_velocity.z);
         double dt = (lastImuTime_repropagate < 0) ? (1.0 / 500.0) : (imuTime - lastImuTime_repropagate);
         imuMeasurementsWindow[i]->integrateMeasurement(accel, gyro, dt);
         lastImuTime_repropagate = imuTime;
       }
       return ;
}

int main(int argc, char **argv)
{
	ros::init(argc, argv, "laserPGO");
	ros::NodeHandle nh;
	nh.param<std::string>("save_directory", save_directory, "/");
    nh.param<std::string>("map_save_directory", map_save_directory, "/");
    pgKITTIformat = save_directory + "optimized_poses.txt";
    odomKITTIformat = save_directory + "odom_poses.txt";
    // pgG2oSaveStream = std::fstream(save_directory + "singlesession_posegraph.g2o", std::fstream::out);
    pgTimeSaveStream = std::fstream(save_directory + "times.txt", std::fstream::out);
    pgTimeSaveStream.precision(std::numeric_limits<double>::max_digits10);
    pgScansDirectory = save_directory + "Scans/";
    auto unused = system((std::string("exec rm -r ") + pgScansDirectory).c_str());
    unused = system((std::string("mkdir -p ") + pgScansDirectory).c_str());

    pgSCDsDirectory = save_directory + "SCDs/"; // SCD: scan context descriptor 
    unused = system((std::string("exec rm -r ") + pgSCDsDirectory).c_str());
    unused = system((std::string("mkdir -p ") + pgSCDsDirectory).c_str());

    // system params 
	nh.param<double>("keyframe_meter_gap", keyframeMeterGap, 2.0); // pose assignment every k m move 
	nh.param<double>("keyframe_deg_gap", keyframeDegGap, 10.0); // pose assignment every k deg rot 
    keyframeRadGap = deg2rad(keyframeDegGap);
	nh.param<double>("sc_dist_thres", scDistThres, 0.2);  
	nh.param<double>("sc_max_radius", scMaximumRadius, 80.0); // 80 is recommended for outdoor, and lower (ex, 20, 40) values are recommended for indoor 
    nh.param<double>("vizMapFreq", vizMapFreq, 0.1);
    nh.param<double>("processIsamFreq", processIsamFreq, 20);
    nh.param<double>("vizPathFreq", vizPathFreq, 20);
    nh.param<double>("loopClosureFreq", loopClosureFreq, 20);
    nh.param<bool>("useCSF", useCSF, true);
    nh.param<bool>("follow", follow, true);
    nh.param<bool>("viewMap", viewMap, true);
    nh.param<bool>("useIsam", useIsam, true);

    scManager.setSCdistThres(scDistThres);
    scManager.setMaximumRadius(scMaximumRadius);

    float filter_size = 0.4; 
    // downSizeFilterScancontext.setLeafSize(filter_size, filter_size, filter_size);
    downSizeFilterICP.setLeafSize(filter_size, filter_size, filter_size);

    // 0.4 --> 0.3
	nh.param<double>("mapviz_filter_size", FilterGroundLeaf, 0.2); // pose assignment every k frames
    downSizeFilterMapPGO.setLeafSize(FilterGroundLeaf, FilterGroundLeaf, FilterGroundLeaf);

    //全局变量初始化
    // 初始化的imu预积分的噪声协方差
    initialImuPreintegratorParam->accelerometerCovariance  = gtsam::Matrix33::Identity(3,3) * pow(imuAccNoise, 2); // acc white noise in continuous
    initialImuPreintegratorParam->gyroscopeCovariance      = gtsam::Matrix33::Identity(3,3) * pow(imuGyrNoise, 2); // gyro white noise in continuous
    initialImuPreintegratorParam->integrationCovariance    = gtsam::Matrix33::Identity(3,3) * pow(1e-4, 2); // error committed in integrating position from velocities
    //初始化首个LiDAR位姿先验协方差 完全固定的先验
    priorPoseNoise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 0, 0, 0, 0, 0, 0).finished());
    //初始化速度先验协方差
    priorVelNoise = gtsam::noiseModel::Isotropic::Sigma(3, 1e4); //m/s
    //初始化IMU零偏先验协方差
    priorBiasNoise = gtsam::noiseModel::Isotropic::Sigma(6, 1e-3); //1e-2 ~ 1e-3 seems to be good;
    //初始化IMU零偏帧间约束的协方差
    noiseModelBetweenBias = (gtsam::Vector(6) << imuAccBiasN, imuAccBiasN, imuAccBiasN, imuGyrBiasN, imuGyrBiasN, imuGyrBiasN).finished();
    // --------------------------------- 订阅后端数据 ---------------------------------
	// ros::Subscriber subCenters = nh.subscribe<sensor_msgs::PointCloud2>("/Center_BA", 100, centerHandler);
//	ros::Subscriber subSurf = nh.subscribe<sensor_msgs::PointCloud2>("/ground_BA", 100, SurfHandler);
//    ros::Subscriber subEdge = nh.subscribe<sensor_msgs::PointCloud2>("/Edge_BA", 100, EdgeHandler);

    ros::Subscriber subLaserOdometry = nh.subscribe<nav_msgs::Odometry>("/BALM_mapped_to_init", 100, laserOdometryHandler);
//	ros::Subscriber subGPS = nh.subscribe<sensor_msgs::NavSatFix>("/gps/fix", 100, gpsHandler);

    //订阅IMU数据
    ros::Subscriber subImu = nh.subscribe<sensor_msgs::Imu>("imu_raw", 2000, imuHandler, ros::TransportHints().tcpNoDelay());
//    ros::Subscriber subImu = nh.subscribe<sensor_msgs::Imu>("imu_raw", 2000, imuHandler);

    // ------------------------------------------------------------------
	pubOdomAftPGO = nh.advertise<nav_msgs::Odometry>("/aft_pgo_odom", 100);
	pubOdomRepubVerifier = nh.advertise<nav_msgs::Odometry>("/repub_odom", 100);
	pubPathAftPGO = nh.advertise<nav_msgs::Path>("/aft_pgo_path", 100);
	// 回环修正后的地图
    pubMapAftPGO = nh.advertise<sensor_msgs::PointCloud2>("/aft_pgo_map", 100);
	pubLoopScanLocal = nh.advertise<sensor_msgs::PointCloud2>("/loop_scan_local", 100);
	pubLoopSubmapLocal = nh.advertise<sensor_msgs::PointCloud2>("/loop_submap_local", 100);

    //执行后端优化的线程
    std::thread measurement_process{process_lio};
 	ros::spin();

	return 0;
}
