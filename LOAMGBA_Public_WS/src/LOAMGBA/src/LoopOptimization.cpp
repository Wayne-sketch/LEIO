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
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/geometry/Point3.h>

#include "utility/common.h"
#include "utility/tic_toc.h"

#include <pcl/io/pcd_io.h>
#include <image_transport/image_transport.h>
#include "utility/Scancontext.h"
#include "lidarFactor.hpp"

using namespace gtsam;

#include "CSF/CSF.h"
//#include "utility/utility.h"
bool useCSF;
bool follow;
bool viewMap;
bool useIsam;

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
std::vector<Frame> frameWindow;  // 存储滑窗内的位姿 速度 零偏
// windowSize - 1 个IMU预积分器，[0, windowSize - 2]，储滑窗内相邻帧的IMU预积分 imuMeasurementsWindow[i]存储的是poseWindow[i]到poseWindow[i + 1]的预积分
std::vector<std::shared_ptr<gtsam::PreintegratedImuMeasurements>> imuMeasurementsWindow;
bool systemInitialized = false;
int windowSize = 10;
//key永远指向最新到来的lidar帧
int key = 1;
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
std::queue<sensor_msgs::Imu::ConstPtr> imuBuf;
std::condition_variable cond_var;
// 定义IMU预积分器 用于IMU预积分器初始化的变量
//todo IMU预积分器初始化的时候 应该给什么样的参数
////todo 为了优化后通过新的零偏更新预积分值 需要使用windowSize个IMU预积分器 并把相邻两帧之间的imu信息保存下来 每次零偏更新的时候重置一下预积分器然后重新积分一次
//todo processImu把信息对齐保存下来 对齐后的IMU信息 滑窗的时候怎么做更快
std::vector<std::vector<sensor_msgs::Imu::ConstPtr>> imuBufAligned;


////todo 维护多个IMU零偏值 每次优化后保存下来 用于重置预积分器
std::vector<gtsam::imuBias::ConstantBias> bias(windownSize, gtsam::imuBias::ConstantBias()); // IMU偏差
////todo 涉及到指针的传递
//std::vector<std::shared_ptr<gtsam::PreintegratedImuMeasurements>> imuIntegrators(windowSize, std::shared_ptr<gtsam::PreintegratedImuMeasurements>);
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
Eigen::Matrix3d extRot;     // xyz坐标系旋转
Eigen::Matrix3d extRPY;     // RPY欧拉角的变换关系
Eigen::Vector3d extTrans;   // xyz坐标系平移
Eigen::Quaterniond extQRPY;
extRot = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(extRotV.data(), 3, 3);
extRPY = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(extRPYV.data(), 3, 3);
extTrans = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(extTransV.data(), 3, 1);
extQRPY = Eigen::Quaterniond(extRPY).inverse();
// 初始化的imu预积分的噪声协方差
std::shared_ptr<gtsam::PreintegrationParams> initialImuPreintegratorParam = gtsam::PreintegrationParams::MakeSharedU(imuGravity);
initialImuPreintegratorParam->accelerometerCovariance  = gtsam::Matrix33::Identity(3,3) * pow(imuAccNoise, 2); // acc white noise in continuous
initialImuPreintegratorParam->gyroscopeCovariance      = gtsam::Matrix33::Identity(3,3) * pow(imuGyrNoise, 2); // gyro white noise in continuous
initialImuPreintegratorParam->integrationCovariance    = gtsam::Matrix33::Identity(3,3) * pow(1e-4, 2); // error committed in integrating position from velocities
gtsam::imuBias::ConstantBias priorImuBias((gtsam::Vector(6) << 0, 0, 0, 0, 0, 0).finished());; // assume zero initial bias
//初始化首个LiDAR位姿先验协方差 完全固定的先验
gtsam::noiseModel::Diagonal::shared_ptr priorPoseNoise;
priorPoseNoise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 0, 0, 0, 0, 0, 0).finished());
//初始化速度先验协方差
gtsam::noiseModel::Diagonal::shared_ptr priorVelNoise; //rad, rad, rad, m, m, m
priorVelNoise = gtsam::noiseModel::Isotropic::Sigma(3, 1e4); //m/s
//初始化IMU零偏先验协方差
gtsam::noiseModel::Diagonal::shared_ptr priorBiasNoise;
priorBiasNoise = gtsam::noiseModel::Isotropic::Sigmas(6, 1e-3); //1e-2 ~ 1e-3 seems to be good;
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

// 尾部压入消息
// 后端位姿
void laserOdometryHandler(const nav_msgs::Odometry::ConstPtr &_laserOdometry)
{
	mBuf.lock();
	odometryBuf.push(_laserOdometry);
	mBuf.unlock();
} // laserOdometryHandler



// 后端质心
void SurfHandler(const sensor_msgs::PointCloud2ConstPtr &_laserSurf)
{
	mBuf.lock();
	surfBuf.push(_laserSurf);
	mBuf.unlock();
}
void EdgeHandler(const sensor_msgs::PointCloud2ConstPtr &_laserEdge)
{
    mBuf.lock();
    edgeBuf.push(_laserEdge);
    mBuf.unlock();
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
        if(recentIdxUpdated > 1) 
        {
            pubPath();
        }
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
    mBuf.lock();
    // imu原始测量数据转换到lidar系，加速度、角速度、RPY
    sensor_msgs::Imu thisImu = imuConverter(*imu_raw);
    // 添加当前帧imu数据到队列
    // test 给零偏加固定偏移
    // thisImu.angular_velocity.x += 3;
    // thisImu.angular_velocity.y += 3;
    // thisImu.angular_velocity.z += 3;
    //这里先不做对齐
    //目前不用IMU的姿态角信息
    imuBuf.push_back(thisImu);
    mBuf.unlock();
}

// 创建IMU预积分器的函数
std::shared_ptr<gtsam::PreintegratedImuMeasurements> createPreintegrator(const gtsam::ImuParams& params) {
    return std::make_shared<gtsam::PreintegratedImuMeasurements>(params);
}

// 获取IMU数据并对齐到两个LiDAR帧之间
void processIMU() {
    while (ros::ok()) {
        std::vector<sensor_msgs::Imu::ConstPtr> imuDataBetweenFrames;
        std::unique_lock<std::mutex> lock(mBuf);
        //todo 这里目前只考虑lidar里程计和imu信息，最后在把回环加入进来时再考虑点云信息
        cond_var.wait(lock, [] { return !odometryBuf.empty() && !imuBuf.empty(); });
        // 对齐两个LiDAR帧之间的IMU数据 把imuBuf分到imuBufAligned
        // 从队列中取出最新的两个激光帧
        if (odometryBuf.size() < 2) continue;
        auto lidarFrame1 = odometryBuf.front();
        odometryBuf.pop();
        if (odometryBuf.empty()) {
            // 等待下一个激光帧
            continue;
        }
        auto lidarFrame2 = odometryBuf.front();

        double startTime = lidarFrame1->header.stamp.toSec();
        double endTime = lidarFrame2->header.stamp.toSec();
        //todo 这里需要处理首尾部分的IMU数据，和lidar对齐，参考LIO-SAM
        while (!imuBuf.empty()) {
            auto imuData = imuBuf.front();
            double imuTime = imuData->header.stamp.toSec();
            if (imuTime >= startTime && imuTime <= endTime) {
                imuDataBetweenFrames.push_back(imuData);
                imuBuf.pop();
            } else if (imuTime > endTime) {
                break
            } else {
              imuBuf.pop();
            }
        }

        if (!imuDataBetweenFrames.empty()) {
          imuBufAligned.push_back(imuDataBetweenFrames);
          // 创建预积分器
          auto preintegrator = std::make_shared<gtsam::PreintegratedImuMeasurements>(initialImuPreintegratorParam, priorImuBias);

          // 进行预积分
          for (const auto& imuData : imuDataBetweenFrames) {
            // 提取IMU加速度和角速度
            // 假设IMU数据中包含必要的信息进行预积分
            gtsam::Vector3 accel(imuData->linear_acceleration.x,
                                 imuData->linear_acceleration.y,
                                 imuData->linear_acceleration.z);
            gtsam::Vector3 gyro(imuData->angular_velocity.x,
                                imuData->angular_velocity.y,
                                imuData->angular_velocity.z);
            //todo 这里lio-sam怎么给的
            double dt = 0.01;  // 假设IMU数据的时间间隔为0.01秒
            // 将IMU数据添加到预积分器
            preintegrator->integrateMeasurement(accel, gyro, dt);
          }
          // 将对齐后的IMU数据存入imuBufAligned
          imuBufAligned.push_back(imuDataBetweenFrames);
          // 将预积分器存入imuMeasurementsWindow
          imuMeasurementsWindow.push_back(preintegrator);
        }

        Frame thisFrame;
        // 滑窗内的位姿存储
        gtsam::Pose3 lidarPose = gtsam::Pose3(gtsam::Rot3::Quaternion(lidarFrame1->pose.pose.orientation.w,
                                             lidarFrame1->pose.pose.orientation.x,
                                             lidarFrame1->pose.pose.orientation.y,
                                             lidarFrame1->pose.pose.orientation.z),
                            Point3(lidarFrame1->pose.pose.position.x,
                                   lidarFrame1->pose.pose.position.y,
                                   lidarFrame1->pose.pose.position.z));
        thisFrame.pose = lidarPose;
        //todo 应该给多少 这里随便给，在process_lio中会利用IMU预积分器推导
        thisFrame.velocity = gtsam::Vector3(0, 0, 0);
        thisFrame.bias = gtsam::imuBias::ConstantBias();
        frameWindow.push_back(thisFrame);
        // 如果滑窗内的数据超过了指定大小，移除最旧的一帧
        if (frameWindow.size() > windowSize) {
            //todo 可以这样写吗？
            frameWindow.pop_front();
        }
        //todo 维护imu预积分器和imuBufAligned的大小
        if (imuMeasurementsWindow.size() > windowSize - 1) {
          imuMeasurementsWindow.pop_front();
        }
        if (imuBufAligned.size() > windowSize - 1) {
          imuBufAligned.pop_front();
        }

        //指向最新帧的索引
        key++;
        cond_var.notify_all();
    }
}

void resetOPtimization() {
  gtsam::ISAM2Params optParameters;
  optParameters.relinearizeThreshold = 0.1;
  optParameters.relinearizeSkip = 1;
  optimizer = gtsam::Optimizer(optParameters);

  gtsam::NonlinearFactorGraph newGraphFactors;
  graphFactors = newGraphFactors;

  gtsam::Values NewGraphValues;
  graphValues = newGraphValues;
}

//imu重新预积分
void imuRepropagate(gtsam::PreintegratedImuMeasurements &imuMeasurement, int i) {
       double lastImuTime_repropagate = -1;
       for (int j = 0; j < imuBufAligned[i].size()) {
         sensor_msgs::Imu thisImu = *imuBufAligned[i][j];
         //todo 确认一下
         double imuTime = ROS_TIME(thisImu.stamp);
         // 提取IMU加速度和角速度数据，进行预积分
         gtsam::Vector3 accel(thisImu.linear_acceleration.x,
                              thisImu.linear_acceleration.y,
                              thisImu.linear_acceleration.z);
         gtsam::Vector3 gyro(thisImu.angular_velocity.x,
                             thisImu.angular_velocity.y,
                             thisImu.angular_velocity.z);
         double dt = (lastImuTime_repropagate < 0) ? (1.0 / 500.0) : imuTime - lastImuTime_repropagate);
         imuMeasurementsWindow[i].integrateMeasurement(accel, gyro, dt);
         lastImuTime_repropagate = imuTime;
       }
       return ;
}

void slideWindow() {
    for (int i = 0; i < windowSize - 1; i++) {
        frameWindow[i] = frameWindow[i + 1];
    }
    for (int i = 0; i < windowSize - 2; i++) {
        imuMeasurementsWindow[i] = imuMeasurementsWindow[i + 1];
        imuBufAligned[i] = imuBufAligned[i + 1];
    }
    //把最后的都扔掉
    frameWindow.pop_back();
    imuMeasurementsWindow.pop_back();
    imuBufAligned.pop_back();
    return;
}


//先实现基本的LIO，然后把LOAMGBA内容融合进来
//维护滑窗并优化
void process_lio() {
    while (ros::ok()) {
        std::unique_lock<std::mutex> lock(mBuf);
        cond_var.wait(lock, [] { return frameWindow.size() == windowSize; });
        //获取当前激光帧位姿
        float p_x;
        float p_y;
        float p_z;
        float r_x;
        float r_y;
        float r_z;
        float r_w;
        //系统初始化
        if (systemInitialized == false) {
          //重置isam2优化器，清空大因子图
          resetOPtimization();
          //todo IMU信息对齐是否采用LIO-SAM模式
          //初始化第一帧作为先验，后续每帧不再是先验
          //添加里程计位姿先验因子 把第一帧的lidar位姿作为固定先验 以lidar位姿为核心
          p_x = frameWindow[0].pose.pose.pose.position.x;
          p_y = frameWindow[0].pose.pose.pose.position.y;
          p_z = frameWindow[0].pose.pose.pose.position.z;
          r_x = frameWindow[0].pose.pose.pose.orientation.x;
          r_y = frameWindow[0].pose.pose.pose.orientation.y;
          r_z = frameWindow[0].pose.pose.pose.orientation.z;
          r_w = frameWindow[0].pose.pose.pose.orientation.w;

          gtsam::Pose3 lidarPose0 = gtsam::Pose3(gtsam::Rot3::Quaternion(r_w, r_x, r_y, r_z), gtsam::Point3(p_x, p_y, p_z));
          gtsam::ProirFactor<gtsam::Pose3> priorPose(gtsam::Symbol('x', 0), lidarPose0, priorPoseNoise);
          graphFactors.add(priorPose);
          //添加速度先验因子 初始化速度定为0
          gtsam::PriorFactor<gtsam::Vector3> priorVel(gtsam::Symbol('v', 0), gtsam::Vector3(0, 0, 0), priorVelNoise);
          graphFactors.add(priorVel);
          //添加imu偏置先验因子
          gtsam::PriorFactor<gtsam::imuBias::ConstantBias> priorBias(gtsam::Symbol('b', 0), gtsam::imuBias::ConstantBias(), priorBiasNoise);
          graphFactors.add(priorBias);
          //变量节点赋初值
          graphValues.insert(gtsam::Symbol('x', 0), lidarPose0);
          graphValues.insert(gtsam::Symbol('v', 0), gtsam::Vector3(0, 0, 0));
          graphValues.insert(gtsam::Symbol('b', 0), gtsam::imuBias::ConstantBias());

          //把滑窗内剩余帧也加入到因子图中
          for (int i = 1; i < windowSize; i++) {
            //添加相对位姿
            p_x = frameWindow[i - 1].pose.pose.pose.position.x;
            p_y = frameWindow[i - 1].pose.pose.pose.position.y;
            p_z = frameWindow[i - 1].pose.pose.pose.position.z;
            r_x = frameWindow[i - 1].pose.pose.pose.orientation.x;
            r_y = frameWindow[i - 1].pose.pose.pose.orientation.y;
            r_z = frameWindow[i - 1].pose.pose.pose.orientation.z;
            r_w = frameWindow[i - 1].pose.pose.pose.orientation.w;
            gtsam::Pose3 lidarPose_prev = gtsam::Pose3(gtsam::Rot3::Quaternion(r_w, r_x, r_y, r_z), gtsam::Point3(p_x, p_y, p_z));

            p_x = frameWindow[i].pose.pose.pose.position.x;
            p_y = frameWindow[i].pose.pose.pose.position.y;
            p_z = frameWindow[i].pose.pose.pose.position.z;
            r_x = frameWindow[i].pose.pose.pose.orientation.x;
            r_y = frameWindow[i].pose.pose.pose.orientation.y;
            r_z = frameWindow[i].pose.pose.pose.orientation.z;
            r_w = frameWindow[i].pose.pose.pose.orientation.w;
            gtsam::Pose3 lidarPose_curr = gtsam::Pose3(gtsam::Rot3::Quaternion(r_w, r_x, r_y, r_z), gtsam::Point3(p_x, p_y, p_z));
            gtsam::Pose3 relPose = lidarPose_prev.between(lidarPose_curr);
            // 添加后端里程计因子，当前帧和上一帧的帧间约束
            graphFactors.add(gtsam::BetweenFactor<gtsam::Pose3>(gtsam::Symbol('x', i - 1), gtsam::Symbol('x', i), relPose, odomNoise));
            graphValues.insert(gtsam::Symbol('x', i), lidarPose_curr);

            //添加IMU预积分因子
            //上一帧位姿，上一帧速度，下一帧位姿，下一帧速度，上一帧帧偏置，预积分量
            gtsam::ImuFactor imu_factor(gtsam::Symbol('x', i - 1), gtsam::Symbol('v', i - 1), gtsam::Symbol('x', i), gtsam::Symbol('v', i), gtsam::Symbol('b', i - 1), imuMeasurementsWindow[i - 1]);
            graphFactors.add(imu_factor);
            //添加imu偏置因子，上一帧偏置，下一帧偏置，观测值为0（零偏尽量不动），噪声协方差（和IMU预积分器的积分时间相关）；deltaTij()是积分段的时间
            //todo 噪声处理一下
            graphFactors.add(gtsam::BetweenFactor<gtsam::imuBias::ConstantBias>(gtsam::Symbol('b', i - 1), gtsam::Symbol('b', i), gtsam::imuBias::ConstantBias(),
                         gtsam::noiseModel::Diagonal::Sigmas(sqrt(imuMeasurementsWindow[i - 1].deltaTij()) * noiseModelBetweenBias)));
            //用前一帧的状态、偏置，施加imu预计分量，得到当前帧的状态
            // 提取前一帧位姿、速度
            prevPose_  = frameWindow[i - 1].pose;
            prevVel_   = frameWindow[i - 1].velocity;
            // 获取前一帧imu偏置
            prevBias_ = frameWindow[i - 1].bias;
            // 获取前一帧状态
            prevState_ = gtsam::NavState(prevPose_, prevVel_);
            gtsam::NavState propState_ = imuMeasurementsWindow[i - 1].predict(prevState_, prevBias_);
            // 变量节点赋初值
            graphValues.insert(gtsam::Symbol('x', i), lidarPose_curr);
            graphValues.insert(gtsam::Symbol('v', i), propState_.v());
            graphValues.insert(gtsam::Symbol('b', i), prevBias_);
          }
          //todo 这里的key去掉 由填odomBuf的地方管理 key永远指向滑窗内最新的一帧，key用于gtsam因子图的索引
          key = 1;
          systemInitialized = true;
        }//end if
        else {
            //todo 先不考虑100帧重置isam优化器
            //创建因子图
            gtSAMgraph.resize(0);
            initialEstimate.clear();
            //先把滑窗内第一帧加到value中 但不作为先验约束
            p_x = frameWinow[0].pose.pose.pose.position.x;
            p_y = frameWinow[0].pose.pose.pose.position.y;
            p_z = frameWinow[0].pose.pose.pose.position.z;
            r_x = frameWinow[0].pose.pose.pose.orientation.x;
            r_y = frameWinow[0].pose.pose.pose.orientation.y;
            r_z = frameWinow[0].pose.pose.pose.orientation.z;
            r_w = frameWinow[0].pose.pose.pose.orientation.w;
            gtsam::Pose3 lidarPose0 = gtsam::Pose3(gtsam::Rot3::Quaternion(r_w, r_x, r_y, r_z), gtsam::Point3(p_x, p_y, p_z));
            graphValues.insert(gtsam::Symbol('x', key - windowSize + 1), lidarPose0);

            //添加相对位姿因子
            //todo 这里确认一下gtsam维护因子图的索引机制 参考LIO-SAM 每100帧重启一次gtsam优化器，前面的信息可以像LIO-SAM一样作为新的先验，也可以边缘化一下，边缘化是最正确合理的方式 gtsam或者自动边缘化，或者计算边缘化协方差后手动删除因子
            //这里的索引需要倒推，确认滑窗大小的一致性，算上最新帧有10帧，windowSize=10
            //把滑窗内所有帧加入到因子图中
            for (int i = 1; i < windowSize; i++) {
              //添加相对位姿
              p_x = frameWinow[i - 1].pose.pose.pose.position.x;
              p_y = frameWinow[i - 1].pose.pose.pose.position.y;
              p_z = frameWinow[i - 1].pose.pose.pose.position.z;
              r_x = frameWinow[i - 1].pose.pose.pose.orientation.x;
              r_y = frameWinow[i - 1].pose.pose.pose.orientation.y;
              r_z = frameWinow[i - 1].pose.pose.pose.orientation.z;
              r_w = frameWinow[i - 1].pose.pose.pose.orientation.w;
              gtsam::Pose3 lidarPose_prev = gtsam::Pose3(gtsam::Rot3::Quaternion(r_w, r_x, r_y, r_z), gtsam::Point3(p_x, p_y, p_z));

              p_x = frameWinow[i].pose.pose.pose.position.x;
              p_y = frameWinow[i].pose.pose.pose.position.y;
              p_z = frameWinow[i].pose.pose.pose.position.z;
              r_x = frameWinow[i].pose.pose.pose.orientation.x;
              r_y = frameWinow[i].pose.pose.pose.orientation.y;
              r_z = frameWinow[i].pose.pose.pose.orientation.z;
              r_w = frameWinow[i].pose.pose.pose.orientation.w;
              gtsam::Pose3 lidarPose_curr = gtsam::Pose3(gtsam::Rot3::Quaternion(r_w, r_x, r_y, r_z), gtsam::Point3(p_x, p_y, p_z));
              gtsam::Pose3 relPose = lidarPose_prev.between(lidarPose_curr);
              // 添加后端里程计因子，当前帧和上一帧的帧间约束
              gtSAMgraph.add(gtsam::BetweenFactor<gtsam::Pose3>(gtsam::Symbol('x', key - windowSize + i), gtsam::Symbol('x', key - windowSize + i + 1), relPose, odomNoise));

              //添加IMU预积分因子
              //上一帧位姿，上一帧速度，下一帧位姿，下一帧速度，上一帧帧偏置，预积分量
              gtsam::ImuFactor imu_factor(gtsam::Symbol('x', key - windowSize + i), gtsam::Symbol('v', key - windowSize + i), gtsam::Symbol('x', key - windowSize + i + 1), gtsam::Symbol('v', key - windowSize + i + 1), gtsam::Symbol('b', key - windowSize + i), imuMeasurementsWindow[i - 1]);
              graphFactors.add(imu_factor);
              //添加imu偏置因子，上一帧偏置，下一帧偏置，观测值(应该给多少)，噪声协方差；deltaTij()是积分段的时间
              //todo 噪声处理一下
              graphFactors.add(gtsam::BetweenFactor<gtsam::imuBias::ConstantBias>(gtsam::Symbol('b', key - windowSize + i), gtsam::Symbol('b', key - windowSize + i + 1), gtsam::imuBias::ConstantBias(),
                         gtsam::noiseModel::Diagonal::Sigmas(sqrt(imuMeasurementsWindow[i - 1].deltaTij()) * noiseModelBetweenBias)));
              // 提取前一帧位姿、速度
              prevPose_  = frameWindow[i - 1].pose;
              prevVel_   = frameWindow[i - 1].velocity;
              // 获取前一帧imu偏置
              prevBias_ = frameWindow[i - 1].bias;
              // 获取前一帧状态
              prevState_ = gtsam::NavState(prevPose_, prevVel_);
              gtsam::NavState propState_ = imuMeasurementsWindow[i - 1].predict(prevState_, prevBias_);
              // 变量节点赋初值
              graphValues.insert(gtsam::Symbol('x', key - windowSize + i + 1), lidarPose_curr);
              graphValues.insert(gtsam::Symbol('v', key - windowSize + i + 1), propState_.v());
              graphValues.insert(gtsam::Symbol('b', key - windowSize + i + 1), prevBias_);
            }//end 添加因子
        }//else end

          //优化
          optimizer.update(graphFactors, graphValues);
          optimizer.update();
          graphFactors.resize(0);
          graphValues.clear()

          //先把结果提取出来 放入frameWindow中
          gtsam::Values result = optimizer.calculateEstimate();
          for (int i = 1; i <= windowSize; i++) {
              frameWindow[i - 1].pose = result.at<gtsam::Pose3>(gtsam::Symbol('x', key - windowSize + i));
              frameWindow[i - 1].velocity = result.at<gtsam::Vector3>(gtsam::Symbol('v', key - windowSize + i));
              frameWindow[i - 1].bias = result.at<gtsam::imuBias::ConstantBias>(gtsam::Symbol('b', key - windowSize + i));
          }

          //优化之后重新计算预积分
          //重置优化之后的偏置
          for (int i = 0; i < windowSize - 1; i++) {
              //重置优化之后的偏置
              imuMeasurementsWindow[i].resetIntegrationAndSetBias(frameWindow[i].bias);
              //重新积分
              imuRepropagate(imuMeasurementsWindow[i], i);
          }

          //todo
          slideWindow();

        cond_var.notify_all();
    } //end while( ros::ok())
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

    // --------------------------------- 订阅后端数据 ---------------------------------
	// ros::Subscriber subCenters = nh.subscribe<sensor_msgs::PointCloud2>("/Center_BA", 100, centerHandler);
	ros::Subscriber subSurf = nh.subscribe<sensor_msgs::PointCloud2>("/ground_BA", 100, SurfHandler);
    ros::Subscriber subEdge = nh.subscribe<sensor_msgs::PointCloud2>("/Edge_BA", 100, EdgeHandler);

    ros::Subscriber subLaserOdometry = nh.subscribe<nav_msgs::Odometry>("/BALM_mapped_to_init", 100, laserOdometryHandler);
//	ros::Subscriber subGPS = nh.subscribe<sensor_msgs::NavSatFix>("/gps/fix", 100, gpsHandler);

    //订阅IMU数据
    ros::Subscriber subImu = nh.subscribe<sensor_msgs::Imu>("imu_raw", 2000, imuHandler, ros::TransportHints().tcpNoDelay());

    // ------------------------------------------------------------------
	pubOdomAftPGO = nh.advertise<nav_msgs::Odometry>("/aft_pgo_odom", 100);
	pubOdomRepubVerifier = nh.advertise<nav_msgs::Odometry>("/repub_odom", 100);
	pubPathAftPGO = nh.advertise<nav_msgs::Path>("/aft_pgo_path", 100);
	// 回环修正后的地图
    pubMapAftPGO = nh.advertise<sensor_msgs::PointCloud2>("/aft_pgo_map", 100);
	pubLoopScanLocal = nh.advertise<sensor_msgs::PointCloud2>("/loop_scan_local", 100);
	pubLoopSubmapLocal = nh.advertise<sensor_msgs::PointCloud2>("/loop_submap_local", 100);

    //threads说明：回调函数向队列中填入信息
    //todo 维护滑窗优化的因子 和真正执行优化的关系还需要考虑一下
//	std::thread posegraph_slam {process_pg}; // 后端里程计因子
    std::thread posegraph_slam {process_lio}; // LIO后端
    //todo 原本执行回环检测，构建完其他部分后再考虑怎么融合进去
//    std::thread lc_detection {process_lcd}; // loop closure detection
//    std::thread icp_calculation {process_icp}; // 后端回环因子
    //todo 原本为真正执行后端优化的部分，考虑和维护滑窗的线程的关系后决定是否要删除
//    std::thread isam_update {process_isam};
    //对齐imu和lidar，做imu预积分
    std::thread imuThread(processIMU);

    // isam2全局优化
    //暂时不要了
//	std::thread viz_map {process_viz_map}; // visualization - map (low frequency because it is heavy)
//	std::thread viz_path {process_viz_path}; // visualization - path (high frequency)

    // 运行ros节点执行ctrl+c后进程会转而执行ros::spin()后面的程序，但是如果在一定时间内程序没有执行完毕，进程会强制退出 
    // 将该文件中的_TIMEOUT_SIGINT = 15.0,15秒改为期望运行的最大时间。 sudo gedit /opt/ros/melodic/lib/python2.7/dist-packages/roslaunch/nodeprocess.py
 	ros::spin();
//    viz_map.join();
//    viz_path.join();
    posegraph_slam.join();
//    lc_detection.join();
//    icp_calculation.join();
//    isam_update.join();
    imuThread.join();
    sleep(10);
	return 0;
}
