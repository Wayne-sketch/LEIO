#pragma once
#ifndef UTILITY_H
#define UTILITY_H
#include <ros/ros.h>

#include <std_msgs/Header.h>
#include <std_msgs/Float64MultiArray.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/NavSatFix.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>

#include <opencv2/opencv.hpp>

#include <vector>
#include <cmath>
#include <algorithm>
#include <queue>
#include <deque>
#include <iostream>
#include <fstream>
#include <ctime>
#include <cfloat>
#include <iterator>
#include <sstream>
#include <string>
#include <limits>
#include <iomanip>
#include <array>
#include <thread>
#include <mutex>

using namespace std;

class ParamServer
{
public:

  ros::NodeHandle nh;

  string imuTopic;

  // IMU参数
  float imuAccNoise;          // 加速度噪声标准差
  float imuGyrNoise;          // 角速度噪声标准差
  float imuAccBiasN;          //
  float imuGyrBiasN;
  float imuGravity;           // 重力加速度
  float imuRPYWeight;
  vector<double> extRotV;
  vector<double> extRPYV;
  vector<double> extTransV;
  Eigen::Matrix3d extRot;     // xyz坐标系旋转
  Eigen::Matrix3d extRPY;     // RPY欧拉角的变换关系
  Eigen::Vector3d extTrans;   // xyz坐标系平移
  Eigen::Quaterniond extQRPY;


  ParamServer()
  {
    nh.param<std::string>("lio_sam/imuTopic", imuTopic, "imu_correct");

    nh.param<float>("lio_sam/imuAccNoise", imuAccNoise, 0.01);
    nh.param<float>("lio_sam/imuGyrNoise", imuGyrNoise, 0.001);
    nh.param<float>("lio_sam/imuAccBiasN", imuAccBiasN, 0.0002);
    nh.param<float>("lio_sam/imuGyrBiasN", imuGyrBiasN, 0.00003);
    nh.param<float>("lio_sam/imuGravity", imuGravity, 9.80511);
    nh.param<float>("lio_sam/imuRPYWeight", imuRPYWeight, 0.01);
    nh.param<vector<double>>("lio_sam/extrinsicRot", extRotV, vector<double>());
    nh.param<vector<double>>("lio_sam/extrinsicRPY", extRPYV, vector<double>());
    nh.param<vector<double>>("lio_sam/extrinsicTrans", extTransV, vector<double>());
    extRot = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(extRotV.data(), 3, 3);
    extRPY = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(extRPYV.data(), 3, 3);
    extTrans = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(extTransV.data(), 3, 1);
    extQRPY = Eigen::Quaterniond(extRPY).inverse();

    usleep(100);
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




};


















#endif //UTILITY_H
