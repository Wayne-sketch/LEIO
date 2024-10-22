#include <ceres/ceres.h>
#include "lio_estimator.h"

#include <pcl/common/transforms.h>
#include <pcl/search/impl/flann_search.hpp>
#include <pcl/filters/extract_indices.h>


//LIO优化器
void SolveOptimization() {
  if (cir_buf_count_ < estimator_config_.window_size && estimator_config_.imu_factor) {
    LOG(ERROR) << "enter optimization before enough count: " << cir_buf_count_ << " < "
               << estimator_config_.window_size;
    return;
  }

  TicToc tic_toc_opt;

  bool turn_off = true;
//  Vector3d P_last0, P_last; /// for convergence check

  ceres::Problem problem;
  ceres::LossFunction *loss_function;
  // NOTE: indoor test
//  loss_function = new ceres::HuberLoss(0.5);
  loss_function = new ceres::CauchyLoss(1.0);

  // NOTE: update from laser transform
  if (estimator_config_.update_laser_imu) {
    DLOG(INFO) << "======= bef opt =======";

    if (!estimator_config_.imu_factor) {
      Twist<double>
          incre = (transform_lb_.inverse() * all_laser_transforms_[cir_buf_count_ - 1].second.transform.inverse()
          * all_laser_transforms_[cir_buf_count_].second.transform * transform_lb_).cast<double>();
      Ps_[cir_buf_count_] = Rs_[cir_buf_count_ - 1] * incre.pos + Ps_[cir_buf_count_ - 1];
      Rs_[cir_buf_count_] = Rs_[cir_buf_count_ - 1] * incre.rot;
    }

    Twist<double> transform_lb = transform_lb_.cast<double>();
    int pivot_idx = int(estimator_config_.window_size - estimator_config_.opt_window_size);

    Eigen::Vector3d Ps_pivot = Ps_[pivot_idx];
    Eigen::Vector3d Vs_pivot = Vs_[pivot_idx];
    Eigen::Matrix3d Rs_pivot = Rs_[pivot_idx];

    Quaterniond rot_pivot(Rs_pivot * transform_lb.rot.inverse());
    Eigen::Vector3d pos_pivot = Ps_pivot - rot_pivot * transform_lb.pos;

    Twist<double> transform_pivot = Twist<double>(rot_pivot, pos_pivot);

    vector<Transform> imu_poses, lidar_poses;

    for (int i = 0; i < estimator_config_.opt_window_size + 1; ++i) {
      int opt_i = int(estimator_config_.window_size - estimator_config_.opt_window_size + i);

      Quaterniond rot_li(Rs_[opt_i] * transform_lb.rot.inverse());
      Eigen::Vector3d pos_li = Ps_[opt_i] - rot_li * transform_lb.pos;
      Twist<double> transform_li = Twist<double>(rot_li, pos_li);

      // DLOG(INFO) << "Ps_[" << opt_i << "] bef: " << Ps_[opt_i].transpose();
      // DLOG(INFO) << "Vs_[" << opt_i << "]: bef " << Vs_[opt_i].transpose();
      /*
      DLOG(INFO) << "Vs_[" << opt_i << "]: " << Vs_[opt_i].transpose();
      DLOG(INFO) << "Rs_[" << opt_i << "]: " << Eigen::Quaterniond(Rs_[opt_i]).coeffs().transpose();
      DLOG(INFO) << "Bas_[" << opt_i << "]: " << Bas_[opt_i].transpose();
      DLOG(INFO) << "Bgs_[" << opt_i << "]: " << Bgs_[opt_i].transpose();
      */
      // DLOG(INFO) << "transform_lb_: " << transform_lb_;
      // DLOG(INFO) << "gravity in world: " << g_vec_.transpose();

      Twist<double> transform_bi = Twist<double>(Eigen::Quaterniond(Rs_[opt_i]), Ps_[opt_i]);
      imu_poses.push_back(transform_bi.cast<float>());
      lidar_poses.push_back(transform_li.cast<float>());
    }

    //region Check for imu res
//    for (int i = 0; i < estimator_config_.window_size; ++i) {
//
//      typedef Eigen::Matrix<double, 15, 15> M15;
//      typedef Eigen::Matrix<double, 15, 1> V15;
//      M15 sqrt_info =
//          Eigen::LLT<M15>(pre_integrations_[i + 1]->covariance_.inverse()).matrixL().transpose();
//
//      V15 res = (pre_integrations_[i + 1]->Evaluate(
//          Ps_[i], Eigen::Quaterniond(Rs_[i]), Vs_[i], Bas_[i], Bgs_[i + 1],
//          Ps_[i + 1], Eigen::Quaterniond(Rs_[i + 1]), Vs_[i + 1], Bas_[i + 1], Bgs_[i + 1]));
//      // DLOG(INFO) << "sqrt_info: " << endl << sqrt_info;
//
//      DLOG(INFO) << "imu res bef: " << res.transpose();
//      // DLOG(INFO) << "weighted pre: " << (sqrt_info * res).transpose();
//      // DLOG(INFO) << "weighted pre: " << (sqrt_info * res).squaredNorm();
//    }
    //endregion

    vis_bef_opt.UpdateMarkers(imu_poses, lidar_poses);
    vis_bef_opt.PublishMarkers();

    DLOG(INFO) << "====================================";
  }

  vector<FeaturePerFrame> feature_frames;

  BuildLocalMap(feature_frames);

  vector<double *> para_ids;

  //region Add pose and speed bias parameters
  for (int i = 0; i < estimator_config_.opt_window_size + 1;
       ++i) {
    ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
    problem.AddParameterBlock(para_pose_[i], SIZE_POSE, local_parameterization);
    problem.AddParameterBlock(para_speed_bias_[i], SIZE_SPEED_BIAS);
    para_ids.push_back(para_pose_[i]);
    para_ids.push_back(para_speed_bias_[i]);
  }
  //endregion

  //region Add extrinsic parameters
  {
    ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
    problem.AddParameterBlock(para_ex_pose_, SIZE_POSE, local_parameterization);
    para_ids.push_back(para_ex_pose_);
    if (extrinsic_stage_ == 0 || estimator_config_.opt_extrinsic == false) {
      DLOG(INFO) << "fix extrinsic param";
      problem.SetParameterBlockConstant(para_ex_pose_);
    } else {
      DLOG(INFO) << "estimate extrinsic param";
    }
  }
  //endregion

//  P_last0 = Ps_.last();

  VectorToDouble();

  vector<ceres::internal::ResidualBlock *> res_ids_marg;
  ceres::internal::ResidualBlock *res_id_marg = NULL;

  //region Marginalization residual
  if (estimator_config_.marginalization_factor) {
    if (last_marginalization_info) {
      // construct new marginlization_factor
      MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
      res_id_marg = problem.AddResidualBlock(marginalization_factor, NULL,
                                             last_marginalization_parameter_blocks);
      res_ids_marg.push_back(res_id_marg);
    }
  }
  //endregion

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

      ImuFactor *f = new ImuFactor(pre_integrations_[opt_j]);
//    {
//      double **tmp_parameters = new double *[5];
//      tmp_parameters[0] = para_pose_[i];
//      tmp_parameters[1] = para_speed_bias_[i];
//      tmp_parameters[2] = para_pose_[j];
//      tmp_parameters[3] = para_speed_bias_[j];
//      tmp_parameters[4] = para_qwi_;
//      f->Check(tmp_parameters);
//      delete[] tmp_parameters;
//    }

      // TODO: is it better to use g_vec_ as global parameter?
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

  vector<ceres::internal::ResidualBlock *> res_ids_proj;

  if (estimator_config_.point_distance_factor) {
    for (int i = 0; i < estimator_config_.opt_window_size + 1; ++i) {
      int opt_i = int(estimator_config_.window_size - estimator_config_.opt_window_size + i);

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

  if (estimator_config_.prior_factor) {
    {
      Twist<double> trans_tmp = transform_lb_.cast<double>();
      PriorFactor *f = new PriorFactor(trans_tmp.pos, trans_tmp.rot);
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

  DLOG(INFO) << "prepare for ceres: " << tic_toc_opt.Toc() << " ms";
  ROS_DEBUG_STREAM("prepare for ceres: " << tic_toc_opt.Toc() << " ms");

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

  ROS_DEBUG_STREAM("t_opt: " << t_opt.Toc() << " ms");
  DLOG(INFO) <<"t_opt: " << t_opt.Toc() << " ms";

  //region residual after optimization
  {
    ///< Aft
    double cost = 0.0;
    ceres::Problem::EvaluateOptions e_option;
    if (estimator_config_.imu_factor) {
      e_option.parameter_blocks = para_ids;
      e_option.residual_blocks = res_ids_pim;
      problem.Evaluate(e_option, &cost, NULL, NULL, NULL);
      DLOG(INFO) << "aft_pim: " << cost;
    }
    if (estimator_config_.point_distance_factor) {
      e_option.parameter_blocks = para_ids;
      e_option.residual_blocks = res_ids_proj;
      problem.Evaluate(e_option, &cost, NULL, NULL, NULL);
      DLOG(INFO) << "aft_proj: " << cost;
    }
    if (estimator_config_.marginalization_factor) {
      if (last_marginalization_info && !res_ids_marg.empty()) {
        e_option.parameter_blocks = para_ids;
        e_option.residual_blocks = res_ids_marg;
        problem.Evaluate(e_option, &cost, NULL, NULL, NULL);
        DLOG(INFO) << "aft_marg: " << cost;
      }
    }
  }
  //endregion

  // FIXME: Is marginalization needed in this framework? Yes, needed for extrinsic parameters.

  DoubleToVector();

//  P_last = Ps_.last();
//  if ((P_last - P_last0).norm() < 0.1) {
//    convergence_flag_ = true;
//  } else {
//    convergence_flag_ = false;
//    if (last_marginalization_info) {
//      delete last_marginalization_info;
//      last_marginalization_info = nullptr;
//    }
//  }

  //region Constraint Marginalization
  if (estimator_config_.marginalization_factor && !turn_off) {

    TicToc t_whole_marginalization;

    MarginalizationInfo *marginalization_info = new MarginalizationInfo();

//    {
//      MarginalizationInfo *marginalization_info0 = new MarginalizationInfo();
//      if (last_marginalization_info) {
//        vector<int> drop_set;
//        for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++) {
//          if (last_marginalization_parameter_blocks[i] == para_pose_[0] ||
//              last_marginalization_parameter_blocks[i] == para_speed_bias_[0])
//            drop_set.push_back(i);
//        }
//        // construct new marginlization_factor
//        MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
//        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
//                                                                       last_marginalization_parameter_blocks,
//                                                                       drop_set);
//
//        marginalization_info0->AddResidualBlockInfo(residual_block_info);
//      }
//
//      if (estimator_config_.imu_factor) {
//        int pivot_idx = estimator_config_.window_size - estimator_config_.opt_window_size;
//        if (pre_integrations_[pivot_idx + 1]->sum_dt_ < 10.0) {
//          ImuFactor *imu_factor = new ImuFactor(pre_integrations_[pivot_idx + 1]);
//          ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(imu_factor, NULL,
//                                                                         vector<double *>{para_pose_[0],
//                                                                                          para_speed_bias_[0],
//                                                                                          para_pose_[1],
//                                                                                          para_speed_bias_[1]},
//                                                                         vector<int>{0, 1});
//          marginalization_info0->AddResidualBlockInfo(residual_block_info);
//        }
//      }
//
//      if (estimator_config_.point_distance_factor) {
//        for (int i = 1; i < estimator_config_.opt_window_size + 1; ++i) {
//          int opt_i = int(estimator_config_.window_size - estimator_config_.opt_window_size + i);
//
//          FeaturePerFrame &feature_per_frame = feature_frames[opt_i];
//          LOG_ASSERT(opt_i == feature_per_frame.id);
//
//          vector<unique_ptr<Feature>> &features = feature_per_frame.features;
//
////        DLOG(INFO) << "features.size(): " << features.size();
//
//          for (int j = 0; j < features.size(); ++j) {
//
//            PointPlaneFeature feature_j;
//            features[j]->GetFeature(&feature_j);
//
//            const double &s = feature_j.score;
//
//            const Eigen::Vector3d &p_eigen = feature_j.point;
//            const Eigen::Vector4d &coeff_eigen = feature_j.coeffs;
//
//            PivotPointPlaneFactor *pivot_point_plane_factor = new PivotPointPlaneFactor(p_eigen,
//                                                                                        coeff_eigen);
//
//            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(pivot_point_plane_factor, loss_function,
//                                                                           vector<double *>{para_pose_[0],
//                                                                                            para_pose_[i],
//                                                                                            para_ex_pose_},
//                                                                           vector<int>{0});
//            marginalization_info0->AddResidualBlockInfo(residual_block_info);
//
//          }
//
//        }
//      }
//
//      TicToc t_pre_margin;
//      marginalization_info0->PreMarginalize();
//      DLOG(INFO) << "pre marginalization: " << t_pre_margin.Toc();
//
//      TicToc t_margin;
//      marginalization_info0->Marginalize();
//      DLOG(INFO) << "marginalization: " << t_margin.Toc();
//
//      {
//        std::unordered_map<long, double *> addr_shift2;
//        for (int i = 1; i < estimator_config_.opt_window_size + 1; ++i) {
//          addr_shift2[reinterpret_cast<long>(para_pose_[i])] = para_pose_[i];
//          addr_shift2[reinterpret_cast<long>(para_speed_bias_[i])] = para_speed_bias_[i];
//        }
//        addr_shift2[reinterpret_cast<long>(para_ex_pose_)] = para_ex_pose_;
//
//        vector<double *> parameter_blocks2 = marginalization_info0->GetParameterBlocks(addr_shift2);
//
//        vector<ceres::internal::ResidualBlock *> res_ids_marg2;
//        ceres::internal::ResidualBlock *res_id_marg2 = NULL;
////      ceres::Problem problem2;
//        MarginalizationFactor *marginalization_factor = new MarginalizationFactor(marginalization_info0);
//        res_id_marg2 = problem.AddResidualBlock(marginalization_factor, NULL,
//                                                parameter_blocks2);
//        res_ids_marg2.push_back(res_id_marg2);
//
//        double aft_cost_marg;
//        ceres::Problem::EvaluateOptions e_option;
//        e_option.parameter_blocks = para_ids;
//        e_option.residual_blocks = res_ids_marg2;
//        problem.Evaluate(e_option, &aft_cost_marg, NULL, NULL, NULL);
//        DLOG(INFO) << "bef_cost_marg: " << aft_cost_marg;
//      }
//
//      if (marginalization_info0)
//        delete marginalization_info0;
//    }

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

    if (estimator_config_.imu_factor) {
      int pivot_idx = estimator_config_.window_size - estimator_config_.opt_window_size;
      if (pre_integrations_[pivot_idx + 1]->sum_dt_ < 10.0) {
        ImuFactor *imu_factor = new ImuFactor(pre_integrations_[pivot_idx + 1]);
        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(imu_factor, NULL,
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

    DLOG(INFO) << "whole marginalization costs: " << t_whole_marginalization.Toc();
    ROS_DEBUG_STREAM("whole marginalization costs: " << t_whole_marginalization.Toc() << " ms");

//    {
//      std::unordered_map<long, double *> addr_shift2;
//      for (int i = 1; i < estimator_config_.opt_window_size + 1; ++i) {
//        addr_shift2[reinterpret_cast<long>(para_pose_[i])] = para_pose_[i];
//        addr_shift2[reinterpret_cast<long>(para_speed_bias_[i])] = para_speed_bias_[i];
//      }
//      addr_shift2[reinterpret_cast<long>(para_ex_pose_)] = para_ex_pose_;
//
//      vector<double *> parameter_blocks2 = marginalization_info->GetParameterBlocks(addr_shift2);
//
//      vector<ceres::internal::ResidualBlock *> res_ids_marg2;
//      ceres::internal::ResidualBlock *res_id_marg2 = NULL;
////      ceres::Problem problem2;
//      MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
//      res_id_marg2 = problem.AddResidualBlock(marginalization_factor, NULL,
//                                              parameter_blocks2);
//      res_ids_marg2.push_back(res_id_marg2);
//
//      double aft_cost_marg;
//      ceres::Problem::EvaluateOptions e_option;
//      e_option.parameter_blocks = para_ids;
//      e_option.residual_blocks = res_ids_marg2;
//      problem.Evaluate(e_option, &aft_cost_marg, NULL, NULL, NULL);
//      DLOG(INFO) << "aft_cost_marg: " << aft_cost_marg;
//    }

  }
  //endregion

  // NOTE: update to laser transform
  if (estimator_config_.update_laser_imu) {
    DLOG(INFO) << "======= aft opt =======";
    Twist<double> transform_lb = transform_lb_.cast<double>();
    Transform &opt_l0_transform = opt_transforms_[0];
    int opt_0 = int(estimator_config_.window_size - estimator_config_.opt_window_size + 0);
    Quaterniond rot_l0(Rs_[opt_0] * transform_lb.rot.conjugate().normalized());
    Eigen::Vector3d pos_l0 = Ps_[opt_0] - rot_l0 * transform_lb.pos;
    opt_l0_transform = Twist<double>{rot_l0, pos_l0}.cast<float>(); // for updating the map

    vector<Transform> imu_poses, lidar_poses;

    for (int i = 0; i < estimator_config_.opt_window_size + 1; ++i) {
      int opt_i = int(estimator_config_.window_size - estimator_config_.opt_window_size + i);

      Quaterniond rot_li(Rs_[opt_i] * transform_lb.rot.conjugate().normalized());
      Eigen::Vector3d pos_li = Ps_[opt_i] - rot_li * transform_lb.pos;
      Twist<double> transform_li = Twist<double>(rot_li, pos_li);

      // DLOG(INFO) << "Ps_[" << opt_i << "]: " << Ps_[opt_i].transpose();
      // DLOG(INFO) << "Vs_[" << opt_i << "]: " << Vs_[opt_i].transpose();
      /*
      DLOG(INFO) << "Vs_[" << opt_i << "]: " << Vs_[opt_i].transpose();
      DLOG(INFO) << "Rs_[" << opt_i << "]: " << Eigen::Quaterniond(Rs_[opt_i]).coeffs().transpose();
      DLOG(INFO) << "Bas_[" << opt_i << "]: " << Bas_[opt_i].transpose();
      DLOG(INFO) << "Bgs_[" << opt_i << "]: " << Bgs_[opt_i].transpose();
      */
//      DLOG(INFO) << "velocity: " << Vs_.last().norm();
//      DLOG(INFO) << "transform_lb_: " << transform_lb_;
      // DLOG(INFO) << "gravity in world: " << g_vec_.transpose();

      Twist<double> transform_bi = Twist<double>(Eigen::Quaterniond(Rs_[opt_i]), Ps_[opt_i]);
      imu_poses.push_back(transform_bi.cast<float>());
      lidar_poses.push_back(transform_li.cast<float>());

    }

    DLOG(INFO) << "velocity: " << Vs_.last().norm();
    DLOG(INFO) << "transform_lb_: " << transform_lb_;

    ROS_DEBUG_STREAM("lb in world: " << (rot_l0.normalized() * transform_lb.pos).transpose());

    //region Check for imu res
//    for (int i = 0; i < estimator_config_.window_size; ++i) {
//
//      typedef Eigen::Matrix<double, 15, 15> M15;
//      typedef Eigen::Matrix<double, 15, 1> V15;
//      M15 sqrt_info =
//          Eigen::LLT<M15>(pre_integrations_[i + 1]->covariance_.inverse()).matrixL().transpose();
//
//      V15 res = (pre_integrations_[i + 1]->Evaluate(
//          Ps_[i], Eigen::Quaterniond(Rs_[i]), Vs_[i], Bas_[i], Bgs_[i + 1],
//          Ps_[i + 1], Eigen::Quaterniond(Rs_[i + 1]), Vs_[i + 1], Bas_[i + 1], Bgs_[i + 1]));
//      // DLOG(INFO) << "sqrt_info: " << endl << sqrt_info;
//
//      DLOG(INFO) << "imu res aft: " << res.transpose();
//      // DLOG(INFO) << "weighted pre: " << (sqrt_info * res).transpose();
//      // DLOG(INFO) << "weighted pre: " << (sqrt_info * res).squaredNorm();
//    }
    //endregion

    vis_aft_opt.UpdateMarkers(imu_poses, lidar_poses);
    vis_aft_opt.UpdateVelocity(Vs_.last().norm());
    vis_aft_opt.PublishMarkers();

    {
      geometry_msgs::PoseStamped ex_lb_msg;
      ex_lb_msg.header = Headers_.last();
      ex_lb_msg.pose.position.x = transform_lb.pos.x();
      ex_lb_msg.pose.position.y = transform_lb.pos.y();
      ex_lb_msg.pose.position.z = transform_lb.pos.z();
      ex_lb_msg.pose.orientation.w = transform_lb.rot.w();
      ex_lb_msg.pose.orientation.x = transform_lb.rot.x();
      ex_lb_msg.pose.orientation.y = transform_lb.rot.y();
      ex_lb_msg.pose.orientation.z = transform_lb.rot.z();
      pub_extrinsic_.publish(ex_lb_msg);

      int pivot_idx = estimator_config_.window_size - estimator_config_.opt_window_size;

      Eigen::Vector3d Ps_pivot = Ps_[pivot_idx];
      Eigen::Matrix3d Rs_pivot = Rs_[pivot_idx];

      Quaterniond rot_pivot(Rs_pivot * transform_lb.rot.inverse());
      Eigen::Vector3d pos_pivot = Ps_pivot - rot_pivot * transform_lb.pos;
      PublishCloudMsg(pub_local_surf_points_,
                      *surf_stack_[pivot_idx + 1],
                      Headers_[pivot_idx + 1].stamp,
                      "/laser_local");

      PublishCloudMsg(pub_local_corner_points_,
                      *corner_stack_[pivot_idx + 1],
                      Headers_[pivot_idx + 1].stamp,
                      "/laser_local");

      PublishCloudMsg(pub_local_full_points_,
                      *full_stack_[pivot_idx + 1],
                      Headers_[pivot_idx + 1].stamp,
                      "/laser_local");

      PublishCloudMsg(pub_map_surf_points_,
                      *local_surf_points_filtered_ptr_,
                      Headers_.last().stamp,
                      "/laser_local");

#ifdef USE_CORNER
      PublishCloudMsg(pub_map_corner_points_,
                      *local_corner_points_filtered_ptr_,
                      Headers_.last().stamp,
                      "/laser_local");
#endif

      laser_local_trans_.setOrigin(tf::Vector3{pos_pivot.x(), pos_pivot.y(), pos_pivot.z()});
      laser_local_trans_.setRotation(tf::Quaternion{rot_pivot.x(), rot_pivot.y(), rot_pivot.z(), rot_pivot.w()});
      laser_local_trans_.stamp_ = Headers_.last().stamp;
      tf_broadcaster_est_.sendTransform(laser_local_trans_);

      Eigen::Vector3d Ps_last = Ps_.last();
      Eigen::Matrix3d Rs_last = Rs_.last();

      Quaterniond rot_last(Rs_last * transform_lb.rot.inverse());
      Eigen::Vector3d pos_last = Ps_last - rot_last * transform_lb.pos;

      Quaterniond rot_predict = (rot_pivot.inverse() * rot_last).normalized();
      Eigen::Vector3d pos_predict = rot_pivot.inverse() * (Ps_last - Ps_pivot);

      PublishCloudMsg(pub_predict_surf_points_, *(surf_stack_.last()), Headers_.last().stamp, "/laser_predict");
      PublishCloudMsg(pub_predict_full_points_, *(full_stack_.last()), Headers_.last().stamp, "/laser_predict");

      {
        // NOTE: full stack into end of the scan
//        PointCloudPtr tmp_points_ptr = boost::make_shared<PointCloud>(PointCloud());
//        *tmp_points_ptr = *(full_stack_.last());
//        TransformToEnd(tmp_points_ptr, transform_es_, 10);
//        PublishCloudMsg(pub_predict_corrected_full_points_,
//                        *tmp_points_ptr,
//                        Headers_.last().stamp,
//                        "/laser_predict");

        TransformToEnd(full_stack_.last(), transform_es_, 10, true);
        PublishCloudMsg(pub_predict_corrected_full_points_,
                        *(full_stack_.last()),
                        Headers_.last().stamp,
                        "/laser_predict");
      }

#ifdef USE_CORNER
      PublishCloudMsg(pub_predict_corner_points_, *(corner_stack_.last()), Headers_.last().stamp, "/laser_predict");
#endif
      laser_predict_trans_.setOrigin(tf::Vector3{pos_predict.x(), pos_predict.y(), pos_predict.z()});
      laser_predict_trans_.setRotation(tf::Quaternion{rot_predict.x(), rot_predict.y(), rot_predict.z(),
                                                      rot_predict.w()});
      laser_predict_trans_.stamp_ = Headers_.last().stamp;
      tf_broadcaster_est_.sendTransform(laser_predict_trans_);
    }

  }

  DLOG(INFO) << "tic_toc_opt: " << tic_toc_opt.Toc() << " ms";
  ROS_DEBUG_STREAM("tic_toc_opt: " << tic_toc_opt.Toc() << " ms");

}

void BALMOdometryHandler() {


  return ;
}


int main(int argc, char** argv)
{
  ros::init(argc, argv, "lio_estimator");
  ros::NodeHandle nh;

  //todo 订阅imu原始数据做预积分 让BALM发布面点+里程计位姿，用同一个时间戳 这里把所有信息做一次对齐 BALM里程计作为滑窗点云初始位姿 然后LIO滑窗紧耦合输出最新帧位姿
  ros::Subscriber subBALMOdometry = nh.subscribe<nav_msgs::Odometry>("/BALM_mapped_to_init", 100, BALMOdometryHandler);
  ros::Subscriber subImu = nh.subscribe<sensor_msgs::PointCloud2>("/imu_raw", 100, imuHandler);
  ros::Subscriber subBALMSurfPoints = nh.subscribe<sensor_msgs::PointCloud2>("/BALM_surf_points", 100, laserCloudHandler);

  ros::spin();

  return 0;
}
