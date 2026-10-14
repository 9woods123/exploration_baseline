#pragma once

#include "adaptive_obb/nelder_mead_base.h"

#include <eigen3/Eigen/Dense>

template <typename VectorType>
class NelderMead : public NelderMeadBase<VectorType> {
  public:
  NelderMead(int max_iter = 20, double sigma = 0.5, double rho = 0.5)
      : max_iter_(max_iter), sigma_(sigma), rho_(rho), NelderMeadBase<VectorType>(max_iter, sigma, rho){};

  void setMaxIter(double max_iter) { max_iter_ = max_iter; };

  VectorType centroid(const std::vector<VectorType>& simplex_points);

  private:
  // double alpha = 1; // alpha != 1 not supported
  // double gamma = 2; // gamma != 2 not supported
  double sigma_;
  double rho_;
  int max_iter_;
};

template <typename VectorType>
VectorType NelderMead<VectorType>::centroid(const std::vector<VectorType>& simplex_points)
{
  auto iter_pre_last = std::prev(simplex_points.end(), 1);
  int N = simplex_points.size();
  VectorType x_g = std::accumulate(simplex_points.begin(), iter_pre_last, VectorType::Zero().eval());
  x_g = x_g / (simplex_points.size() - 1.0);
  return x_g;
}
