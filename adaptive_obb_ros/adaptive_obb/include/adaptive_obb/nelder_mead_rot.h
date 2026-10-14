#pragma once

#include "adaptive_obb/nelder_mead_base.h"

#include <eigen3/Eigen/Dense>

template <typename MatrixType>
class NelderMeadRot : public NelderMeadBase<MatrixType> {
  public:
  NelderMeadRot(int max_iter = 20, double sigma = 0.5, double rho = 0.5)
      : max_iter_(max_iter), sigma_(sigma), rho_(rho), NelderMeadBase<MatrixType>(max_iter, sigma, rho){};

  void setMaxIter(double max_iter) { max_iter_ = max_iter; };

  MatrixType centroid(const std::vector<MatrixType>& simplex_points);
  MatrixType reflection(const MatrixType& x_g, const MatrixType& x);
  MatrixType expansion(const MatrixType& x_g, const MatrixType& x_r, const MatrixType& x);
  MatrixType contraction(const MatrixType& x_g, const MatrixType& x);
  MatrixType reduction(const MatrixType& x_min, const MatrixType& x_i);

  private:
  // double alpha = 1; // alpha != 1 not supported
  // double gamma = 2; // gamma != 2 not supported
  double sigma_;
  double rho_;
  int max_iter_;
};

template <typename MatrixType>
MatrixType NelderMeadRot<MatrixType>::centroid(const std::vector<MatrixType>& simplex_points)
{
  auto iter_pre_last = std::prev(simplex_points.end(), 1);
  MatrixType x_g = std::accumulate(simplex_points.begin(), iter_pre_last, MatrixType::Zero().eval());
  x_g = x_g / (simplex_points.size() - 1.0);
  // QR decomposition
  // Since Q is unitary, matrix is right handed
  x_g = x_g.householderQr().householderQ();
  return x_g;
}

template <typename MatrixType>
MatrixType NelderMeadRot<MatrixType>::reflection(const MatrixType& x_g, const MatrixType& x)
{
  return x_g * x.transpose() * x_g;
}

template <typename MatrixType>
MatrixType NelderMeadRot<MatrixType>::expansion(const MatrixType& x_g, const MatrixType& x_r,
                                                const MatrixType& x)
{
  return x_g * x.transpose() * x_r;
}

template <typename MatrixType>
MatrixType NelderMeadRot<MatrixType>::contraction(const MatrixType& x_g, const MatrixType& x)
{
  // xc = x_g + rho*(x - x_g)
  MatrixType x_c = x_g + rho_ * (x - x_g);
  // QR decomposition
  // Since Q is unitary, matrix is right handed
  x_c = x_c.householderQr().householderQ();
  return x_c;
}

template <typename MatrixType>
MatrixType NelderMeadRot<MatrixType>::reduction(const MatrixType& x_min, const MatrixType& x_i)
{
  MatrixType x = x_min + sigma_ * (x_i - x_min);
  // QR decomposition
  // Since Q is unitary, matrix is right handed
  x = x.householderQr().householderQ();
  return x;
}
