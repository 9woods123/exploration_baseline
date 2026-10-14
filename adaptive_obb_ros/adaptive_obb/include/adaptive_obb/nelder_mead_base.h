#pragma once

#include <algorithm>
#include <boost/function.hpp>
#include <numeric>
#include <vector>

template <typename VertexType>
class NelderMeadBase {
  public:
  NelderMeadBase(int max_iter = 20, double sigma = 0.5, double rho = 0.5)
      : max_iter_(max_iter), sigma_(sigma), rho_(rho){};
  ~NelderMeadBase(){};

  void setMaxIter(double max_iter) { max_iter_ = max_iter; };

  void solve(const boost::function<double(const VertexType)> objFun, std::vector<VertexType>& simplex_points);

  virtual VertexType centroid(const std::vector<VertexType>& simplex_points);
  virtual VertexType reflection(const VertexType& x_g, const VertexType& x);
  virtual VertexType expansion(const VertexType& x_g, const VertexType& x_r, const VertexType& x);
  virtual VertexType contraction(const VertexType& x_g, const VertexType& x);
  virtual VertexType reduction(const VertexType& x_min, const VertexType& x_i);

  const boost::function<double(const VertexType)> objFun;

  private:
  // double alpha = 1; // alpha != 1 not supported
  // double gamma = 2; // gamma != 2 not supported
  double sigma_;
  double rho_;
  int max_iter_;
};

template <typename VertexType>
VertexType NelderMeadBase<VertexType>::centroid(const std::vector<VertexType>& simplex_points)
{
  auto iter_pre_last = std::prev(simplex_points.end(), 1);
  VertexType x_g = std::accumulate(simplex_points.begin(), iter_pre_last, VertexType());
  x_g = x_g / (simplex_points.size() - 1);
  return x_g;
}

template <typename VertexType>
VertexType NelderMeadBase<VertexType>::reflection(const VertexType& x_g, const VertexType& x)
{
  return x_g + (x_g - x);
}

template <typename VertexType>
VertexType NelderMeadBase<VertexType>::expansion(const VertexType& x_g, const VertexType& x_r,
                                                 const VertexType& x)
{
  return x_r + (x_g - x);
}

template <typename VertexType>
VertexType NelderMeadBase<VertexType>::contraction(const VertexType& x_g, const VertexType& x)
{
  return x_g + rho_ * (x - x_g);
}

template <typename VertexType>
VertexType NelderMeadBase<VertexType>::reduction(const VertexType& x_min, const VertexType& x_i)
{
  return x_min + sigma_ * (x_i - x_min);
}

template <typename VertexType>
void NelderMeadBase<VertexType>::solve(const boost::function<double(const VertexType)> objFun,
                                       std::vector<VertexType>& simplex_points)
{
  const int N_Simplex = simplex_points.size();
  static bool print = true;

  std::vector<VertexType> simplex_points_old = simplex_points;

  std::vector<double> simplex_fn_value(N_Simplex);
  std::vector<double> simplex_fn_value_old(N_Simplex);

  for (int iter = 0; iter < max_iter_; iter++) {
    // Step 1: Reordering
    for (int i = 0; i < N_Simplex; i++) {
      objFun(simplex_points[i]);
      simplex_fn_value[i] = objFun(simplex_points[i]);
    }

    std::vector<size_t> indexes(simplex_fn_value.size());
    iota(indexes.begin(), indexes.end(), 0);
    stable_sort(indexes.begin(), indexes.end(), [&simplex_fn_value](size_t i1, size_t i2) {
      return simplex_fn_value[i1] < simplex_fn_value[i2];
    });

    simplex_points_old = simplex_points;
    simplex_fn_value_old = simplex_fn_value;

    for (int i = 0; i < N_Simplex; i++) {
      simplex_points[i] = simplex_points_old[indexes[i]];
      simplex_fn_value[i] = simplex_fn_value_old[indexes[i]];
    }

    // Step 2: Compute centroid and reflection point
    VertexType x_g = centroid(simplex_points);
    VertexType x_r = reflection(x_g, simplex_points[N_Simplex - 1]);

    double f_r = objFun(x_r);
    double f_max = simplex_fn_value[N_Simplex - 1];
    double f_min = simplex_fn_value[0];

    VertexType x_max = simplex_points[N_Simplex - 1];
    VertexType x_min = simplex_points[0];

    if (f_r < simplex_fn_value[N_Simplex - 2]) {
      if (f_r >= f_min) {
        // Step 3: Reflection
        // reflected point is neither best nor worst in the new simplex, replace with
        simplex_points[N_Simplex - 1] = x_r;
      }
      else {
        // Step 4: Expansion
        // reflected point is better than the current best; try to go farther along this direction
        VertexType x_e = expansion(x_g, x_r, x_max);
        double f_e = objFun(x_e);

        if (f_e < f_r) {
          simplex_points[N_Simplex - 1] = x_e;
        }
        else {
          simplex_points[N_Simplex - 1] = x_r;
        }
      }
    }
    else {
      // Step 5: Contraction
      VertexType x_c = contraction(x_g, x_max);
      double f_c = objFun(x_c);

      if (f_c <= f_max) {
        simplex_points[N_Simplex - 1] = x_c;
      }
      else {
        // Step 6: Reduction
        for (int i = 1; i < N_Simplex; i++) {
          simplex_points[i] = reduction(x_min, simplex_points[i]);
        }
      }
    }
  }
}
