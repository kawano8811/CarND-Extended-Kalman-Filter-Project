#include "tools.h"
#include <iostream>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;
using std::cout;
using std::endl;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
   * Calculate the RMSE here.
   */
  VectorXd rmse(4);
  rmse << 0,0,0,0;

  // check the validity of the following inputs:
  //  * the estimation vector size should not be zero
  //  * the estimation vector size should equal ground truth vector size
  if (estimations.size() != ground_truth.size()
      || estimations.size() == 0) {
    cout << "Invalid estimation or ground_truth data" << endl;
    return rmse;
  }

  // accumulate squared residuals
  for (unsigned int i=0; i < estimations.size(); ++i) {

    VectorXd residual = estimations[i] - ground_truth[i];

    // coefficient-wise multiplication
    residual = residual.array()*residual.array();
    rmse += residual;
  }

  // calculate the mean
  rmse = rmse/estimations.size();

  // calculate the squared root
  rmse = rmse.array().sqrt();

  // return the result
  return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  /**
   * Calculate a Jacobian
   */
  MatrixXd Hj(3,4);
  // recover state parameters
  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);

  // pre-compute a set of terms to avoid repeated calculation
  float c1 = px * px + py * py;
  float c2 = sqrt(c1);
  float c3 = pow(c1, 3/2);

  // check division by zero
  if (fabs(c1) < 0.0001) {
    cout << "CalculateJacobian () - Error - Division by Zero" << endl;
    return Hj;
  }

  // compute the Jacobian matrix
  Hj << (px/c2), (py/c2), 0, 0,
      -(py/c1), (px/c1), 0, 0,
      py * (vx * py - vy * px) / c3, px * (px * vy - py * vx) / c3, px / c2, py / c2;

  return Hj;
}

MatrixXd Tools::CalculateHx(const VectorXd& x_state) {
  /**
   * Calculate a Hx matrix
   */
  double px = x_state(0);
  double py = x_state(1);
  double vx = x_state(2);
  double vy = x_state(3);

  // calculate y
  double rho = sqrt(px * px + py * py);
  double phi = atan2(py, px);
  double rho_dot = 0.0;
  if (rho > 0.0001) {
    rho_dot = (px * vx + py * vy) / rho;
  }
  VectorXd hx = VectorXd(3);
  hx << rho, phi, rho_dot;
  return hx;
}

VectorXd Tools::PolarToCartesian(const VectorXd& polar) {
  /**
   * Convert cartesian to polar
   */
  double rho = polar(0);
  double phi = polar(1);
  double rho_dot = polar(2);

  double px = rho * cos(phi);
  double py = rho * sin(phi);
  double vx = rho_dot * cos(phi);
  double vy = rho_dot * sin(phi);
  VectorXd cartesian = VectorXd(4);
  cartesian << px, py, vx, vy;
  return cartesian;
}
