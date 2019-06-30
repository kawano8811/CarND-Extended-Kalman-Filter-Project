#include "kalman_filter.h"
#include "tools.h"
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::cout;
using std::endl;

/*
 * Please note that the Eigen library does not initialize
 *   VectorXd or MatrixXd objects with zeros upon creation.
 */

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  /**
   * predict the state
   */
  x_ = F_ * x_;
  MatrixXd Ft = F_.transpose();
  P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  /**
   * update the state by using Kalman Filter equations
   */
  VectorXd y = z - (H_ * x_);
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;

  MatrixXd Si = S.inverse();
  MatrixXd PHt = P_ * Ht;
  MatrixXd K = PHt * Si;

  //new estimate
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /**
   * TODO: update the state by using Extended Kalman Filter equations
   */
  double px = x_(0);
  double py = x_(1);
  double vx = x_(2);
  double vy = x_(3);

  // calculate y
  double rho = sqrt(px * px + py * py);
  double phi = atan2(py, px);
  double rho_dot = 0.0;
  if (rho > 0.0001) {
    rho_dot = (px * vx + py * vy) / rho;
  }
  VectorXd hx = VectorXd(3);
  hx << rho,
        phi,
        rho_dot;
  VectorXd y = z - hx;
  // Normalizing Angles
  while (y(1) > M_PI) {
      y(1) -= M_PI;
  }
  while (y(1) < -M_PI) {
      y(1) += M_PI;
  }

  // Calculate Jacobian of H
  double rho_2 = z(0);
  double phi_2 = z(1);
  double rho_dot_2 = z(2);

  double px_2 = rho_2 * cos(phi_2);
	double py_2 = rho_2 * sin(phi_2);
	double vx_2 = rho_dot_2 * cos(phi_2);
	double vy_2 = rho_dot_2 * sin(phi_2);
  VectorXd x_2 = VectorXd(4);
  x_2 << px_2, py_2, vx_2, vy_2;

  Tools tools;
  MatrixXd Hj = tools.CalculateJacobian(x_2);
  MatrixXd Hjt = Hj.transpose();
  MatrixXd hjp = Hj * P_;
  MatrixXd S = Hj * P_ * Hjt + R_;

  MatrixXd Si = S.inverse();
  MatrixXd PHt = P_ * Hjt;
  MatrixXd K = PHt * Si;

  //new estimate
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * Hj) * P_;

}
