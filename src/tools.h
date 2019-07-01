#ifndef TOOLS_H_
#define TOOLS_H_

#include <vector>
#include "Eigen/Dense"

class Tools {
 public:
  /**
   * Constructor.
   */
  Tools();

  /**
   * Destructor.
   */
  virtual ~Tools();

  /**
   * A helper method to calculate RMSE.
   */
  Eigen::VectorXd CalculateRMSE(const std::vector<Eigen::VectorXd> &estimations,
                                const std::vector<Eigen::VectorXd> &ground_truth);

  /**
   * A helper method to calculate Jacobians.
   */
  Eigen::MatrixXd CalculateJacobian(const Eigen::VectorXd& x_state);

  /**
   * A helper method to calculate Hx matrix.
   */
  Eigen::MatrixXd CalculateHx(const Eigen::VectorXd& x_state);

  /**
   * Convert cartesian to polar
   */
  Eigen::VectorXd PolarToCartesian(const Eigen::VectorXd& polar);

};

#endif  // TOOLS_H_
