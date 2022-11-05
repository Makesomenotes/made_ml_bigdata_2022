package utility

import breeze.linalg.{DenseMatrix, DenseVector, sum}
import breeze.numerics.{pow, sqrt}

class LinRegReg {
  def fit(X: DenseMatrix[Double], y: DenseVector[Double],
          n: Int, learnRate: Double): (DenseVector[Double], Double) = {
    var (weights, bias) = (DenseVector.zeros[Double](X.cols), .0)
    for (_ <- 0 to n) {
      val y_hat = (X * weights) + bias
      weights :-= learnRate * 2 * (X.t * (y_hat - y))
      weights = weights.map(el => el / X.rows)
      bias -= learnRate * 2 * sum(y_hat - y) / X.rows
    }
    (weights, bias)
  }

  def predict(X: DenseMatrix[Double], weights: DenseVector[Double],
              bias: Double): DenseVector[Double] = {
    val res = (X * weights)  + bias
    res
  }

  def RMSE(y_true: DenseVector[Double], y_pred: DenseVector[Double]): Double = {
    val error = sum((y_true - y_pred).map(el => pow(el, 2))) / y_true.length
    sqrt(error)
  }
}
