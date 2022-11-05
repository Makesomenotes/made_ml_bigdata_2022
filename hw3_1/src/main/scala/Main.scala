import breeze.linalg.{DenseMatrix, DenseVector, csvread, csvwrite}
import utility.LinRegReg

import java.io.File


object Main {
  def main(args: Array[String]): Unit = {
    println("start")
    val data_path = "/Users/maksimgritskikh/MADE/made_ml_bigdata_2022/hw3_1/data/diabetes_data.csv"
    val target_path = "/Users/maksimgritskikh/MADE/made_ml_bigdata_2022/hw3_1/data/diabetes_target.csv"
    val out_path = "/Users/maksimgritskikh/MADE/made_ml_bigdata_2022/hw3_1/data/pred.csv"

    val model = new LinRegReg
    val data = csvread(file=new File(data_path), separator=',')
    val target = csvread(file=new File(target_path), separator=',').toDenseVector
    val n = target.length
    val m = data.cols
    val train_part = (n * 0.8).toInt
    println((n, train_part))
    val (yTrain, yTest) = (target(0 until train_part), target(train_part until n))
    val (xTrain, xTest) = (data(0 until train_part, ::).toDenseMatrix, data(train_part until n, ::).toDenseMatrix)
    println(s"Data processed")
    val (weights, bias) = model.fit(xTrain, yTrain, 1000, 0.01)
    println(s"Model fitted")
    val y_pred = model.predict(xTest, weights, bias)
    csvwrite(new File(out_path), separator = ',', mat = y_pred.toDenseMatrix)
    println(s"Got predictions. RMSE on validation dataset: ${model.RMSE(yTest, y_pred)}")
    println(s"All done. Predictions are saved to: ${out_path}")
  }
}
