package org.apache.spark.ml.made

import breeze.linalg._
import breeze.stats.distributions.RandBasis
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.SparkSession

object Main extends App {
  val spark = SparkSession.builder
    .master(s"local[1]")
    .getOrCreate()

  import spark.sqlContext.implicits._

  val X = DenseMatrix.rand[Double](100000, 3)
  val y: DenseVector[Double] = X * DenseVector[Double](1.5, 0.3, -0.7)
  val data: DenseMatrix[Double] = DenseMatrix.horzcat(X, y.asDenseMatrix.t)

  val df = data(*, ::).iterator
    .map(x => (x(0), x(1), x(2), x(3)))
    .toSeq
    .toDF("x1", "x2", "x3", "y")

  val pipeline = new Pipeline().setStages(Array(
    new VectorAssembler()
      .setInputCols(Array("x1", "x2", "x3", "y"))
      .setOutputCol("features"),
    new LinearRegression().setInputCol("features").setOutputCol("label")
  ))

  val model = pipeline.fit(df)

 model.transform(df)
   .show(true)  // смотрим результат

  spark.stop()

}