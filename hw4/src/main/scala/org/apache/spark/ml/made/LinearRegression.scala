package org.apache.spark.ml.made

import org.apache.spark.ml.attribute.AttributeGroup
import org.apache.spark.ml.linalg.{DenseVector, Vector, VectorUDT, Vectors}
import org.apache.spark.ml.param.{DoubleParam, IntParam, ParamMap}
import org.apache.spark.ml.param.shared.{HasInputCol, HasOutputCol}
import org.apache.spark.ml.util._
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset, Encoder}



trait LinearRegressionParams 
  extends HasInputCol 
  with HasOutputCol {
  def setInputCol(value: String) : this.type = set(inputCol, value)
  def setOutputCol(value: String): this.type = set(outputCol, value)

  val iters = new IntParam(this, "iters", "Number of iterations")

  def get_iters: Int = $(iters)
  def set_iters(value: Int): this.type = set(iters, value)
  val lr = new DoubleParam(this, "lr", "step size")
  def get_lr: Double = $(lr)
  def set_lr(value: Double): this.type = set(lr, value)

  setDefault(iters, 100000)
  setDefault(lr, 1e-3)

  protected def validateAndTransformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, getInputCol, new VectorUDT())

    if (schema.fieldNames.contains($(outputCol))) {
      SchemaUtils.checkColumnType(schema, getOutputCol, new VectorUDT())
      schema
    } else {
      SchemaUtils.appendColumn(schema, schema(getInputCol).copy(name = getOutputCol))
    }
  }
}


class LinearRegression(override val uid: String) 
  extends Estimator[LinearRegressionModel] 
  with LinearRegressionParams
  with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("linearRegression"))

  override def fit(dataset: Dataset[_]): LinearRegressionModel = {

    implicit val encoder : Encoder[Vector] = ExpressionEncoder()
    val vectors: Dataset[Vector] = dataset.select(dataset($(inputCol)).as[Vector])
    val dim: Int = AttributeGroup.fromStructField((dataset.schema($(inputCol)))).numAttributes.getOrElse(
      vectors.first().size
    )
    val weights = 2.0 * breeze.linalg.DenseVector.rand[Double](dim) - 1.0 // breeze.linalg.DenseVector(0.0, 0.0, 0.0, 0.0)//

    val iters: Int = get_iters
    val lr: Double = get_lr
    val count: Double = vectors.count()
    val mean_lr Double = lr / count

    var i: Int = 0
    for (i <- 0 to iters) {
      val loss = features.rdd.mapPartitions((data: Iterator[Vector]) => {
        val summarizer = new MultivariateOnlineSummarizer()
        data.foreach(v => { val X = v.asBreeze(0 until weights.size).toDenseVector
          val y = v.asBreeze(-1)
          val grads = X * (breeze.linalg.sum(X * weights) - y)
          summarizer.add(mllib.linalg.Vectors.fromBreeze(grads))
        })
        Iterator(summarizer)
      }).reduce(_ merge _)
      weights = weights - mean_lr * loss.mean.asBreeze
    }
    
    copyValues(new LinearRegressionModel(Vectors.fromBreeze(weights))).setParent(this)
  }

  override def copy(extra: ParamMap): Estimator[LinearRegressionModel] = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

}

object LinearRegression extends DefaultParamsReadable[LinearRegression]

class LinearRegressionModel private[made](
                                         override val uid: String,
                                         val stds: DenseVector) extends Model[LinearRegressionModel] 
   with LinearRegressionParams with MLWritable {


  def this(stds: Vector) =
    this(Identifiable.randomUID("linearRegressionModel"), stds.toDense)

  override def copy(extra: ParamMap): LinearRegressionModel = copyValues(
    new LinearRegressionModel(stds: DenseVector), extra)

  override def transform(dataset: Dataset[_]): DataFrame = {
    val breezeCoeff = stds.asBreeze.toDenseVector
    val coeff = Vectors.fromBreeze(breezeCoeff(0 to breezeCoeff.length - 1))

    val transformUdf =
      dataset.sqlContext.udf.register(
        uid + "_transform",
        (x: Vector) => {
          x.dot(coeff)
        }
      )

    dataset.withColumn($(outputCol), transformUdf(dataset($(inputCol))))
  }

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

  override def write: MLWriter = new DefaultParamsWriter(this) {
    override protected def saveImpl(path: String): Unit = {
      super.saveImpl(path)
    }
  }
}

object LinearRegressionModel extends MLReadable[LinearRegressionModel] {
  override def read: MLReader[LinearRegressionModel] = new MLReader[LinearRegressionModel] {
    override def load(path: String): LinearRegressionModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc)

      val vectors = sqlContext.read.parquet(path + "/vectors")

      // Used to convert untyped dataframes to datasets with vectors
      implicit val encoder : Encoder[Vector] = ExpressionEncoder()

      val std =  vectors.select(vectors("_1").as[Vector]).first()

      val model = new LinearRegressionModel(std)
      metadata.getAndSetParams(model)
      model
    }
  }
}