package org.apache.spark.ml.made

import org.apache.spark.ml.attribute.AttributeGroup
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.{DenseVector, Vector, VectorUDT, Vectors}
import org.apache.spark.ml.param.{DoubleParam, IntParam, ParamMap}
import org.apache.spark.ml.param.shared.{HasFeaturesCol, HasLabelCol, HasOutputCol}
import org.apache.spark.ml.util._
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.mllib
import org.apache.spark.mllib.stat.MultivariateOnlineSummarizer
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.types.{DoubleType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, Encoder}
import org.apache.spark.sql.functions.lit


trait LinearRegressionParams extends HasFeaturesCol with HasLabelCol with HasOutputCol {
  def setFeatureCol(value: String) : this.type = set(featuresCol, value)
  def setLabelCol(value: String): this.type = set(labelCol, value)
  def setOutputCol(value: String): this.type = set(outputCol, value)

  val iters = new IntParam(
    this, "iters","Number of iterations"
  )
  def getIters: Int = $(iters)
  def setIters(value: Int): this.type = set(iters, value)

  val lr = new DoubleParam(
    this, "lr","learning rate")
  def getLr : Double = $(lr)
  def setLr(value: Double) : this.type = set(lr, value)

  setDefault(iters -> 100)
  setDefault(lr -> 0.001)

  protected def validateAndTransformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, getFeaturesCol, new VectorUDT())
    SchemaUtils.checkColumnType(schema, getLabelCol, DoubleType)

    if (schema.fieldNames.contains($(outputCol))) {
      SchemaUtils.checkColumnType(schema, getOutputCol, DoubleType)
      schema
    } else {
      SchemaUtils.appendColumn(schema, StructField(getOutputCol, DoubleType))
    }
  }
}

class LinearRegression(override val uid: String) extends Estimator[LinearRegressionModel] with LinearRegressionParams
  with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("LinearRegression"))

  override def fit(dataset: Dataset[_]): LinearRegressionModel = {

    // Used to convert untyped dataframes to datasets with vectors
    implicit val encoder : Encoder[Vector] = ExpressionEncoder()

    val data = dataset.select(
      lit(1.0).as("bias"),
      dataset($(featuresCol)),
      dataset($(labelCol))
    )

    val assembler = new VectorAssembler()
      .setInputCols(Array("bias", $(featuresCol), $(labelCol)))
      .setOutputCol("concatenation")
    val assembledData = assembler.transform(data)

    val vectors: Dataset[Vector] = assembledData.select(assembledData("concatenation").as[Vector])

    val dim: Int = AttributeGroup.fromStructField(assembledData.schema("concatenation")).numAttributes.getOrElse(
      vectors.first().size
    )

    var WeightsBias = Vectors.zeros(dim - 1).asBreeze.toDenseVector

    for (_ <- 1 to getIters) {
      val summary = vectors.rdd.mapPartitions((data: Iterator[Vector]) => {
        val result = data.foldLeft(new MultivariateOnlineSummarizer())(
          (summarizer, vector) => summarizer.add(
            {
              val x = vector.asBreeze(0 until dim - 1).toDenseVector
              val y = vector.asBreeze(dim - 1)
              mllib.linalg.Vectors.fromBreeze(2.0 * x * ((x dot WeightsBias) - y))
            }
          ))
        Iterator(result)
      }).reduce(_ merge _)

      WeightsBias = WeightsBias - summary.mean.asML.asBreeze.toDenseVector * $(lr)
    }

    val weights = Vectors.fromBreeze(WeightsBias(1 until WeightsBias.size))
    val bias = WeightsBias(0)

    copyValues(new LinearRegressionModel(
      weights,
      bias
    )
    ).setParent(this)

  }

  override def copy(extra: ParamMap): Estimator[LinearRegressionModel] = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

}

object LinearRegression extends DefaultParamsReadable[LinearRegression]

class LinearRegressionModel private[made](
                                           override val uid: String,
                                           val weights: DenseVector,
                                           val bias: Double
                                         ) extends Model[LinearRegressionModel] with LinearRegressionParams with MLWritable {


  private[made] def this(weights: Vector, bias: Double) =
    this(Identifiable.randomUID("LinearRegressionModel"), weights.toDense, bias)

  override def copy(extra: ParamMap): LinearRegressionModel = copyValues(
    new LinearRegressionModel(weights, bias), extra)

  override def transform(dataset: Dataset[_]): DataFrame = {
    val transformUdf = dataset.sqlContext.udf.register(uid + "_transform",
      (x : Vector) => {
        x.asBreeze.dot(weights.asBreeze) + bias
      })

    dataset.withColumn($(outputCol), transformUdf(dataset($(featuresCol))))
  }

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

  override def write: MLWriter = new DefaultParamsWriter(this) {
    override protected def saveImpl(path: String): Unit = {
      super.saveImpl(path)
      val vectors = weights.toArray :+ bias
      sqlContext.createDataFrame(Seq(Tuple1(Vectors.dense(vectors)))).write.parquet(path + "/vectors")
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
      val params = vectors.select(vectors("_1")
        .as[Vector]).first().asBreeze.toDenseVector

      val weights = Vectors.fromBreeze(params(0 until params.size - 1))
      val bias = params(params.size - 1)

      val model = new LinearRegressionModel(weights, bias)
      metadata.getAndSetParams(model)
      model
    }
  }
}