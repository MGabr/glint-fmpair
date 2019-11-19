package org.apache.spark.ml.recommendation

import breeze.numerics.exp
import org.apache.spark.HashPartitioner
import ServerSideGlintFMPairModel.ServerSideGlintFMPairModelWriter
import glint.{Client, FMPairArguments}
import org.apache.spark.ml.linalg.{SparseVector, VectorUDT}
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.param.{FloatParam, IntParam, ParamMap, ParamValidators, Params}
import org.apache.spark.ml.param.shared.{HasInputCols, HasMaxIter, HasOutputCols, HasSeed, HasStepSize}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable, MLReadable, MLReader, MLWritable, MLWriter, SchemaUtils}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{FloatType, IntegerType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Dataset}

import scala.concurrent.duration._
import scala.concurrent.{Await, ExecutionContext, Future}
import scala.util.Random

private[feature] trait ServerSideGlintFMPairBase extends Params
  with HasInputCols with HasOutputCols with HasMaxIter with HasStepSize with HasSeed {

  setDefault(inputCols -> Array("userid", "userctxfeatures", "itemid", "itemfeatures"))

  protected final val useridColIdx = 0
  protected final val userctxfeaturesColIdx = 1
  protected final val itemidColIdx = 2
  protected final val itemfeaturesColIdx = 3

  protected def getUseridCol: String = $(inputCols)(useridColIdx)
  protected def getUserctxfeaturesCol: String = $(inputCols)(userctxfeaturesColIdx)
  protected def getItemidCol: String = $(inputCols)(itemidColIdx)
  protected def getItemfeaturesCol: String = $(inputCols)(itemfeaturesColIdx)

  setDefault(outputCols -> Array("itemid", "score"))

  protected final val itemidOutputColIdx = 0
  protected final val scoreOutputColIdx = 1

  protected def getItemidOutputCol: String = $(outputCols)(itemidOutputColIdx)
  protected def getScoreOutputCol: String = $(outputCols)(scoreOutputColIdx)

  final val topN = new IntParam(this, "topN", "the number of top items to recommend", ParamValidators.gt(0))
  setDefault(topN -> 500)

  // epochs = maxIter

  // batch size
  final val batchSize = new IntParam(this, "batchSize", "the mini-batch size", ParamValidators.gt(0))
  setDefault(batchSize -> 4096)

  def getBatchSize: Int = $(batchSize)

  // num_dims
  final val numDims = new IntParam(this, "numDims", "the number of dimensions (k)", ParamValidators.gt(0))

  def getNumDims: Int = $(numDims)

  // regularization rate
  final val linearReg = new FloatParam(this, "linearReg", "the regularization rate for the linear weights", ParamValidators.gtEq(0))

  def getLinearReg: Float = $(linearReg)

  final val factorsReg = new FloatParam(this, "factorsReg", "the regularization rate for the factor weights", ParamValidators.gtEq(0))

  def getFactorsReg: Float = $(factorsReg)

  // init_mean
  final val initMean = new FloatParam(this, "initMean", "the initialization mean for model weights", ParamValidators.gtEq(0))

  def getInitMean: Float = $(initMean)

  // lr = step_size
}

class ServerSideGlintFMPair(override val uid: String)
  extends Estimator[ServerSideGlintFMPairModel] with ServerSideGlintFMPairBase with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("glint-fmpair"))

  def setBatchSize(value: Int): this.type = set(batchSize, value)

  def setNumDims(value: Int): this.type = set(numDims, value)

  def setLinearReg(value: Float): this.type = set(linearReg, value)

  def setFactorsReg(value: Float): this.type = set(factorsReg, value)

  def setInitMean(value: Float): this.type = set(initMean, value)

  override def fit(dataset: Dataset[_]): ServerSideGlintFMPairModel = {

    // broadcasted mapping array for item id -> item features
    // this is assumed to fit on each worker
    val itemFeaturesMapping = dataset
      .select(getItemidCol, getItemfeaturesCol)
      .groupBy(getItemidCol)
      .agg(first(getItemfeaturesCol))
      .sort(getItemidCol)
      .collect()
      .map(_.getAs[SparseVector](itemidColIdx))
    val bcItemFeatures = dataset.sqlContext.sparkContext.broadcast(itemFeaturesMapping)

    @transient
    implicit val ec = ExecutionContext.Implicits.global

    @transient
    val client = Client()

    val args = FMPairArguments(getNumDims, getBatchSize, getStepSize.toFloat, getLinearReg, getFactorsReg)
    val numFeatures = 0  // TODO
    val avgActiveFeatures = 0 // TODO

    @transient
    val linear = client.fmpairVector(args, numFeatures, avgActiveFeatures)
    @transient
    val factors = client.fmpairMatrix(args, numFeatures, avgActiveFeatures)

    dataset.select(getUseridCol, getItemidCol, getUserctxfeaturesCol).rdd
      .map(row => (
        row.getInt(useridColIdx),
        row.getInt(itemidColIdx),
        row.getAs[SparseVector](userctxfeaturesColIdx)
      ))
      .keyBy(t => t._1)
      .partitionBy(new HashPartitioner(100))  // TODO
      .foreachPartition(iter => {

        var interactions = iter.map(_._2).toSeq
        val userItems = interactions.groupBy(_._1).mapValues(v => v.map(_._2).toSet)
        val itemFeatures = bcItemFeatures.value

        val epochs = getMaxIter
        val epochSize = interactions.length
        val batchSize = getBatchSize
        val seed = getSeed

        for (epoch <- 0 to epochs) {
          // sample interactions (users, contexts and positive items)
          val random = new Random(seed + epoch)
          interactions = random.shuffle(interactions)

          // TODO: make customizable - album, artist, exp, instance - (userid, random, itemFeatures)
          // sample negative items
          def sampleNegatives(userid: Int): Int = {
            var negative = random.nextInt() % itemFeatures.length
            while (userItems(userid).contains(negative)) {
              negative = random.nextInt() % itemFeatures.length
            }
            negative
          }
          val negatives = interactions.map(i => sampleNegatives(i._1))

          for (batch <- 0 to epochSize by batchSize) {
            // get required indices and weights of mini-batch
            val batchInteractions = interactions.slice(batch * batchSize, (batch + 1) * batchSize)
            val batchNegatives = negatives.slice(batch * batchSize, (batch + 1) * batchSize)

            val iUser = batchInteractions.map(_._3.indices).toArray
            val wUser = batchInteractions.map(_._3.values.map(_.toFloat)).toArray
            val iItem = (batchInteractions.map(i => itemFeatures(i._2).indices) ++
              batchNegatives.map(i => itemFeatures(i).indices)).toArray
            val wItem = (batchInteractions.map(i => itemFeatures(i._2).values.map(_.toFloat)) ++
              batchNegatives.map(i => itemFeatures(i).values.map(v => (-v).toFloat))).toArray

            // actual training - communicate with parameter servers
            val batchFutureLinear = linear.pullSum(iItem, wItem)
            val batchFutureFactors = factors.dotprod(iUser, wUser, iItem, wItem)
            val batchFuture = for {
              (fLinear, cacheKeysLinear) <- batchFutureLinear
              (fFactors, cacheKeysFactors) <- batchFutureFactors
            } yield {
              val g = fLinear.zip(fFactors).map{ case (fL, fF) => exp(-(fL + fF)) }.map(e => e / e + 1.0f)
              Future.sequence(Seq(linear.adjust(g, cacheKeysLinear), factors.adjust(g, cacheKeysFactors)))
            }
            Await.ready(batchFuture, 1 minute)
          }
        }
      })
  }

  override def copy(extra: ParamMap): Estimator[ServerSideGlintFMPairModel] = {

  }

  override def transformSchema(schema: StructType): StructType = {
    require(getInputCols.length == 4,
      s"inputCols have to be of length 4 but were actually of length ${getInputCols.length}")
    SchemaUtils.checkColumnType(schema, getUseridCol, IntegerType)
    SchemaUtils.checkColumnType(schema, getUserctxfeaturesCol, new VectorUDT)
    SchemaUtils.checkColumnType(schema, getItemidCol, IntegerType)
    SchemaUtils.checkColumnType(schema, getItemfeaturesCol, new VectorUDT)
    schema
  }
}

object ServerSideGlintFMPair extends DefaultParamsReadable[ServerSideGlintFMPair] {

  override def load(path: String): ServerSideGlintFMPair = super.load(path)
}

class ServerSideGlintFMPairModel private[ml](override val uid: String)
  extends Model[ServerSideGlintFMPairModel] with ServerSideGlintFMPairBase with MLWritable {

  // TODO: efficient recommendation
  // TODO: other ALS methods
  override def transform(dataset: Dataset[_]): DataFrame = {

  }

  override def transformSchema(schema: StructType): StructType = {
    require(getInputCols.length == 2,
      s"inputCols have to be of length 2 but were actually of length ${getInputCols.length}")
    SchemaUtils.checkColumnType(schema, getUseridCol, IntegerType)
    SchemaUtils.checkColumnType(schema, getUserctxfeaturesCol, new VectorUDT)

    require(getOutputCols.length == 2,
      s"outputCols have to be of length 2 but were actually of length ${getOutputCols.length}")
    SchemaUtils.checkColumnType(schema, getItemidOutputCol, IntegerType)
    SchemaUtils.checkColumnType(schema, getScoreOutputCol, FloatType)

    val fields = schema.fields.filter(f => !f.name.equals(getUserctxfeaturesCol))
    StructType(fields :+ StructField(getItemidOutputCol, IntegerType) :+ StructField(getScoreOutputCol, FloatType))
  }

  override def copy(extra: ParamMap): ServerSideGlintFMPairModel = {

  }

  override def write: MLWriter = new ServerSideGlintFMPairModelWriter(this)


}

object ServerSideGlintFMPairModel extends MLReadable[ServerSideGlintFMPairModel] {

  private[GlintFMModel] class ServerSideGlintFMPairModelWriter(instance: ServerSideGlintFMPairModel) extends MLWriter {

    override protected def saveImpl(path: String): Unit = {

    }
  }

  private class ServerSideGlintFMPairModelReader extends MLReader[ServerSideGlintFMPairModel] {

    override def load(path: String): ServerSideGlintFMPairModel = {

    }
  }

  override def read: MLReader[ServerSideGlintFMPairModel] = new ServerSideGlintFMPairModelReader

  override def load(path: String): ServerSideGlintFMPairModel = super.load(path)
}
