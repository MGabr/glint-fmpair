package org.apache.spark.ml.recommendation

import java.io.{ObjectInputStream, ObjectOutputStream}

import breeze.linalg.{rank => breezeRank, sum => breezeSum, _}
import breeze.numerics.sigmoid
import com.typesafe.config._
import glint.models.client.{BigFMPairMatrix, BigFMPairVector}
import glint.{Client, FMPairArguments}
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.{SparkContext, TaskContext}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.linalg.{SparseVector, VectorUDT}
import org.apache.spark.ml.param.shared.{HasMaxIter, HasPredictionCol, HasSeed, HasStepSize}
import org.apache.spark.ml.param._
import org.apache.spark.ml.recommendation.GlintFMPair.{Interaction, MetaData, SampledFeatures, SampledInteraction}
import org.apache.spark.ml.util._
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.sql.api.java.UDF2
import org.apache.spark.sql.catalyst.encoders.RowEncoder
import org.apache.spark.sql.expressions.{Aggregator, Window}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{ArrayType, FloatType, IntegerType, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, Encoder, Encoders, Row, SparkSession}
import org.apache.spark.util.collection.PrimitiveKeyOpenHashMap
import org.eclipse.collections.impl.map.mutable.primitive.IntObjectHashMap
import spire.implicits.cforRange

import scala.collection.BitSet
import scala.concurrent.duration._
import scala.concurrent.{Await, ExecutionContext, Future}
import scala.util.Random

private[recommendation] trait GlintFMPairParams extends Params with HasMaxIter with HasStepSize with HasSeed
  with HasPredictionCol {

  /**
   * The name of the user id column of integers
   * Default: "userid"
   *
   * @group param
   */
  final val userCol = new Param[String](this, "userCol", "the name of the user id column")
  setDefault(userCol -> "userid")

  /** @group getParam */
  def getUserCol: String = $(userCol)

  /**
   * The name of the item id column of integers from 0 to number of items in training dataset
   * Default: "itemid"
   *
   * @group param
   */
  final val itemCol = new Param[String](this, "itemCol", "the name of the item id column")
  setDefault(itemCol -> "itemid")

  /** @group getParam */
  def getItemCol: String = $(itemCol)

  /**
   * The name of the user and context feature column of sparse vectors
   * Default: "userctxfeatures"
   *
   * @group param
   */
  final val userctxfeaturesCol = new Param[String](this, "userctxfeaturesCol",
    "the name of the user and context features column")
  setDefault(userctxfeaturesCol -> "userctxfeatures")

  /** @group getParam */
  def getUserctxfeaturesCol: String = $(userctxfeaturesCol)

  /**
   * The name of the item feature column of sparse vectors
   * Default: "itemfeatures"
   *
   * @group param
   */
  final val itemfeaturesCol = new Param[String](this, "itemfeaturesCol",
    "the name of the item features column")
  setDefault(itemfeaturesCol -> "itemfeatures")

  /** @group getParam */
  def getItemfeaturesCol: String = $(itemfeaturesCol)

  /**
   * The name of the column of integers to use for sampling. If empty all items are accepted as negative items
   * otherwise only items where there does not exist an interaction between the user and the sampling column value
   * of the item. Usually the sampling column is the same as [[itemCol]] but it may also be another column with an
   * n-to-1 relation from item column value to sampling column value.
   *
   * Consider the example of playlists with "pid" as user column amd tracks with "traid" as item column.
   * Another column "artid" holds the artist of the track.
   * With "traid" as sampling column, only tracks which are not in the playlist are accepted as negative items.
   * With "artid" as sampling column, only tracks whose artists are not in the playlist are accepted as negative item.
   *
   * Default: ""
   *
   * @group param
   */
  final val samplingCol = new Param[String](this, "samplingCol",
    "the name of the column to use for acceptance sampling , usually same as itemCol")
  setDefault(samplingCol -> "")

  /** @group getParam */
  def getSamplingCol: String = $(samplingCol)

  /**
   * The name of the integer arrays column containing the [[itemCol]] ids of the items to filter from recommendations.
   * If empty, recommendations are not filtered. Usually the arrays will contain the ids of the items of the user
   *
   * Default: ""
   *
   * @group param
   */
  final val filterItemsCol = new Param[String](this, "filterItemsCol",
    "the name of the column to use for recommendation filtering")
  setDefault(filterItemsCol -> "")

  /** @group getParam */
  def getFilterItemsCol: String = $(filterItemsCol)


  /**
   * The sampler to use.
   *
   * "uniform" means sampling negative items uniformly, as originally proposed for BPR.
   *
   * "exp" means sampling negative items with probability proportional to their exponential popularity distribution,
   * as proposed in LambdaFM.
   *
   * "crossbatch" means sampling negative items uniformly, but using crossbatch-BPR loss,
   * as proposed in my masters thesis
   *
   * Default: "uniform"
   *
   * @group param
   */
  final val sampler = new Param[String](this, "sampler",
    "the sampler to use, one of uniform, exp and crossbatch")
  setDefault(sampler -> "uniform")

  /** @group getParam */
  def getSampler: String = $(sampler)

  /**
   * The rho value to use for the "exp" sampler. Has to be between 0.0 and 1.0
   * Default: 1.0
   *
   * @group param
   */
  final val rho = new DoubleParam(this, "rho", "the rho value to use for the exp sampler")
  setDefault(rho -> 1.0)

  /** @group getParam */
  def getRho: Double = $(rho)

  setDefault(maxIter -> 1000)
  setDefault(stepSize -> 0.1)
  setDefault(seed -> 1)

  /**
   * The per-worker mini-batch size
   * Default: 256
   *
   * @group param
   */
  final val batchSize = new IntParam(this, "batchSize",
    "the worker mini-batch size", ParamValidators.gt(0))
  setDefault(batchSize -> 256)

  /** @group getParam */
  def getBatchSize: Int = $(batchSize)

  /**
   * The number of latent factor dimensions (k)
   * Default: 150
   *
   * @group param
   */
  final val numDims = new IntParam(this, "numDims",
    "the number of dimensions (k)", ParamValidators.gt(0))
  setDefault(numDims -> 150)

  /** @group getParam */
  def getNumDims: Int = $(numDims)

  /**
   * The regularization rate for the linear weights
   * Default: 0.01f
   *
   * @group param
   */
  final val linearReg = new FloatParam(this, "linearReg",
    "the regularization rate for the linear weights", ParamValidators.gtEq(0))
  setDefault(linearReg -> 0.01f)

  /** @group getParam */
  def getLinearReg: Float = $(linearReg)

  /**
   * The regularization rate for the latent factor weights
   * Default: 0.001f
   *
   * @group param
   */
  final val factorsReg = new FloatParam(this, "factorsReg",
    "the regularization rate for the factor weights", ParamValidators.gtEq(0))
  setDefault(factorsReg -> 0.001f)

  /** @group getParam */
  def getFactorsReg: Float = $(factorsReg)


  /**
   * The number of parameter servers
   * Default: 3
   *
   * @group param
   */
  final val numParameterServers = new IntParam(this, "numParameterServers",
    "the number of parameter servers")
  setDefault(numParameterServers -> 3)

  /** @group getParam */
  def getNumParameterServers: Int = $(numParameterServers)

  /**
   * The master host of the running parameter servers.
   * If this is not set a standalone parameter server cluster is started in this Spark application.
   * Default: ""
   *
   * @group param
   */
  final val parameterServerHost = new Param[String](this, "parameterServerHost",
    "the master host of the running parameter servers. " +
      "If this is not set a standalone parameter server cluster is started in this Spark application.")
  setDefault(parameterServerHost -> "")

  /** @group getParam */
  def getParameterServerHost: String = $(parameterServerHost)

  /**
   * The parameter server configuration.
   * Allows for detailed configuration of the parameter servers with the default configuration as fallback.
   * Default: ConfigFactory.empty()
   *
   * @group param
   */
  final val parameterServerConfig = new Param[Config](this, "parameterServerConfig",
    "The parameter server configuration. Allows for detailed configuration of the parameter servers with the " +
      "default configuration as fallback.") {

    override def jsonEncode(value: Config): String = {
      value.root().render(ConfigRenderOptions.concise().setJson(true))
    }

    override def jsonDecode(json: String): Config = {
      ConfigFactory.parseString(json, ConfigParseOptions.defaults().setSyntax(ConfigSyntax.JSON))
    }
  }
  setDefault(parameterServerConfig -> ConfigFactory.empty())

  /** @group getParam */
  def getParameterServerConfig: Config = $(parameterServerConfig)

  /**
   * Whether the meta data of the data frame to fit should be loaded from HDFS.
   * This allows skipping the meta data computation stages when fitting on the same data frame
   * with different parameters. Meta data for "cross-batch" and "uniform" sampling is intercompatible
   * but "exp" requires its own meta data
   *
   * Default: false
   */
  final val loadMetadata = new BooleanParam(this, "loadMetadata",
    "Whether the meta data of the data frame to fit should be loaded from HDFS. " +
      "This allows skipping the meta data computation stages when fitting on the same data frame " +
      "with different parameters. Meta data for \"cross-batch\" and \"uniform\" sampling is intercompatible " +
      "but \"exp\" requires its own meta data")
  setDefault(loadMetadata -> false)

  /** @group getParam */
  def getLoadMetadata: Boolean = $(loadMetadata)

  /**
   * Whether the meta data of the fitted data frame should be saved to HDFS.
   * Default: false
   */
  final val saveMetadata = new BooleanParam(this, "saveMetadata",
    "Whether the meta data of the fitted data frame should be saved to HDFS")
  setDefault(saveMetadata -> false)

  /** @group getParam */
  def getSaveMetadata: Boolean = $(saveMetadata)

  /**
   * The HDFS path to load meta data for the fit data frame from or to save the fitted meta data to.
   * Default: ""
   */
  final val metadataPath = new Param[String](this, "metadataPath",
    "The HDFS path to load meta data for the fit data frame from or to save the fitted meta data to")
  setDefault(metadataPath -> "")

  /** @group getParam */
  def getMetadataPath: String = $(metadataPath)

  /**
   * The depth to use for tree reduce when computing the meta data.
   * To avoid OOM errors, this has to be set sufficiently large but lower depths might lead to faster runtimes
   */
  final val treeDepth = new IntParam(this, "treeDepth",
    "The depth to use for tree reduce when computing the meta data. " +
      "To avoid OOM errors, this has to be set sufficiently large but lower depths might lead to faster runtimes")
  setDefault(treeDepth -> 3)

  /** @group getParam */
  def getTreeDepth: Int = $(treeDepth)
}


class GlintFMPair(override val uid: String)
  extends Estimator[GlintFMPairModel] with GlintFMPairParams with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("glint-fmpair"))


  /** @group setParam */
  def setUserCol(value: String): this.type = set(userCol, value)

  /** @group setParam */
  def setItemCol(value: String): this.type = set(itemCol, value)

  /** @group setParam */
  def setUserctxFeaturesCol(value: String): this.type = set(userctxfeaturesCol, value)

  /** @group setParam */
  def setItemFeaturesCol(value: String): this.type = set(itemfeaturesCol, value)

  /** @group setParam */
  def setPredictionCol(value: String): this.type = set(predictionCol, value)

  /** @group setParam */
  def setSamplingCol(value: String): this.type = set(samplingCol, value)

  /** @group setParam */
  def setFilterItemsCol(value: String): this.type = set(filterItemsCol, value)


  /** @group setParam */
  def setSampler(value: String): this.type = set(sampler, value)

  /** @group setParam */
  def setMaxIter(value: Int): this.type = set(maxIter, value)

  /** @group setParam */
  def setStepSize(value: Double): this.type = set(stepSize, value)

  /** @group setParam */
  def setSeed(value: Long): this.type = set(seed, value)

  /** @group setParam */
  def setBatchSize(value: Int): this.type = set(batchSize, value)

  /** @group setParam */
  def setNumDims(value: Int): this.type = set(numDims, value)

  /** @group setParam */
  def setLinearReg(value: Float): this.type = set(linearReg, value)

  /** @group setParam */
  def setFactorsReg(value: Float): this.type = set(factorsReg, value)


  /** @group setParam */
  def setNumParameterServers(value: Int): this.type = set(numParameterServers, value)

  /** @group setParam */
  def setParameterServerHost(value: String): this.type = set(parameterServerHost, value)

  /** @group setParam */
  def setParameterServerConfig(value: Config): this.type = set(parameterServerConfig, value.resolve())


  /** @group setParam */
  def setLoadMetadata(value: Boolean): this.type = set(loadMetadata, value)

  /** @group setParam */
  def setSaveMetadata(value: Boolean): this.type = set(saveMetadata, value)

  /** @group setParam */
  def setMetadataPath(value: String): this.type = set(metadataPath, value)


  /**
   * Fits a [[org.apache.spark.ml.recommendation.GlintFMPairModel GlintFMPairModel]] on the data set
   *
   * @param dataset The data set containing columns (userCol: Int, itemCol: Int, itemFeaturesCol: SparseVector,
   *                userctxFeaturesCol: SparseVector) and if acceptance sampling should be used also samplingCol.
   */
  override def fit(dataset: Dataset[_]): GlintFMPairModel = {

    val spark = dataset.sparkSession
    import spark.implicits._
    val sc = spark.sparkContext

    val (df, numItemFeatures, numFeatures) = toDF(dataset)

    val meta = fitMeta(df, numItemFeatures, numFeatures)

    val bcItem2features = sc.broadcast(meta.item2features)
    val bcItemsByCount = sc.broadcast(meta.itemsByCount)
    val bcItem2sampling = sc.broadcast(meta.item2sampling)

    val numWorkers = Client.getNumExecutors(sc)
    val numWorkerCores = Client.getExecutorCores(sc)

    @transient
    implicit val ec = ExecutionContext.Implicits.global
    @transient
    val (client, linear, factors) = initParameterServers(sc, meta.itemFeatureProbs, meta.featureProbs,
      meta.avgActiveFeatures, numWorkers, numWorkerCores)

    try {
      df.select(getUserCol, getItemCol, getUserctxfeaturesCol)
        .repartition(numWorkers * numWorkerCores, col(getUserCol)) // partition data frame by users
        .map(row => Interaction(row.getInt(0), row.getInt(1), row.getAs[SparseVector](2)))
        .foreachPartition((iter: Iterator[Interaction]) => {

          val interactions = iter.toArray

          val epochs = getMaxIter
          val batchSize = getBatchSize
          val seed = getSeed
          val sampler = getSampler

          val item2features = bcItem2features.value
          val item2sampling = bcItem2sampling.value
          val itemsByCount = bcItemsByCount.value

          // create mapping of user ids to their set of sampling ids
          val userSamplings = item2sampling.map(i2s => {
            val us = new IntObjectHashMap[BitSet]() // instead of scala map, essential for performance
            interactions.groupBy(_.userId).foreach { case (userId, userInteractions) =>
              us.put(userId, BitSet(userInteractions.map(i => i2s(i.itemId)): _*))
            }
            us
          })

          // create sampling function to use
          val sample = if (item2sampling.isDefined) {
            sampler match {
              case "uniform" => GlintFMPair.uniformSampler(userSamplings.get, item2sampling.get)
              case "exp" => GlintFMPair.expSampler(userSamplings.get, item2sampling.get, itemsByCount.get, getRho)
              case "crossbatch" => GlintFMPair.crossbatchSampler(userSamplings.get, item2sampling.get)
            }
          } else {
            sampler match {
              case "uniform" | "crossbatch" => GlintFMPair.uniformAllSampler(item2features.length)
              case "exp" => GlintFMPair.expAllSampler(itemsByCount.get, getRho)
            }
          }

          @transient
          implicit val ec = ExecutionContext.Implicits.global

          // now the actual fit pipeline
          val fitFinishedFuture = (0 until epochs).iterator
            .flatMap { epoch =>

              if (epoch % 10 == 0) {
                logInfo(s"Epoch $epoch of $epochs")
              }

              val random = new Random(seed + TaskContext.getPartitionId() + epoch)
              // sample positive users, contexts and items
              GlintFMPair.shuffle(random, interactions)
              // group into batches and sample negative items
              interactions.grouped(batchSize).map(i => sample(random, i))

            }.map {
            // add dummy non-acceptance matrix if necessary
            case (batch: Array[SampledInteraction], na: DenseMatrix[Float]) => (batch, na)
            case batch: Array[SampledInteraction] => (batch, DenseMatrix.zeros[Float](1, 1))

          }.map { case (batch, na) =>
            // lookup features of sampled items
            val fBatch = batch.map { case SampledInteraction(userId, positemId, negitemId, userctxFeatures) =>
              SampledFeatures(userctxFeatures, item2features(positemId), item2features(negitemId))
            }
            (fBatch, na)

          }.map { case (batch, na) =>
            // convert features to the arrays required by the parameter servers
            val iUser = batch.map(_.userctxFeatures.indices)
            val wUser = batch.map(_.userctxFeatures.values.map(_.toFloat))
            val (iItem, wItem) = sampler match {
              case "uniform" | "exp" =>
                val i = batch.map(i => i.positemFeatures.indices ++ i.negitemFeatures.indices)
                val w = batch.map(i => i.positemFeatures.values.map(_.toFloat) ++
                  i.negitemFeatures.values.map(v => (-v).toFloat))
                (i, w)
              case "crossbatch" =>
                val i = batch.map(_.positemFeatures.indices) ++ batch.map(_.negitemFeatures.indices)
                val w = batch.map(_.positemFeatures.values.map(_.toFloat)) ++
                  batch.map(_.negitemFeatures.values.map(_.toFloat))
                (i, w)
            }
            (iUser, wUser, iItem, wItem, na)

          }.foldLeft(Future.successful(Seq(true))) { case (prevBatchFuture, (iUser, wUser, iItem, wItem, na)) =>
            // wait until communication with parameter servers for previous batches is finished
            // this allows already pre-processing the next batch while waiting for the parameter server responses
            Await.result(prevBatchFuture, 5 minute)

            // communicate with the parameter servers for SGD step
            sampler match {
              case "uniform" | "exp" => // "normal" BPR loss
                val batchFutureLinear = linear.pullSum(iItem, wItem)
                val batchFutureFactors = factors.dotprod(iUser, wUser, iItem, wItem)
                val batchFuture = for {
                  (sLinear, cacheKeysLinear) <- batchFutureLinear
                  (fFactors, cacheKeysFactors) <- batchFutureFactors
                } yield {
                  val g = GlintFMPair.computeBPRGradients(sLinear, fFactors)
                  Future.sequence(Seq(linear.pushSum(g, cacheKeysLinear), factors.adjust(g, cacheKeysFactors)))
                }
                batchFuture.flatMap(identity)

              case "crossbatch" => // crossbatch-BPR loss
                val batchFutureLinear = linear.pullSum(iItem, wItem)
                val batchFutureFactors = factors.pullSum(iUser ++ iItem, wUser ++ wItem)
                val batchFuture = for {
                  (sLinear, cacheKeysLinear) <- batchFutureLinear
                  (sFactors, cacheKeysFactors) <- batchFutureFactors
                } yield {
                  val (gLinear, gFactors) = GlintFMPair.computeCrossbatchBPRGradients(sLinear, sFactors, na)
                  Future.sequence(Seq(
                    linear.pushSum(gLinear, cacheKeysLinear),
                    factors.pushSum(gFactors, cacheKeysFactors)))
                }
                batchFuture.flatMap(identity)
            }
          }

          // wait until communication with parameter servers for last batch is finished
          Await.result(fitFinishedFuture, 5 minute)
          ()
        })

      copyValues(new GlintFMPairModel(this.uid, bcItem2features, linear, factors, client).setParent(this))

    } catch { case e: Exception =>
      factors.destroy()
      linear.destroy()
      client.terminateOnSpark(sc)
      bcItem2features.destroy()
      throw e
    } finally {
      bcItem2sampling.destroy()
      bcItemsByCount.destroy()
    }
  }

  /**
   * Converts the data set to a data frame required for this model. If the features have separate index ranges,
   * the user / context feature indices are shifted so that they start after the item feature indices
   */
  private def toDF(dataset: Dataset[_]): (DataFrame, Int, Int) = {
    val sampleRow = dataset.select(getItemfeaturesCol, getUserctxfeaturesCol).first()
    val numItemFeatures = sampleRow.getAs[SparseVector](0).size
    val numUserctxFeatures = sampleRow.getAs[SparseVector](1).size
    val numFeatures = if (numItemFeatures == numUserctxFeatures) {
      numItemFeatures
    } else {
      numItemFeatures + numUserctxFeatures
    }

    var cols = Array(col(getUserCol).cast(IntegerType), col(getItemCol).cast(IntegerType))
    if (numItemFeatures == numFeatures) {
      cols = cols :+ col(getItemfeaturesCol)
      cols = cols :+ col(getUserctxfeaturesCol)
    } else {
      val resize = udf((itemFeatures: SparseVector) =>
        new SparseVector(numFeatures, itemFeatures.indices, itemFeatures.values))
      cols = cols :+ resize(col(getItemfeaturesCol)).as(getItemfeaturesCol)
      val shift = udf((userctxFeatures: SparseVector) =>
        new SparseVector(numFeatures, userctxFeatures.indices.map(i => numItemFeatures + i), userctxFeatures.values))
      cols = cols :+ shift(col(getUserctxfeaturesCol)).as(getUserctxfeaturesCol)
    }
    if (getSamplingCol.nonEmpty && getSamplingCol != getItemCol) {
      cols = cols :+ col(getSamplingCol).cast(IntegerType)
    }
    val df = dataset.select(cols :_*)

    (df, numItemFeatures, numFeatures)
  }

  /**
   * Computes the meta data of a data frame.
   * Also supports saving the meta data to HDFS and loading the meta data from HDFS to skip the computations.
   */
  private def fitMeta(df: DataFrame, numItemFeatures: Int, numFeatures: Int): MetaData = {

    if ((getLoadMetadata || getSaveMetadata) && getMetadataPath.isEmpty) {
      throw new IllegalArgumentException("No meta data path set")
    }

    if (getLoadMetadata) {
      val fs = FileSystem.get(df.sparkSession.sparkContext.hadoopConfiguration)
      if (fs.exists(new Path(getMetadataPath))) {
        val dataStream = new ObjectInputStream(fs.open(new Path(getMetadataPath)))
        val meta = dataStream.readObject().asInstanceOf[MetaData]
        dataStream.close()

        if (numItemFeatures != meta.itemFeatureProbs.length) {
          throw new IllegalArgumentException("Number of item features does not match with loaded meta data: " +
            "%s and %s".format(numItemFeatures, meta.itemFeatureProbs.length))
        }
        if (numFeatures != meta.featureProbs.length) {
          throw new IllegalArgumentException("Number of features does not match with loaded meta data: " +
            "%s and %s".format(numFeatures, meta.featureProbs.length))
        }
        if (meta.itemsByCount.isEmpty && getSampler.equals("exp")) {
          throw new IllegalArgumentException("\"exp\" sampler does not work with loaded meta data")
        }
        if (meta.itemsByCount.isDefined && !getSampler.equals("exp")) {
          throw new IllegalArgumentException(
            "Sampler \"%s\" does not work with meta data fitted with \"exp\" sampler".format(getSampler))
        }

        return meta
      }
    }

    val dfItem2features = df
      .select(getItemCol, getItemfeaturesCol)
      .groupBy(getItemCol)
      .agg(first(getItemfeaturesCol).as(getItemfeaturesCol))

    val featureProbs = computeFeatureProbs(df, dfItem2features)
    val itemFeatureProbs = featureProbs.slice(0, numItemFeatures)
    val avgActiveFeatures = featureProbs.sum.toInt

    val item2features = computeItem2features(dfItem2features)
    val itemsByCount = computeItemsByCount(df)
    val item2sampling = computeItem2sampling(df)
    val meta = MetaData(itemFeatureProbs, featureProbs, avgActiveFeatures, item2features, itemsByCount, item2sampling)

    if (getSaveMetadata) {
      val fs = FileSystem.get(df.sparkSession.sparkContext.hadoopConfiguration)
      val dataStream = new ObjectOutputStream(fs.create(new Path(getMetadataPath)))
      dataStream.writeObject(meta)
      dataStream.close()
    }

    meta
  }

  /**
   * Computes the feature probabilities of a data frame
   */
  private def computeFeatureProbs(df: DataFrame, dfItem2features: DataFrame): Array[Float] = {
    val featureProbs = aggFeatureProbabilities(df.select(getItemfeaturesCol, getUserctxfeaturesCol), 2)
    val negFeatureProbs = computeNegFeatureProbs(df, dfItem2features)
    for (i <- negFeatureProbs.indices) {
      featureProbs(i) += negFeatureProbs(i)
    }
    featureProbs
  }

  /**
   * Computes the negative item feature probabilities for the used sampling method
   */
  private def computeNegFeatureProbs(df: DataFrame, dfItem2features: DataFrame): Array[Float] = {

    if (getSampler.equals("exp")) {
      val itemCount = dfItem2features.count()
      val rho = getRho
      val exp = udf((rank: Int) => -(rank.toDouble + 1) / (itemCount.toDouble * rho.toDouble))

      val dfItems2Exp = df
        .select(getItemCol, getItemfeaturesCol)
        .groupBy(getItemCol)
        .agg(first(getItemfeaturesCol).as(getItemfeaturesCol), count(getItemfeaturesCol).as("count"))
        .select(col(getItemfeaturesCol), exp(rank().over(Window.orderBy(desc("count")))).as("exp"))

      val expSum = dfItems2Exp.select("exp").groupBy().sum().first.get(0)

      aggWeightedFeatureProbabilities(dfItems2Exp.select(col("exp").divide(expSum), col(getItemfeaturesCol)))
    } else {
      aggFeatureProbabilities(dfItem2features.select(getItemfeaturesCol), 1)
    }
  }

  /**
   * Computes the feature probabilities of sparse feature vector columns
   * through mapPartitions for efficiency and through treeReduce to avoid OOM on the driver
   *
   * @param df The dataframe to aggregate, must have sparse vector columns starting at column 0
   * @param numCols The number of sparse vector columns
   */
  def aggFeatureProbabilities(df: DataFrame, numCols: Int): Array[Float] = {
    val (count, featureCounts) = df.rdd.mapPartitions { iter =>
      var count = 0L
      var featureCounts = new Array[Long](0)
      for (row <- iter) {
        if (count == 0L) {
          featureCounts = new Array[Long](row.getAs[SparseVector](0).size)
        }
        count += 1L
        for (iCol <- 0 until numCols) {
          for (i <- row.getAs[SparseVector](iCol).indices) {
            featureCounts(i) += 1L
          }
        }
      }
      Iterator((count, featureCounts))
    }.treeReduce( (p1, p2) => {
      val (count1, featureCounts1) = p1
      val (count2, featureCounts2) = p2
      if (count1 == 0L) {
        (count2, featureCounts2)
      } else {
        cforRange(0 until featureCounts2.length) { i =>
          featureCounts1(i) += featureCounts2(i)
        }
        (count1 + count2, featureCounts1)
      }
    }, getTreeDepth)
    featureCounts.map(c => (c.toDouble / count.toDouble).toFloat)
  }

  /**
   * Computes the weighted feature probabilities of sparse feature vector columns
   * through mapPartitions for efficiency and through treeReduce to avoid OOM on the driver
   *
   * @param df The dataframe to aggregate, must have a double weight column at column 0 and
   *           sparse vector column at column 1
   */
  def aggWeightedFeatureProbabilities(df: DataFrame): Array[Float] = {
    df.rdd.mapPartitions { iter =>
      var featureProbs = new Array[Double](0)
      for (row <- iter) {
        if (featureProbs.length == 0) {
          featureProbs = new Array[Double](row.getAs[SparseVector](1).size)
        }
        val weight = row.getDouble(0)
        for (i <- row.getAs[SparseVector](1).indices) {
          featureProbs(i) += weight
        }
      }
      Iterator(featureProbs)
    }.treeReduce( (featureProbs1, featureProbs2) => {
      if (featureProbs1.length == 0) {
        featureProbs2
      } else {
        cforRange(0 until featureProbs2.length) { i =>
          featureProbs1(i) += featureProbs2(i)
        }
        featureProbs1
      }
    }, getTreeDepth).map(_.toFloat)
  }

  /**
   * Initializes parameter server client, vector and matrices
   */
  private def initParameterServers(sc: SparkContext,
                                   itemFeatureProbs: Array[Float],
                                   featureProbs: Array[Float],
                                   avgActiveFeatures: Int,
                                   numWorkers: Int,
                                   numWorkerCores: Int)
                                  (implicit ec: ExecutionContext): (Client, BigFMPairVector, BigFMPairMatrix) = {

    val client = if (getParameterServerHost.isEmpty) {
      Client.runOnSpark(sc, getParameterServerConfig, getNumParameterServers, numWorkers)
    } else {
      Client(Client.getHostConfig(getParameterServerHost).withFallback(getParameterServerConfig))
    }

    try {
      val args = FMPairArguments(getNumDims, getBatchSize, getStepSize.toFloat, getLinearReg, getFactorsReg)

      val linear = client.fmpairVector(args, itemFeatureProbs, sc.hadoopConfiguration, numWorkers * numWorkerCores,
        avgActiveFeatures)
      val factors = client.fmpairMatrix(args, featureProbs, sc.hadoopConfiguration, numWorkers * numWorkerCores,
        avgActiveFeatures, getNumParameterServers)

      (client, linear, factors)
    } catch { case e: Exception =>
      client.terminateOnSpark(sc)
      throw e
    }
  }

  /**
   * Creates a mapping array of item ids to item features
   */
  private def computeItem2features(dfItem2features: DataFrame): Array[SparseVector] = {
    dfItem2features
      .sort(getItemCol)
      .collect()
      .map(_.getAs[SparseVector](getItemfeaturesCol))
  }

  /**
   * Creates an array of item ids sorted by their counts ... if exp sampling is used
   */
  private def computeItemsByCount(df: DataFrame): Option[Array[Int]] = {
    if (getSampler.equals("exp")) {
      Some(df
        .select(getItemCol)
        .groupBy(getItemCol)
        .count()
        .sort(desc("count"))
        .select(getItemCol)
        .rdd
        .map(row => row.getInt(0))
        .collect())
    } else {
      None
    }
  }

  /**
   * Creates a mapping array of item ids to sampling ids ... if accepted sampling / sampling column is used
   */
  private def computeItem2sampling(df: DataFrame): Option[Array[Int]] = {
    if (getSamplingCol.nonEmpty) {
      Some(df
        .select(getItemCol, getSamplingCol)
        .groupBy(getItemCol)
        .agg(first(getSamplingCol))
        .sort(getItemCol)
        .collect()
        .map(_.getInt(1)))
    } else {
      None
    }
  }

  override def copy(extra: ParamMap): Estimator[GlintFMPairModel] = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, getUserCol, IntegerType)
    SchemaUtils.checkColumnType(schema, getUserctxfeaturesCol, new VectorUDT)
    SchemaUtils.checkColumnType(schema, getItemCol, IntegerType)
    SchemaUtils.checkColumnType(schema, getItemfeaturesCol, new VectorUDT)
    schema
  }
}

/**
 * UDF to resize a sparse vector to the same size as another sparse vector
 */
class ResizeVector extends UDF2[SparseVector, SparseVector, SparseVector] {
  override def call(vector1: SparseVector, vector2: SparseVector): SparseVector = {
    new SparseVector(vector2.size, vector1.indices, vector1.values)
  }
}


object GlintFMPair extends DefaultParamsReadable[GlintFMPair] {

  /** The required meta data about the training instances */
  private case class MetaData(itemFeatureProbs: Array[Float],
                              featureProbs: Array[Float],
                              avgActiveFeatures: Int,
                              item2features: Array[SparseVector],
                              itemsByCount: Option[Array[Int]],
                              item2sampling: Option[Array[Int]])

  /** An interaction from the training instances */
  private case class Interaction(userId: Int, itemId: Int, userctxFeatures: SparseVector)

  /** An interaction from the training instances, combined with a sampled negative interaction */
  private case class SampledInteraction(userId: Int, positemId: Int, negitemId: Int, userctxFeatures: SparseVector)

  /** The features of a positive interaction from the training instances and a sampled negative interaction */
  private case class SampledFeatures(userctxFeatures: SparseVector,
                                     positemFeatures: SparseVector,
                                     negitemFeatures: SparseVector)

  /**
   * Samples positive items / interactions by shuffling the interactions in-place.
   * Uses Fisher-Yates shuffle algorithm
   *
   * @param random The random number generator to use
   * @param interactions The interactions to shuffle
   */
  private def shuffle(random: Random, interactions: Array[Interaction]): Unit = {
    cforRange(interactions.length - 1 until 0 by -1)(i => {
      val j = random.nextInt(i + 1)
      val tmp = interactions(j)
      interactions(j) = interactions(i)
      interactions(i) = tmp
    })
  }

  /**
   * Samples negative items uniformly, accepting all items as negative items
   *
   * @param numItems The number of user ids
   * @return A sampler function creating an array of sampled interactions
   */
  private def uniformAllSampler(numItems: Int): (Random, Array[Interaction]) => Array[SampledInteraction] = {

    def uniformSample(random: Random, interactions: Array[Interaction]): Array[SampledInteraction] = {
      interactions.map { case Interaction(userId, itemId, userctxFeatures) =>
        SampledInteraction(userId, itemId, random.nextInt(numItems), userctxFeatures)
      }
    }
    uniformSample
  }

  /**
   * Samples negative items with probabilities proportional to the exponential popularity distribution from LambdaFM,
   * accepting all items as negative items. Uses inverse transformation sampling from truncated exponential distribution
   *
   * @param itemsByCount The item indices ordered by descending popularity / occurrence count
   * @param rho The rho parameter for the exponential distribution
   * @return A sampler function creating an array of sampled interactions
   */
  private def expAllSampler(itemsByCount: Array[Int], rho: Double):
  (Random, Array[Interaction]) => Array[SampledInteraction] = {

    val numItems = itemsByCount.length
    val truncationCDF = 1.0 - math.exp(-1.0 / rho)

    def expSample(random: Random, interactions: Array[Interaction]): Array[SampledInteraction] = {
      interactions.map { case Interaction(userId, itemId, userctxFeatures) =>
        val negitemId = itemsByCount((-numItems * rho * math.log(1 - random.nextDouble() * truncationCDF)).toInt)
        SampledInteraction(userId, itemId, negitemId, userctxFeatures)
      }
    }
    expSample
  }

  /**
   * Samples negative items uniformly
   *
   * @param userSamplings The mapping of user ids to their set of sampling ids
   * @param item2sampling The mapping of item ids to their sampling id
   * @return A sampler function creating an array of sampled interactions
   */
  private def uniformSampler(userSamplings: IntObjectHashMap[BitSet], item2sampling: Array[Int]):
  (Random, Array[Interaction]) => Array[SampledInteraction] = {

    val numItems = item2sampling.length

    def uniformSample(random: Random, interactions: Array[Interaction]): Array[SampledInteraction] = {
      interactions.map { case Interaction(userId, itemId, userctxFeatures) =>
        var negitemId = random.nextInt(numItems)
        while (userSamplings.get(userId).contains(item2sampling(negitemId))) {
          negitemId = random.nextInt(numItems)
        }
        SampledInteraction(userId, itemId, negitemId, userctxFeatures)
      }
    }
    uniformSample
  }

  /**
   * Samples negative items with probabilities proportional to the exponential popularity distribution from LambdaFM.
   * Uses inverse transformation sampling from truncated exponential distribution
   *
   * @param userSamplings The mapping of user ids to their set of sampling ids
   * @param item2sampling The mapping of item ids to their sampling id
   * @param itemsByCount The item indices ordered by descending popularity / occurrence count
   * @param rho The rho parameter for the exponential distribution
   * @return A sampler function creating an array of sampled interactions
   */
  private def expSampler(userSamplings: IntObjectHashMap[BitSet],
                         item2sampling: Array[Int],
                         itemsByCount: Array[Int],
                         rho: Double): (Random, Array[Interaction]) => Array[SampledInteraction] = {

    val numItems = item2sampling.length
    val truncationCDF = 1.0 - math.exp(-1.0 / rho)

    def expSample(random: Random, interactions: Array[Interaction]): Array[SampledInteraction] = {
      interactions.map { case Interaction(userId, itemId, userctxFeatures) =>
        var negitemId = itemsByCount((-numItems * rho * math.log(1 - random.nextDouble() * truncationCDF)).toInt)
        while (userSamplings.get(userId).contains(item2sampling(negitemId))) {
          negitemId = itemsByCount((-numItems * rho * math.log(1 - random.nextDouble() * truncationCDF)).toInt)
        }
        SampledInteraction(userId, itemId, negitemId, userctxFeatures)
      }
    }
    expSample
  }

  /**
   * Samples negative items uniformly and return matrix of non-accepted negative items across batch
   *
   * @param userSamplings The mapping of user ids to their set of sampling ids
   * @param item2sampling The mapping of item ids to their sampling id
   * @return A sampler function creating an array of sampled interactions and a non-acceptance matrix
   */
  private def crossbatchSampler(userSamplings: IntObjectHashMap[BitSet], item2sampling: Array[Int]):
  (Random, Array[Interaction]) => (Array[SampledInteraction], DenseMatrix[Float]) = {

    val numItems = item2sampling.length

    def crossbatchSample(random: Random,
                         interactions: Array[Interaction]): (Array[SampledInteraction], DenseMatrix[Float]) = {

      val samples = interactions.map { case Interaction(userId, itemId, userctxFeatures) =>
        SampledInteraction(userId, itemId, random.nextInt(numItems), userctxFeatures)
      }

      val batchSize = samples.length
      val na = Array.ofDim[Float](batchSize, batchSize)
      cforRange(0 until batchSize)(i => {
        val s = samples(i)
        val userId = s.userId
        val negitemId = s.negitemId
        cforRange(0 until batchSize)(j => {
          if (userSamplings.get(userId).contains(item2sampling(negitemId))) {
            na(i)(j) = 1.0f
          }
        })
      })

      (samples, DenseMatrix(na :_*))
    }

    crossbatchSample
  }

  /**
   * Computes the general BPR gradient
   *
   * @param sLinear The sums of the linear weights per training instance
   * @param fFactors The dot products of the latent factors per training instance
   * @return The general BPR gradients per training instance
   */
  private def computeBPRGradients(sLinear: Array[Float], fFactors: Array[Float]): Array[Float] = {
    sigmoid(-(DenseVector(sLinear) + DenseVector(fFactors))).toArray
  }

  /**
   * Computes gradients according to crossbatch-BPR loss.
   * This can be computationally expensive (batchSize*batchSize*numDims) as a matrix multiplication is required.
   * An optimized BLAS library is therefore recommended
   *
   * @param sLinear The sums of the linear weights per training instance
   * @param sFactors The sums of the latent factors per training instance
   * @param na A matrix of shape [batchSize, batchSize] with ones for non-accepted negative items and zeros otherwise
   * @return The linear weight and latent factors gradients per training instance
   */
  private def computeCrossbatchBPRGradients(sLinear: Array[Float],
                                            sFactors: Array[Array[Float]],
                                            na: DenseMatrix[Float]): (Array[Float], Array[Array[Float]]) = {

    // parameter server arrays to vectors and matrices
    val sLength = sLinear.length / 2
    val sPosItemsVector = DenseVector(sLinear.slice(0, sLength))
    val sNegItemsVector = DenseVector(sLinear.slice(sLength, sLength * 2))
    val sUsersMatrix = DenseMatrix(sFactors.slice(0, sLength) :_*)
    val sPosItemsMatrix = DenseMatrix(sFactors.slice(sLength, sLength * 2) :_*)
    val sNegItemsMatrix = DenseMatrix(sFactors.slice(sLength * 2, sLength * 3) :_*)

    // BPR utility, with non-accepted negatives removed
    val nbprMatrix = {
      val yhatPos_ = sUsersMatrix *:* sPosItemsMatrix
      val yhatPos = breezeSum(yhatPos_(*,::)) + sPosItemsVector
      val yhatNeg_ = sUsersMatrix * sNegItemsMatrix.t
      val yhatNeg = -(yhatNeg_(*,::) + sNegItemsVector)
      val gz = yhatNeg(::,*) + yhatPos
      val bpr = sigmoid(-gz)
      if (na.rows > 1) bpr - na *:* bpr else bpr
    }

    // linear gradients
    val gPosItemsVector = breezeSum(nbprMatrix(*,::))
    val gNegItemsVector = -breezeSum(nbprMatrix(::,*))

    // factors gradients
    val gUsersMatrix = {
      val gPos = sPosItemsMatrix(::,*) *:* gPosItemsVector
      val gNeg = nbprMatrix * sNegItemsMatrix
      gPos - gNeg
    }
    val gPosItemsMatrix = sUsersMatrix(::,*) *:* gPosItemsVector
    val gNegItemsMatrix = -(nbprMatrix.t * sUsersMatrix)

    // gradients to arrays required for parameter servers, transpose for row-major instead of column-major
    val gCatVector = DenseVector.vertcat(gPosItemsVector, gNegItemsVector.t) / sLength.toFloat
    val gCatMatrix = DenseMatrix.vertcat(gUsersMatrix, gPosItemsMatrix, gNegItemsMatrix) / sLength.toFloat
    (gCatVector.toArray, gCatMatrix.t.toArray.grouped(gCatMatrix.cols).toArray)
  }
}

class GlintFMPairModel private[ml](override val uid: String,
                                   private[spark] val bcItemFeatures: Broadcast[Array[SparseVector]],
                                   private[spark] val linear: BigFMPairVector,
                                   private[spark] val factors: BigFMPairMatrix,
                                   @transient private[spark] val client: Client)
  extends Model[GlintFMPairModel] with GlintFMPairParams with MLWritable {

  /** @group setParam */
  def setFilterItemsCol(value: String): this.type = set(filterItemsCol, value)

  @transient
  implicit private lazy val ec: ExecutionContext = ExecutionContext.Implicits.global

  override def transform(dataset: Dataset[_]): DataFrame = {
    throw new NotImplementedError()  // TODO
  }

  override def transformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, getUserCol, IntegerType)
    SchemaUtils.checkColumnType(schema, getUserctxfeaturesCol, new VectorUDT)
    SchemaUtils.appendColumn(schema, getPredictionCol, FloatType)
  }

  // TODO
  override def copy(extra: ParamMap): GlintFMPairModel = {
    val copied = new GlintFMPairModel(uid, bcItemFeatures, linear, factors, client)
    copyValues(copied, extra).setParent(parent)
  }

  override def write: MLWriter = new GlintFMPairModel.GlintFMPairModelWriter(this)

  /**
   * Returns top numItems items recommended for each user id in the input data set
   *
   * @param dataset The dataset containing a column of user ids and user context features. The column names must match
   *                userCol, userctxFeaturesCol and, if filtering should be used, also filterItemsCol.
   * @param numItems The maximum number of recommendations for each user
   * @return A dataframe of (userCol: Int, recommendations), where recommendations are stored
   *         as an array of (itemCol: Int, score: Float) rows.
   */
  def recommendForUserSubset(dataset: Dataset[_], numItems: Int): DataFrame = {

    val recommendType = ArrayType(new StructType().add(getItemCol, IntegerType).add("score", FloatType))
    val recommendSchema = new StructType().add(getUserCol, IntegerType).add("recommendations", recommendType)
    val rowEncoder = RowEncoder(recommendSchema)

    toDf(dataset).mapPartitions { iter =>

      if (iter.isEmpty) {
        Iterator.empty  // handle empty partition

      } else {
        val (userIds, userItemIds, userIndices, userWeights, numArgtopItems) = toArrays(iter, numItems)
        val batchSize = math.max(getBatchSize, numArgtopItems)

        // pull sums of user features
        val topFuture = factors.pullSum(userIndices, userWeights, false).flatMap { case (userFactors, _) =>

          val userMatrix = DenseMatrix(userFactors  :_*)

          val initialScoresMatrix = DenseMatrix.zeros[Float](userMatrix.rows, 0)
          val initialArgMatrix = DenseMatrix.zeros[Int](userMatrix.rows, 0)
          val initialBatchFuture = Future.successful((initialScoresMatrix, initialArgMatrix))

          bcItemFeatures.value
            .grouped(batchSize) // group item features into batches
            .zipWithIndex
            .foldLeft(initialBatchFuture) { case (prevBatchFuture, (itemBatch, i)) =>

              // convert features to the arrays required by the parameter servers
              val itemIndices = itemBatch.map(_.indices)
              val itemWeights = itemBatch.map(_.values.map(_.toFloat))

              val argBatchMatrix = tile(DenseVector((i * batchSize until (i + 1) * batchSize).toArray),
                1, userMatrix.rows).t

              // pull sums of item features of batch
              val itemLinearFuture = linear.pullSum(itemIndices, itemWeights, false)
              val itemFactorsFuture = factors.pullSum(itemIndices, itemWeights, false)

              // wait until communication with parameter servers for previous batches is finished
              var (scoresMatrix, argMatrix) = Await.result(prevBatchFuture, 5 minute)

              for {
                (itemLinear, _) <- itemLinearFuture
                (itemFactors, _) <- itemFactorsFuture
              } yield {
                // compute scores for users and all items of batch
                val itemVector = DenseVector(itemLinear)
                val itemMatrix = DenseMatrix(itemFactors :_*)
                val scoresBatchMatrix_ = userMatrix * itemMatrix.t
                val scoresBatchMatrix = scoresBatchMatrix_(*, ::) + itemVector

                // concatenate with previous top scores and keep top scores of concatenation
                val scoresCatMatrix = DenseMatrix.horzcat(scoresMatrix, scoresBatchMatrix)
                val argCatMatrix = DenseMatrix.horzcat(argMatrix, argBatchMatrix)
                if (scoresMatrix.cols == 0) {  // if first concat then initialize the top matrices properly now
                  scoresMatrix = DenseMatrix.zeros[Float](userMatrix.rows, numArgtopItems)
                  argMatrix = DenseMatrix.zeros[Int](userMatrix.rows, numArgtopItems)
                }
                val scoresArgtop = argtopk(scoresCatMatrix(*, ::), numArgtopItems)
                cforRange(0 until scoresMatrix.rows)(j => {
                  scoresMatrix(j, ::) := scoresCatMatrix(j, scoresArgtop(j))
                  argMatrix(j, ::) := argCatMatrix(j, scoresArgtop(j))
                })

                (scoresMatrix, argMatrix)
              }
            }
        }
        val numBatches = math.max(bcItemFeatures.value.length / batchSize, 5)
        val (scoresMatrix, argMatrix) = Await.result(topFuture, numBatches minutes)
        toRowIter(userIds, userItemIds, argMatrix, scoresMatrix, numItems)
      }
    }(rowEncoder).toDF(getUserCol, "recommendations")
    .select(col(getUserCol), col("recommendations").cast(recommendType))
  }

  /**
   * Converts a data set to a data frame required for this model. Filtering is considered.
   * If the features have separate index ranges, the user / context feature indices are shifted
   * so that they start after the item feature indices
   */
  private def toDf(dataset: Dataset[_]): DataFrame = {

    val numItemFeatures = linear.size.toInt
    val numFeatures = factors.rows.toInt

    var cols = Array(col(getUserCol))
    if (numItemFeatures == numFeatures) {
      cols = cols :+ col(getUserctxfeaturesCol)
    } else {
      val shift = udf((userctxFeatures: SparseVector) =>
        new SparseVector(numFeatures, userctxFeatures.indices.map(i => numItemFeatures + i), userctxFeatures.values))
      cols = cols :+ shift(col(getUserctxfeaturesCol)).as(getUserctxfeaturesCol)
    }
    if (getFilterItemsCol.nonEmpty) {
      cols = cols :+ col(getFilterItemsCol)
    }
    dataset.select(cols :_*)
  }

  /**
   * Converts a row iterator to arrays of user information. Filtering is considered
   */
  private def toArrays(iter: Iterator[Row], numItems: Int):
  (Array[Int], Array[BitSet], Array[Array[Int]], Array[Array[Float]], Int) = {

    val rows = iter.toArray

    val userIds = rows.map(_.getInt(0))
    val userctxFeatures = rows.map(row => row.getAs[SparseVector](1))
    val userIndices = userctxFeatures.map(_.indices)
    val userWeights = userctxFeatures.map(_.values.map(_.toFloat))

    val userItemIds = if (getFilterItemsCol.nonEmpty) rows.map(r => BitSet(r.getSeq[Int](2) :_*)) else new Array[BitSet](0)
    val numArgtopItems = if (getFilterItemsCol.nonEmpty) numItems + userItemIds.map(_.count(_ => true)).max else numItems

    (userIds, userItemIds, userIndices, userWeights, numArgtopItems)
  }

  /**
   * Converts arrays of user information and score matrices to an iterator of (userCol: Int, recommendations) rows,
   * where recommendations are stored as an array of (itemCol: Int, score: Float) rows.
   *
   * Only numItems recommendations are returned per user and user items are filtered if a filterItemsCol is set
   */
  private def toRowIter(userIds: Array[Int],
                        userItemIds: Array[BitSet],
                        argMatrix: DenseMatrix[Int],
                        scoresMatrix: DenseMatrix[Float],
                        numItems: Int): Iterator[Row] = {

    val userIter = if (getFilterItemsCol.nonEmpty) userIds.iterator.zip(userItemIds.iterator) else userIds.iterator
    userIter.zip(argMatrix(*,::).iterator.zip(scoresMatrix(*,::).iterator)).map {

      case ((userid: Int, userItemids: BitSet), (itemids, scores)) =>
        val recs = itemids.valuesIterator.zip(scores.valuesIterator)
          .filter { case (itemid, _) => !userItemids.contains(itemid) }
          .toArray
          .sortBy { case (_, score) => -score }
          .take(numItems)
          .map { case (itemid, score) => Row(itemid, score) }
        Row(userid, recs)

      case (userid: Int, (itemids, scores)) =>
        val recs = itemids.valuesIterator.zip(scores.valuesIterator)
          .toArray
          .sortBy { case (_, score) => -score }
          .map { case (itemid, score) => Row(itemid, score) }
        Row(userid, recs)
    }
  }

  /**
   * Destroys the model and releases the underlying distributed models and broadcasts.
   * This model can't be used anymore afterwards.
   *
   * @param terminateOtherClients If other clients should be terminated. This is necessary if a glint cluster in
   *                              another Spark application should be terminated.
   */
  def destroy(terminateOtherClients: Boolean = false): Unit = {
    linear.destroy()
    factors.destroy()
    client.terminateOnSpark(SparkSession.builder().getOrCreate().sparkContext, terminateOtherClients)
    bcItemFeatures.destroy()
  }
}


object GlintFMPairModel extends MLReadable[GlintFMPairModel] {

  private[GlintFMPairModel] class GlintFMPairModelWriter(instance: GlintFMPairModel)
    extends MLWriter {

    @transient
    implicit private lazy val ec: ExecutionContext = ExecutionContext.Implicits.global

    override protected def saveImpl(path: String): Unit = {
      DefaultParamsWriter.saveMetadata(instance, path, sc)
      val savedLinearFuture = instance.linear.save(path + "/linear", sc.hadoopConfiguration)
      val savedFactorsFuture = instance.factors.save(path + "/factors", sc.hadoopConfiguration)
      sc.parallelize(instance.bcItemFeatures.value, 1).saveAsObjectFile(path + "/itemfeatures")
      Await.result(Future.sequence(Seq(savedLinearFuture, savedFactorsFuture)), 5 minutes)
    }
  }

  private class GlintFMPairModelReader extends MLReader[GlintFMPairModel] {

    private val className = classOf[GlintFMPairModel].getName

    override def load(path: String): GlintFMPairModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc, className)
      val parameterServerHost = metadata.getParamValue("parameterServerHost").values.asInstanceOf[String]
      val parameterServerConfig = ConfigFactory.parseMap(toJavaPathMap(
        metadata.getParamValue("parameterServerConfig").values.asInstanceOf[Map[String, _]]))
      load(metadata, path, parameterServerHost, parameterServerConfig)
    }

    def load(path: String, parameterServerHost: String): GlintFMPairModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc, className)
      val parameterServerConfig = ConfigFactory.parseMap(toJavaPathMap(
        metadata.getParamValue("parameterServerConfig").values.asInstanceOf[Map[String, _]]))
      load(metadata, path, parameterServerHost, parameterServerConfig)
    }

    def load(path: String, parameterServerHost: String, parameterServerConfig: Config): GlintFMPairModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc, className)
      load(metadata, path, parameterServerHost, parameterServerConfig)
    }

    def load(metadata: DefaultParamsReader.Metadata,
             path: String,
             parameterServerHost: String,
             parameterServerConfig: Config): GlintFMPairModel = {

      val config = Client.getHostConfig(parameterServerHost).withFallback(parameterServerConfig)
      val client = if (parameterServerHost.isEmpty) {
        Client.runOnSpark(sc, config, Client.getNumExecutors(sc), Client.getExecutorCores(sc))
      } else {
        Client(config)
      }

      try {
        val numPartitions = Client.getNumExecutors(sc) * Client.getExecutorCores(sc)
        val linear = client.loadFMPairVector(path + "/linear", sc.hadoopConfiguration, numPartitions)
        val factors = client.loadFMPairMatrix(path + "/factors", sc.hadoopConfiguration, numPartitions)

        val itemFeatures = sc.objectFile[SparseVector](path + "/itemfeatures", minPartitions = 1).collect()
        val bcItemFeatures = sc.broadcast(itemFeatures)

        val model = new GlintFMPairModel(metadata.uid, bcItemFeatures, linear, factors, client)
        metadata.getAndSetParams(model)
        model.set(model.parameterServerHost, parameterServerHost)
        model.set(model.parameterServerConfig, parameterServerConfig.resolve())
        model
      } catch { case e: Exception =>
        client.terminateOnSpark(sc)
        throw e
      }
    }

    private def toJavaPathMap(map: Map[String, _],
                              pathMap: java.util.Map[String, Object] = new java.util.HashMap[String, Object](),
                              root: String = ""): java.util.Map[String, Object] = {
      map.foreach {
        case (key: String, value: Map[String, _]) =>
          val path = if (root == "") key else s"$root.$key"
          toJavaPathMap(value, pathMap, path)
        case (key: String, value: Object) =>
          val path = if (root == "") key else s"$root.$key"
          pathMap.put(path, value)
      }
      pathMap
    }
  }

  override def read: MLReader[GlintFMPairModel] = new GlintFMPairModelReader

  /**
   * Loads a [[org.apache.spark.ml.recommendation.GlintFMPairModel GlintFMPairModel]]
   * and either starts a parameter server cluster in this Spark application or connects to running parameter servers
   * depending on the saved parameter server host and parameter server configuration.
   *
   * @param path The path
   * @return The [[org.apache.spark.ml.recommendation.GlintFMPairModel GlintFMPairModel]]
   */
  override def load(path: String): GlintFMPairModel = super.load(path)

  /**
   * Loads a [[org.apache.spark.ml.recommendation.GlintFMPairModel GlintFMPairModel]] and uses
   * the saved parameter server configuration.
   *
   * @param path The path
   * @param parameterServerHost The master host of the running parameter servers. If this is not set a standalone
   *                            parameter server cluster is started in this Spark application.
   * @return The [[org.apache.spark.ml.recommendation.GlintFMPairModel GlintFMPairModel]]
   */
  def load(path: String, parameterServerHost: String): GlintFMPairModel = {
    new GlintFMPairModelReader().load(path, parameterServerHost)
  }

  /**
   * Loads a [[org.apache.spark.ml.recommendation.GlintFMPairModel GlintFMPairModel]]
   *
   * @param path The path
   * @param parameterServerHost The master host of the running parameter servers. If this is not set a standalone
   *                            parameter server cluster is started in this Spark application.
   * @param parameterServerConfig The parameter server configuration
   * @return The [[org.apache.spark.ml.recommendation.GlintFMPairModel GlintFMPairModel]]
   */
  def load(path: String, parameterServerHost: String, parameterServerConfig: Config): GlintFMPairModel = {
    new GlintFMPairModelReader().load(path, parameterServerHost, parameterServerConfig)
  }
}
