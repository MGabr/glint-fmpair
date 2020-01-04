package org.apache.spark.ml.recommendation

import breeze.linalg.{sum => breezeSum, _}
import breeze.numerics.sigmoid
import com.typesafe.config._
import glint.models.client.{BigFMPairMatrix, BigFMPairVector}
import glint.{Client, FMPairArguments}
import org.apache.spark.TaskContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.linalg.{SparseVector, VectorUDT}
import org.apache.spark.ml.param.shared.{HasMaxIter, HasPredictionCol, HasSeed, HasStepSize}
import org.apache.spark.ml.param._
import org.apache.spark.ml.recommendation.ServerSideGlintFMPair.{Interaction, SampledFeatures, SampledInteraction}
import org.apache.spark.ml.util._
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.sql.catalyst.encoders.RowEncoder
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{ArrayType, FloatType, IntegerType, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.eclipse.collections.impl.map.mutable.primitive.IntObjectHashMap
import spire.implicits.cforRange

import scala.collection.BitSet
import scala.concurrent.duration._
import scala.concurrent.{Await, ExecutionContext, Future}
import scala.util.Random

private[recommendation] trait ServerSideGlintFMPairParams extends Params with HasMaxIter with HasStepSize with HasSeed
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
    "the name of the column to use for sampling aceptance, usually same as itemCol")
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
  final val filterItemsCol = new Param[String](this, "userItems", "")
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
   * The mini-batch size
   * Default: 4096
   *
   * @group param
   */
  final val batchSize = new IntParam(this, "batchSize", "the mini-batch size", ParamValidators.gt(0))
  setDefault(batchSize -> 4096)

  /** @group getParam */
  def getBatchSize: Int = $(batchSize)

  /**
   * The number of latent factor dimensions (k)
   * Default: 50
   *
   * @group param
   */
  final val numDims = new IntParam(this, "numDims", "the number of dimensions (k)", ParamValidators.gt(0))
  setDefault(numDims -> 50)

  /** @group getParam */
  def getNumDims: Int = $(numDims)

  /**
   * The regularization rate for the linear weights
   * Default: 0.01f
   *
   * @group param
   */
  final val linearReg = new FloatParam(this, "linearReg", "the regularization rate for the linear weights", ParamValidators.gtEq(0))
  setDefault(linearReg -> 0.01f)

  /** @group getParam */
  def getLinearReg: Float = $(linearReg)

  /**
   * The regularization rate for the latent factor weights
   * Default: 0.001f
   *
   * @group param
   */
  final val factorsReg = new FloatParam(this, "factorsReg", "the regularization rate for the factor weights", ParamValidators.gtEq(0))
  setDefault(factorsReg -> 0.001f)

  /** @group getParam */
  def getFactorsReg: Float = $(factorsReg)

  /**
   * The initialization mean for the linear and latent factor weights
   * Default: 0.1f
   *
   * @group param
   */
  final val initMean = new FloatParam(this, "initMean", "the initialization mean for model weights", ParamValidators.gtEq(0))
  setDefault(initMean -> 0.1f)

  /** @group getParam */
  def getInitMean: Float = $(initMean)


  /**
   * The number of parameter servers to create
   * Default: 5
   *
   * @group param
   */
  final val numParameterServers = new IntParam(this, "numParameterServers",
    "the number of parameter servers to create")
  setDefault(numParameterServers -> 5)

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
}


class ServerSideGlintFMPair(override val uid: String)
  extends Estimator[ServerSideGlintFMPairModel] with ServerSideGlintFMPairParams with DefaultParamsWritable {

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
  def setInitMean(value: Float): this.type = set(initMean, value)


  /** @group setParam */
  def setNumParameterServers(value: Int): this.type = set(numParameterServers, value)

  /** @group setParam */
  def setParameterServerHost(value: String): this.type = set(parameterServerHost, value)

  /** @group setParam */
  def setParameterServerConfig(value: Config): this.type = set(parameterServerConfig, value.resolve())


  override def fit(dataset: Dataset[_]): ServerSideGlintFMPairModel = {

    val args = FMPairArguments(getNumDims, getBatchSize, getStepSize.toFloat, getLinearReg, getFactorsReg)

    // get number of features and estimation for average number of features from a sampled first row
    val sampleRow = dataset.select(getUserctxfeaturesCol, getItemfeaturesCol).first()
    val sampleUserctxFeature = sampleRow.getAs[SparseVector](0)
    val sampleItemFeature = sampleRow.getAs[SparseVector](1)
    val numFeatures = sampleUserctxFeature.size + sampleItemFeature.size
    val avgActiveItemFeatures = sampleItemFeature.indices.length * 2
    val avgActiveFeatures = sampleUserctxFeature.indices.length * 2 + avgActiveItemFeatures

    // get number of possible partitions from available Spark resources
    val spark = dataset.sparkSession
    val sc = spark.sparkContext
    val numPartitions = Client.getNumExecutors(sc) * Client.getExecutorCores(sc)

    import spark.implicits._

    @transient
    implicit val ec = ExecutionContext.Implicits.global

    @transient
    val client = if (getParameterServerHost.isEmpty) {
      Client.runOnSpark(sc, getParameterServerConfig, getNumParameterServers, Client.getExecutorCores(sc))
    } else {
      Client(Client.getHostConfig(getParameterServerHost).withFallback(getParameterServerConfig))
    }

    @transient
    val linear = client.fmpairVector(args, numFeatures, avgActiveItemFeatures, 1)
    @transient
    val factors = client.fmpairMatrix(args, numFeatures, avgActiveFeatures, getNumParameterServers)

    // create and broadcast mapping array of item ids to item features
    val bcItem2features = dataset.sqlContext.sparkContext.broadcast {
      dataset
        .select(getItemCol, getItemfeaturesCol)
        .groupBy(getItemCol)
        .agg(first(getItemfeaturesCol).as(getItemfeaturesCol))
        .sort(getItemCol)
        .collect()
        .map(_.getAs[SparseVector](getItemfeaturesCol))
    }

    // create and broadcast mapping array of item ids to sampling ids - if accepted sampling / sampling column is used
    val bcItem2sampling = dataset.sqlContext.sparkContext.broadcast {
      if (getSamplingCol.nonEmpty) {
        Some(dataset
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

    // create and broadcast array of item ids sorted by their counts - if exp sampling is used
    val bcItemsByCount = dataset.sqlContext.sparkContext.broadcast {
      if (getSampler.equals("exp")) {
        Some(dataset
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

    dataset
      .select(getUserCol, getItemCol, getUserctxfeaturesCol)
      .repartition(numPartitions, col(getUserCol))  // partition data by users
      .map(row => Interaction(row.getInt(0), row.getInt(1), row.getAs[SparseVector](2)))
      .foreachPartition((iter: Iterator[Interaction]) => {

        @transient
        implicit val ec = ExecutionContext.Implicits.global

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
          val us = new IntObjectHashMap[BitSet]()  // instead of scala map, essential for performance
          interactions.groupBy(_.userId).foreach { case (userId, items) =>
            us.put(userId, BitSet(items.map(i => i2s(i.itemId)) :_*))
          }
          us
        })

        // create sampling function to use
        val sample = if (item2sampling.isDefined) {
          sampler match {
            case "uniform" => ServerSideGlintFMPair.uniformSampler(userSamplings.get, item2sampling.get)
            case "exp" => ServerSideGlintFMPair.expSampler(userSamplings.get, item2sampling.get, itemsByCount.get, getRho)
            case "crossbatch" => ServerSideGlintFMPair.crossbatchSampler(userSamplings.get, item2sampling.get)
          }
        } else {
          sampler match {
            case "uniform" | "crossbatch" => ServerSideGlintFMPair.uniformAllSampler(item2features.length)
            case "exp" => ServerSideGlintFMPair.expAllSampler(itemsByCount.get, getRho)
          }
        }

        // now the actual fit pipeline
        val fitFinishedFuture = (0 until epochs).iterator
          .flatMap { epoch =>

            if (epoch % 10 == 0) {
              logInfo(s"Epoch $epoch of $epochs")
            }

            val random = new Random(seed + TaskContext.getPartitionId() + epoch)
            ServerSideGlintFMPair.shuffle(random, interactions)  // sample positive users, contexts and items
            interactions
              .grouped(batchSize)  // group into batches
              .map(i => sample(random, i))  // sample negative items

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
            val iItem = batch.map(i => i.positemFeatures.indices ++ i.negitemFeatures.indices)
            val wItem = batch.map(i =>
              i.positemFeatures.values.map(_.toFloat) ++ i.negitemFeatures.values.map(v => (-v).toFloat))
            (iUser, wUser, iItem, wItem, na)

          }.foldLeft(Future.successful(Seq(true))) { case (prevBatchFuture, (iUser, wUser, iItem, wItem, na)) =>
            // wait until communication with parameter servers for previous batches is finished
            // this allows already pre-processing the next batch while waiting for the parameter server responses
            Await.ready(prevBatchFuture, 1 minute)

            // communicate with the parameter servers for SGD step
            sampler match {
              case "uniform" | "exp" =>  // "normal" BPR loss
                val batchFutureLinear = linear.pullSum(iItem, wItem)
                val batchFutureFactors = factors.dotprod(iUser, wUser, iItem, wItem)
                val batchFuture = for {
                  (sLinear, cacheKeysLinear) <- batchFutureLinear
                  (fFactors, cacheKeysFactors) <- batchFutureFactors
                } yield {
                  val g = ServerSideGlintFMPair.computeBPRGradients(sLinear, fFactors)
                  Future.sequence(Seq(linear.pushSum(g, cacheKeysLinear), factors.adjust(g, cacheKeysFactors)))
                }
                batchFuture.flatMap(identity)

              case "crossbatch" =>  // crossbatch-BPR loss
                val batchFutureLinear = linear.pullSum(iItem, wItem)
                val batchFutureFactors = factors.pullSum(iUser ++ iItem, wUser ++ wItem)
                val batchFuture = for {
                  (sLinear, cacheKeysLinear) <- batchFutureLinear
                  (sFactors, cacheKeysFactors) <- batchFutureFactors
                } yield {
                  val (gLinear, gFactors) = ServerSideGlintFMPair.computeCrossbatchBPRGradients(sLinear, sFactors, na)
                  Future.sequence(Seq(
                    linear.pushSum(gLinear, cacheKeysLinear),
                    factors.pushSum(gFactors, cacheKeysFactors)))
                }
                batchFuture.flatMap(identity)
            }
          }

        // wait until communication with parameter servers for last batch is finished
        Await.ready(fitFinishedFuture, 1 minute)
        ()
      })

    copyValues(new ServerSideGlintFMPairModel(this.uid, bcItem2features, linear, factors, client).setParent(this))
  }

  override def copy(extra: ParamMap): Estimator[ServerSideGlintFMPairModel] = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, getUserCol, IntegerType)
    SchemaUtils.checkColumnType(schema, getUserctxfeaturesCol, new VectorUDT)
    SchemaUtils.checkColumnType(schema, getItemCol, IntegerType)
    SchemaUtils.checkColumnType(schema, getItemfeaturesCol, new VectorUDT)
    schema
  }
}


object ServerSideGlintFMPair extends DefaultParamsReadable[ServerSideGlintFMPair] {

  /** An interaction from the training instances */
  private case class Interaction(userId: Int, itemId: Int, userctxFeatures: SparseVector)

  /** An interaction from the training instances, combined with a sampled negative interaction */
  private case class SampledInteraction(userId: Int, positemId: Int, negitemId: Int, userctxFeatures: SparseVector)

  /** The features of a positive interaction from the training instances and a negative interaction */
  private case class SampledFeatures(userctxFeatures: SparseVector,
                                     positemFeatures: SparseVector,
                                     negitemFeatures: SparseVector)

  /**
   * Sample positive items / interactions by shuffling the interactions in-place.
   * Uses Fisher-Yates shuffle algorithm
   *
   * @param random The random number generator to use
   * @param interactions The interactions to shuffle
   */
  private def shuffle(random: Random, interactions: Array[Interaction]): Unit = {
    cforRange(interactions.length - 1 to 0 by -1)(i => {
      val j = random.nextInt(i + 1)
      val tmp = interactions(j)
      interactions(j) = interactions(i)
      interactions(i) = tmp
    })
  }

  /**
   * Sample negative items uniformly, accepting all items as negative items
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
   * Sample negative items with probabilities proportional to the exponential popularity distribution from LambdaFM,
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
   * Sample negative items uniformly
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
   * Sample negative items with probabilities proportional to the exponential popularity distribution from LambdaFM.
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
   * Sample negative items uniformly and return matrix of non-accepted negative items across batch
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
        cforRange(0 until batchSize)(j => {
          if (userSamplings.get(userId).contains(item2sampling(s.negitemId))) {
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
    sigmoid(1e-10f - (DenseVector(sLinear) + DenseVector(fFactors))).toArray
  }

  /**
   * Compute gradients according to crossbatch-BPR loss.
   * This can be computationally expensive (batchSize*batchSize*numDims) as a matrix multiplication is required.
   * An optimized BLAS library is therefore recommended
   *
   * @param sLinear The sums of the linear weights per training instance
   * @param sFactors The sums of the latent factors per training instance
   * @param na A matrix of shape [batchSize * batchSize] with ones for non-accepted negative items and zeros otherwise
   * @return The linear weight and latent factors gradients per training instance
   */
  private def computeCrossbatchBPRGradients(sLinear: Array[Float],
                                            sFactors: Array[Array[Float]],
                                            na: DenseMatrix[Float]): (Array[Float], Array[Array[Float]]) = {
    // BPR utility
    val sLength = sFactors.length / 2
    val sLinearVector = DenseVector(sLinear)
    val sUsersMatrix = DenseMatrix(sFactors.slice(0, sLength) :_*)
    val sItemsMatrix = DenseMatrix(sFactors.slice(sLength, sLength * 2) :_*)
    val gMatrix_ = sUsersMatrix * sItemsMatrix.t
    val gMatrix = sigmoid(1e-10f - (gMatrix_(*,::) + sLinearVector))

    // BPR utility with non-accepted negatives removed
    val ngMatrix = if (na.rows > 1) gMatrix - na *:* gMatrix else gMatrix

    // linear gradient
    val gVector = breezeSum(ngMatrix(*,::))  / sLength.toFloat

    // factors gradient
    val gUsersMatrix = ngMatrix * sItemsMatrix
    val gItemsMatrix = ngMatrix.t * sUsersMatrix
    val gCatMatrix = DenseMatrix.vertcat(gUsersMatrix, gItemsMatrix) / sLength.toFloat

    // transpose for row-major instead of column-major
    (gVector.toArray, gCatMatrix.t.toArray.grouped(gCatMatrix.cols).toArray)
  }
}

class ServerSideGlintFMPairModel private[ml](override val uid: String,
                                             private[spark] val bcItemFeatures: Broadcast[Array[SparseVector]],
                                             private[spark] val linear: BigFMPairVector,
                                             private[spark] val factors: BigFMPairMatrix,
                                             @transient private[spark] val client: Client)
  extends Model[ServerSideGlintFMPairModel] with ServerSideGlintFMPairParams with MLWritable {

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
  override def copy(extra: ParamMap): ServerSideGlintFMPairModel = {
    val copied = new ServerSideGlintFMPairModel(uid, bcItemFeatures, linear, factors, client)
    copyValues(copied, extra).setParent(parent)
  }

  override def write: MLWriter = new ServerSideGlintFMPairModel.ServerSideGlintFMPairModelWriter(this)

  /**
   * Converts a dataset to a data frame required for this model. Filtering is considered
   */
  private def toDf(dataset: Dataset[_]): DataFrame = {
    if (getFilterItemsCol.nonEmpty) {
      dataset.select(getUserCol, getUserctxfeaturesCol, getFilterItemsCol)
    } else {
      dataset.select(getUserCol, getUserctxfeaturesCol)
    }
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
    val numArgtopItems = if (getFilterItemsCol.nonEmpty) numItems + userItemIds.map(_.size).max else numItems

    (userIds, userItemIds, userIndices, userWeights, numArgtopItems)
  }

  /**
   * Converts arrays of user information and score matrices to an iterator of (userCol: Int, recommendations) rows,
   * where recommendations are stored as an array of (itemCol: Int, score: Float) rows.
   *
   * Only numItems recommendations are returned per user. User items are filtered if filterUserItems is set.
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
          .take(numItems)
          .map { case (itemid, score) => Row(itemid, score) }.toArray
        Row(userid, recs)

      case (userid: Int, (itemids, scores)) =>
        val recs = itemids.valuesIterator.zip(scores.valuesIterator)
          .map { case (itemid, score) => Row(itemid, score) }.toArray
        Row(userid, recs)
    }
  }

  /**
   * Returns top numItems items recommended for each user id in the input data set
   *
   * @param dataset The dataset containing a column of user ids and user context features. The column names must match
   *                userCol, userctxfeaturesCol and, if filtering should be used, also filterItemsCol.
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

          val userMatrix = DenseMatrix(userFactors: _*)

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

              // wait until communication with parameter servers for previous batches is finished
              // this allows already pre-processing the next batch while waiting for the parameter server responses
              var (scoresMatrix, argMatrix) = Await.result(prevBatchFuture, 1 minute)

              // pull sums of item features of batch
              val itemLinearFuture = linear.pullSum(itemIndices, itemWeights, false)
              val itemFactorsFuture = factors.pullSum(itemIndices, itemWeights, false)

              for {
                (itemLinear, _) <- itemLinearFuture
                (itemFactors, _) <- itemFactorsFuture
              } yield {
                // compute scores for users and all items of batch
                val itemVector = DenseVector(itemLinear)
                val itemMatrix = DenseMatrix(itemFactors: _*)
                val scoresBatchMatrix_ = userMatrix * itemMatrix.t
                val scoresBatchMatrix = scoresBatchMatrix_(*, ::) + itemVector

                // concatenate with previous top scores and keep top scores of concatenation
                val scoresCatMatrix = DenseMatrix.horzcat(scoresMatrix, scoresBatchMatrix)
                val argCatMatrix = DenseMatrix.horzcat(argMatrix, argBatchMatrix)
                if (scoresMatrix.cols == 0) { // if first concat then initialize the top matrices properly now
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
        val (scoresMatrix, argMatrix) = Await.result(topFuture, 5 minutes)
        toRowIter(userIds, userItemIds, argMatrix, scoresMatrix, numItems)
      }
    }(rowEncoder).toDF(getUserCol, "recommendations")
    .select(col(getUserCol), col("recommendations").cast(recommendType))
  }

  /**
   * Stops the model and releases the underlying distributed models and broadcasts.
   * This model can't be used anymore afterwards.
   *
   * @param terminateOtherClients If other clients should be terminated. This is necessary if a glint cluster in
   *                              another Spark application should be terminated.
   */
  def stop(terminateOtherClients: Boolean = false): Unit = {
    linear.destroy()
    factors.destroy()
    client.terminateOnSpark(SparkSession.builder().getOrCreate().sparkContext, terminateOtherClients)
    bcItemFeatures.destroy()
  }
}


object ServerSideGlintFMPairModel extends MLReadable[ServerSideGlintFMPairModel] {

  private[ServerSideGlintFMPairModel] class ServerSideGlintFMPairModelWriter(instance: ServerSideGlintFMPairModel)
    extends MLWriter {

    @transient
    implicit private lazy val ec: ExecutionContext = ExecutionContext.Implicits.global

    override protected def saveImpl(path: String): Unit = {
      DefaultParamsWriter.saveMetadata(instance, path, sc)
      val savedLinearFuture = instance.linear.save(path + "/linear", sc.hadoopConfiguration)
      val savedFactorsFuture = instance.factors.save(path + "/factors", sc.hadoopConfiguration)
      sc.parallelize(instance.bcItemFeatures.value, 1).saveAsObjectFile(path + "/itemfeatures")
      Await.ready(Future.sequence(Seq(savedLinearFuture, savedFactorsFuture)), 5 minutes)
    }
  }

  private class ServerSideGlintFMPairModelReader extends MLReader[ServerSideGlintFMPairModel] {

    private val className = classOf[ServerSideGlintFMPairModel].getName

    override def load(path: String): ServerSideGlintFMPairModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc, className)
      val parameterServerHost = metadata.getParamValue("parameterServerHost").values.asInstanceOf[String]
      val parameterServerConfig = ConfigFactory.parseMap(toJavaPathMap(
        metadata.getParamValue("parameterServerConfig").values.asInstanceOf[Map[String, _]]))
      load(metadata, path, parameterServerHost, parameterServerConfig)
    }

    def load(path: String, parameterServerHost: String): ServerSideGlintFMPairModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc, className)
      val parameterServerConfig = ConfigFactory.parseMap(toJavaPathMap(
        metadata.getParamValue("parameterServerConfig").values.asInstanceOf[Map[String, _]]))
      load(metadata, path, parameterServerHost, parameterServerConfig)
    }

    def load(path: String, parameterServerHost: String, parameterServerConfig: Config): ServerSideGlintFMPairModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc, className)
      load(metadata, path, parameterServerHost, parameterServerConfig)
    }

    def load(metadata: DefaultParamsReader.Metadata,
             path: String,
             parameterServerHost: String,
             parameterServerConfig: Config): ServerSideGlintFMPairModel = {
      val config = Client.getHostConfig(parameterServerHost).withFallback(parameterServerConfig)
      val client = if (parameterServerHost.isEmpty) {
        Client.runOnSpark(sc, config, Client.getNumExecutors(sc), Client.getExecutorCores(sc))
      } else {
        Client(config)
      }
      val linear = client.loadFMPairVector(path + "/linear", sc.hadoopConfiguration)
      val factors = client.loadFMPairMatrix(path + "/factors", sc.hadoopConfiguration)
      val itemFeatures = sc.objectFile[SparseVector](path + "/itemfeatures", minPartitions = 1).collect()
      val bcItemFeatures = sc.broadcast(itemFeatures)
      val model = new ServerSideGlintFMPairModel(metadata.uid, bcItemFeatures, linear, factors, client)
      metadata.getAndSetParams(model)
      model.set(model.parameterServerHost, parameterServerHost)
      model.set(model.parameterServerConfig, parameterServerConfig.resolve())
      model
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

  override def read: MLReader[ServerSideGlintFMPairModel] = new ServerSideGlintFMPairModelReader

  /**
   * Loads a [[org.apache.spark.ml.recommendation.ServerSideGlintFMPairModel ServerSideGlintFMPairModel]]
   * and either starts a parameter server cluster in this Spark application or connects to running parameter servers
   * depending on the saved parameter server host and parameter server configuration.
   *
   * @param path The path
   * @return The [[org.apache.spark.ml.recommendation.ServerSideGlintFMPairModel ServerSideGlintFMPairModel]]
   */
  override def load(path: String): ServerSideGlintFMPairModel = super.load(path)

  /**
   * Loads a [[org.apache.spark.ml.recommendation.ServerSideGlintFMPairModel ServerSideGlintFMPairModel]] and uses
   * the saved parameter server configuration.
   *
   * @param path The path
   * @param parameterServerHost The master host of the running parameter servers. If this is not set a standalone
   *                            parameter server cluster is started in this Spark application.
   * @return The [[org.apache.spark.ml.recommendation.ServerSideGlintFMPairModel ServerSideGlintFMPairModel]]
   */
  def load(path: String, parameterServerHost: String): ServerSideGlintFMPairModel = {
    new ServerSideGlintFMPairModelReader().load(path, parameterServerHost)
  }

  /**
   * Loads a [[org.apache.spark.ml.recommendation.ServerSideGlintFMPairModel ServerSideGlintFMPairModel]]
   *
   * @param path The path
   * @param parameterServerHost The master host of the running parameter servers. If this is not set a standalone
   *                            parameter server cluster is started in this Spark application.
   * @param parameterServerConfig The parameter server configuration
   * @return The [[org.apache.spark.ml.recommendation.ServerSideGlintFMPairModel ServerSideGlintFMPairModel]]
   */
  def load(path: String, parameterServerHost: String, parameterServerConfig: Config): ServerSideGlintFMPairModel = {
    new ServerSideGlintFMPairModelReader().load(path, parameterServerHost, parameterServerConfig)
  }
}
