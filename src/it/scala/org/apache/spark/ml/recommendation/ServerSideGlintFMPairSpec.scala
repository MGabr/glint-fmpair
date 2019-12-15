package org.apache.spark.ml.recommendation

import java.net.InetAddress

import com.typesafe.config.ConfigFactory
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, FileUtil, Path}
import org.apache.spark.ml.feature.{OneHotEncoderEstimator, OneHotEncoderModel, VectorAssembler}
import org.apache.spark.ml.param.{ParamMap, ParamPair}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.scalatest.{BeforeAndAfterAll, FlatSpec, Inspectors, Matchers}
import org.scalatest.concurrent.ScalaFutures

import scala.collection.mutable
import scala.concurrent.ExecutionContext

object ServerSideGlintFMPairSpec {

  /**
   * Helper function to load preprocessed csv data from path as dataframe
   */
  def load(s: SparkSession, dataPath: String): DataFrame = {
    s.read.format("csv").option("header", "true").option("inferSchema", "true").load(dataPath)
  }

  /**
   * Helper function to convert loaded train dataframe into one-hot encoded user and item feature vectors.
   * Returns fitted one-hot encoders to use for converting loaded test dataframe
   */
  def toFeatures(data: DataFrame): (DataFrame, OneHotEncoderModel, OneHotEncoderModel, OneHotEncoderModel) = {

    val userCols = Array("pid", "userid", "category")
    val userEncoderModel = new OneHotEncoderEstimator()
      .setInputCols(userCols)
      .setOutputCols(userCols.map(_ + "_encoded"))
      .setDropLast(true)  // missing user features are simply ignored, no problem for ranking
      .fit(data)

    val itemCols = Array("traid", "albid", "artid", "year")
    val itemEncoderModel = new OneHotEncoderEstimator()
      .setInputCols(itemCols)
      .setOutputCols(itemCols.map(_ + "_encoded"))
      .setDropLast(false)  // missing item features are mapped to the missing feature index
      .setHandleInvalid("keep")  // unknown item features are mapped to the unknown (not missing!) feature index - not optimal
      .fit(data)

    val ctxitemCols = Array("prev_traid", "prev_albid", "prev_artid", "prev_year")
    val ctxitemEncoderModel = itemEncoderModel.copy(ParamMap(
      ParamPair(itemEncoderModel.inputCols, ctxitemCols),
      ParamPair(itemEncoderModel.outputCols, ctxitemCols.map(_ + "_encoded")),
      ParamPair(itemEncoderModel.dropLast, true)))  // missing user features are simply ignored, no problem for ranking

    (toFeatures(data, userEncoderModel, itemEncoderModel, ctxitemEncoderModel),
      userEncoderModel, itemEncoderModel, ctxitemEncoderModel)
  }

  /**
   * Helper function to convert loaded dataframe into one-hot encoded user and item feature vectors.
   * Uses one-hot encoders fitted on train dataframe
   */
  def toFeatures(data: DataFrame,
                 userEncoderModel: OneHotEncoderModel,
                 itemEncoderModel: OneHotEncoderModel,
                 ctxitemEncoderModel: OneHotEncoderModel): DataFrame = {

    var featuresData = userEncoderModel.transform(data)
      .withColumnRenamed("pid", "userid_")  // keep pid as the user id required for FMPair
      .drop(userEncoderModel.getInputCols :_*)
      .withColumnRenamed("userid_", "userid")

    featuresData = itemEncoderModel.transform(featuresData)
      .withColumnRenamed("traid", "itemid")  // keep traid as the item id required for FMPair
      .withColumnRenamed("artid", "artid_")  // keep artid as the sampling column for FMPair
      .drop(itemEncoderModel.getInputCols :_*)
      .withColumnRenamed("artid_", "artid")

    featuresData = ctxitemEncoderModel.transform(featuresData)
      .drop(ctxitemEncoderModel.getInputCols :_*)

    val userCols = Array("pid", "userid", "category", "prev_traid", "prev_albid", "prev_artid", "prev_year")
    val userInputCols = userCols.map(_ + "_encoded")
    val userAssembler = new VectorAssembler().setInputCols(userInputCols).setOutputCol("userctxfeatures")
    featuresData = userAssembler.transform(featuresData)

    val itemCols = Array("traid", "albid", "artid", "year")
    val itemInputCols = itemCols.map(_ + "_encoded")
    val itemAssembler = new VectorAssembler().setInputCols(itemInputCols).setOutputCol("itemfeatures")
    featuresData = itemAssembler.transform(featuresData)

    featuresData.drop(userInputCols ++ itemInputCols :_*)
  }
}

class ServerSideGlintFMPairSpec extends FlatSpec with ScalaFutures with BeforeAndAfterAll with Matchers
  with Inspectors {

  /**
   * Path to small preprocessed subsets of the AOTM-2011 dataset.
   *
   * The train dataset contains all playlists which contain the most popular track, with the last track hidden.
   *
   * The test dataset contains the last tracks of 250 playlists from the train dataset
   * where the last track occurred already in the train dataset.
   */
  private val traindataPath = "AOTM-2011-small-train.csv"
  private val testdataPath = "AOTM-2011-small-test.csv"

  /**
   * Path to save model to. The first test will create it and subsequent tests will rely on it being present
   */
  private val modelPath = "/var/tmp/AOTM-2011-small.model"
  private var modelCreated = false

  /**
   * Path to save model created using accepted artist exp sampling to.
   * The first test will create it and subsequent tests will rely on it being present
   */
  private val expModelPath = "/var/tmp/exp/AOTM-2011-small.model"
  private var expModelCreated = false

  /**
   * Path to save model created using separate Glint cluster to.
   * The first test will create it and subsequent tests will rely on it being present
   */
  private val separateGlintModelPath = "/var/tmp/separate/AOTM-2011-small.model"
  private var separateGlintModelCreated = false
  private val separateGlintConfig = ConfigFactory.parseResourcesAnySyntax("separate-glint.conf")

  /**
   * The Spark session to use
   */
  private lazy val s: SparkSession = SparkSession.builder().appName(getClass.getSimpleName).getOrCreate()

  implicit val ec = ExecutionContext.Implicits.global

  override def beforeAll(): Unit = {
    super.beforeAll()

    val fs = FileSystem.get(new Configuration())
    fs.delete(fs.getHomeDirectory, true)
    fs.copyFromLocalFile(new Path(traindataPath), new Path(traindataPath))
    fs.copyFromLocalFile(new Path(testdataPath), new Path(testdataPath))
  }

  override def afterAll(): Unit = {
    val fs = FileSystem.get(new Configuration())
    fs.delete(new Path(modelPath), true)
    fs.delete(new Path(expModelPath), true)
    fs.delete(new Path(separateGlintModelPath), true)

    s.stop()
  }

  "ServerSideGlintFMPair" should "train and save a model" in {
    val (traindata, _, _, _) = ServerSideGlintFMPairSpec.toFeatures(ServerSideGlintFMPairSpec.load(s, traindataPath))

    val fmpair = new ServerSideGlintFMPair()
      .setBatchSize(1024)
      .setStepSize(0.001f)
      .setLinearReg(0.1f)
      .setFactorsReg(0.01f)
      .setNumParameterServers(2)

    val model = fmpair.fit(traindata)
    try {
      model.save(modelPath)
      FileSystem.get(s.sparkContext.hadoopConfiguration).exists(new Path(modelPath)) shouldBe true
      modelCreated = true
    } finally {
      model.stop()
    }
  }

  it should "train and save a model using accepted artist exp sampling" in {
    val (traindata, _, _, _) = ServerSideGlintFMPairSpec.toFeatures(ServerSideGlintFMPairSpec.load(s, traindataPath))

    val fmpair = new ServerSideGlintFMPair()
      .setBatchSize(1024)
      .setStepSize(0.001f)
      .setLinearReg(0.1f)
      .setFactorsReg(0.01f)
      .setNumParameterServers(2)
      .setSampler("exp")
      .setSamplingCol("artid")

    val model = fmpair.fit(traindata)
    try {
      model.save(expModelPath)
      FileSystem.get(s.sparkContext.hadoopConfiguration).exists(new Path(expModelPath)) shouldBe true
      expModelCreated = true
    } finally {
      model.stop()
    }
  }

  it should "train and save a model using crossbatch sampling and a separate Glint cluster" in {
    val (traindata, _, _, _) = ServerSideGlintFMPairSpec.toFeatures(ServerSideGlintFMPairSpec.load(s, traindataPath))

    val fmpair = new ServerSideGlintFMPair()
      .setBatchSize(1024)
      .setStepSize(0.01f)
      .setLinearReg(0.1f)
      .setFactorsReg(0.01f)
      .setNumParameterServers(2)
      .setParameterServerHost(InetAddress.getLocalHost.getHostAddress)
      .setParameterServerConfig(separateGlintConfig)
      .setSampler("crossbatch")
      .setSamplingCol("itemid")
      .setMaxIter(10)

    val model = fmpair.fit(traindata)
    try {
      model.save(separateGlintModelPath)
      FileSystem.get(s.sparkContext.hadoopConfiguration).exists(new Path(separateGlintModelPath)) shouldBe true
      separateGlintModelCreated = true
    } finally {
      model.stop()
    }
  }

  it should "load a model" in {
    if (!modelCreated) {
      pending
    }

    val model = ServerSideGlintFMPairModel.load(modelPath)
    try {

      model.getBatchSize shouldBe 1024
      model.getLinearReg shouldBe 0.1f
      model.getFactorsReg shouldBe 0.01f
      model.getMaxIter shouldBe 1000
      model.getNumParameterServers shouldBe 2
    } finally {
      model.stop()
    }
  }

  it should "load a model onto a separate Glint cluster" in {
    if (!modelCreated) {
      pending
    }

    val model = ServerSideGlintFMPairModel.load(modelPath, InetAddress.getLocalHost.getHostAddress, separateGlintConfig)
    try {

      model.getBatchSize shouldBe 1024
      model.getLinearReg shouldBe 0.1f
      model.getFactorsReg shouldBe 0.01f
      model.getMaxIter shouldBe 1000
      model.getNumParameterServers shouldBe 2

      // because loaded onto separate Glint cluster
      model.getParameterServerHost shouldEqual InetAddress.getLocalHost.getHostAddress
      model.getParameterServerConfig should not be empty
    } finally {
      model.stop()
    }
  }

  it should "load a model trained on a separate Glint cluster" in {
    if (!separateGlintModelCreated) {
      pending
    }

    val model = ServerSideGlintFMPairModel.load(separateGlintModelPath)
    try {

      model.getBatchSize shouldBe 1024
      model.getLinearReg shouldBe 0.1f
      model.getFactorsReg shouldBe 0.01f
      model.getMaxIter shouldBe 10
      model.getNumParameterServers shouldBe 2
      model.getSampler shouldBe "crossbatch"
      model.getSamplingCol shouldBe "itemid"

      // because loaded onto separate Glint cluster
      model.getParameterServerHost shouldEqual InetAddress.getLocalHost.getHostAddress
      model.getParameterServerConfig should not be empty
    } finally {
      model.stop()
    }
  }

  it should "have a high enough hit rate and ndcg for the top 50 recommendations" in {
    if (!modelCreated) {
      pending
    }

    val (_, userEncoder, itemEncoder, itemctxEncoder) = ServerSideGlintFMPairSpec.toFeatures(
      ServerSideGlintFMPairSpec.load(s, traindataPath))
    val testdata = ServerSideGlintFMPairSpec.toFeatures(
      ServerSideGlintFMPairSpec.load(s, testdataPath), userEncoder, itemEncoder, itemctxEncoder)
    val testquerydata = testdata.drop("itemid")

    val model = ServerSideGlintFMPairModel.load(modelPath)
    try {
      val dcgs = model.recommendForUserSubset(testquerydata, 50)
        .join(testdata, "userid")
        .rdd
        .map(row => (row.getAs[Int]("itemid"), row.getAs[mutable.WrappedArray[Row]]("recommendations")))
        .map { case (item, recs) =>
          recs.zipWithIndex.map { case (rec, i) =>
            if (rec.getAs[Int]("itemid") == item) 1.0 / (math.log10(i + 2) / math.log10(2)) else 0.0
          }.sum
        }.collect()

      val hitRate = dcgs.count(dcg => dcg != 0.0).toDouble / dcgs.length
      val ndcg = dcgs.sum / dcgs.length

      hitRate should be > 0.35
      ndcg should be > 0.27
    } finally {
      model.stop()
    }
  }

  it should "have a high enough hit rate and ndcg for the top 50 recommendations - accepted artist exp" in {
    if (!expModelCreated) {
      pending
    }

    val (_, userEncoder, itemEncoder, itemctxEncoder) = ServerSideGlintFMPairSpec.toFeatures(
      ServerSideGlintFMPairSpec.load(s, traindataPath))
    val testdata = ServerSideGlintFMPairSpec.toFeatures(
      ServerSideGlintFMPairSpec.load(s, testdataPath), userEncoder, itemEncoder, itemctxEncoder)
    val testquerydata = testdata.drop("itemid")

    val model = ServerSideGlintFMPairModel.load(expModelPath)
    try {
      val dcgs = model.recommendForUserSubset(testquerydata, 50)
        .join(testdata, "userid")
        .rdd
        .map(row => (row.getAs[Int]("itemid"), row.getAs[mutable.WrappedArray[Row]]("recommendations")))
        .map { case (item, recs) =>
          recs.zipWithIndex.map { case (rec, i) =>
            if (rec.getAs[Int]("itemid") == item) 1.0 / (math.log10(i + 2) / math.log10(2)) else 0.0
          }.sum
        }.collect()

      val hitRate = dcgs.count(dcg => dcg != 0.0).toDouble / dcgs.length
      val ndcg = dcgs.sum / dcgs.length

      hitRate should be >= 0.32
      ndcg should be > 0.25
    } finally {
      model.stop()
    }
  }

  it should "have a high enough hit rate and ndcg for the top 50 recommendations - crossbatch" in {
    if (!separateGlintModelCreated) {
      pending
    }

    val (_, userEncoder, itemEncoder, itemctxEncoder) = ServerSideGlintFMPairSpec.toFeatures(
      ServerSideGlintFMPairSpec.load(s, traindataPath))
    val testdata = ServerSideGlintFMPairSpec.toFeatures(
      ServerSideGlintFMPairSpec.load(s, testdataPath), userEncoder, itemEncoder, itemctxEncoder)
    val testquerydata = testdata.drop("itemid")

    val model = ServerSideGlintFMPairModel.load(separateGlintModelPath)
    try {
      val dcgs = model.recommendForUserSubset(testquerydata, 50)
        .join(testdata, "userid")
        .rdd
        .map(row => (row.getAs[Int]("itemid"), row.getAs[mutable.WrappedArray[Row]]("recommendations")))
        .map { case (item, recs) =>
          recs.zipWithIndex.map { case (rec, i) =>
            if (rec.getAs[Int]("itemid") == item) 1.0 / (math.log10(i + 2) / math.log10(2)) else 0.0
          }.sum
        }.collect()

      val hitRate = dcgs.count(dcg => dcg != 0.0).toDouble / dcgs.length
      val ndcg = dcgs.sum / dcgs.length

      hitRate should be > 0.3
      ndcg should be > 0.26
    } finally {
      model.stop(terminateOtherClients=true)
    }
  }
}
