package org.apache.spark.ml.recommendation

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.scalatest.{BeforeAndAfterAll, FlatSpec, Inspectors, Matchers}

import scala.collection.mutable

object KNNSpec {

  /**
   * Helper function to load preprocessed csv data from path as dataframe
   */
  def load(s: SparkSession, dataPath: String): DataFrame = {
    s.read.format("csv").option("header", "true").option("inferSchema", "true").load(dataPath)
      .select(col("pid").as("userid"), col("traid").as("itemid"))
  }

  /**
   * Computes the mean hit rate and mean normalized discounted cumulative gain for the recommendations data frame
   */
  def toHitRateAndNDCG(data: DataFrame): (Double, Double) = {
    val dcgs = data.rdd
      .map(row => (row.getAs[Int]("itemid"), row.getAs[mutable.WrappedArray[Row]]("recommendations")))
      .map { case (item, recs) =>
        recs.zipWithIndex.map { case (rec, i) =>
          if (rec.getAs[Int]("itemid") == item) 1.0 / (math.log10(i + 2) / math.log10(2)) else 0.0
        }.sum
      }.collect()
    val hitRate = dcgs.count(dcg => dcg != 0.0).toDouble / dcgs.length
    val ndcg = dcgs.sum / dcgs.length
    (hitRate, ndcg)
  }
}

class KNNSpec extends FlatSpec with BeforeAndAfterAll with Matchers with Inspectors {

  /**
   * Path to small preprocessed subsets of the AOTM-2011 dataset.
   *
   * The train dataset contains all playlists which contain the most popular track, with the last track hidden.
   *
   * The test dataset contains the last tracks of 250 playlists from the train dataset
   * where the last track occurred already in the train dataset.
   */
  private val traindataPath = "AOTM-2011-small-train.csv"
  private val testquerydataPath = "AOTM-2011-small-test-queryseeds.csv"
  private val testdataPath = "AOTM-2011-small-test.csv"

  /**
   * Path to save model to. The first test will create it and subsequent tests will rely on it being present
   */
  private val modelPath = "/var/tmp/AOTM-2011-small.model"
  private var modelCreated = false

  /**
   * The Spark session to use
   */
  private lazy val s: SparkSession = SparkSession.builder().appName(getClass.getSimpleName).getOrCreate()

  override def beforeAll(): Unit = {
    super.beforeAll()

    val fs = FileSystem.get(new Configuration())
    fs.delete(fs.getHomeDirectory, true)
    fs.copyFromLocalFile(new Path(traindataPath), new Path(traindataPath))
    fs.copyFromLocalFile(new Path(testdataPath), new Path(testdataPath))
    fs.copyFromLocalFile(new Path(testquerydataPath), new Path(testquerydataPath))
  }

  override def afterAll(): Unit = {
    val fs = FileSystem.get(new Configuration())
    fs.delete(new Path(modelPath), true)

    s.stop()
  }

  "KNN" should "have a high enough hit rate and ndcg for the top 50 recommendations" in {

    val traindata = KNNSpec.load(s, traindataPath)
    val testdata = KNNSpec.load(s, testdataPath)
    val testquerydata = KNNSpec.load(s, testquerydataPath)

    val knn = new KNN().setK(150)

    val (hitRate, ndcg) = KNNSpec.toHitRateAndNDCG(
      knn.recommendForUserSubset(testquerydata, traindata, 50).join(testdata, "userid"))

    hitRate should be >= 0.32
    ndcg should be > 0.19
  }

  it should "have a high enough hit rate and ndcg for the top 50 recommendations - filter user items" in {

    val traindata = KNNSpec.load(s, traindataPath)
    val testdata = KNNSpec.load(s, testdataPath)
    val testquerydata = KNNSpec.load(s, testquerydataPath)

    val knn = new KNN().setK(150).setFilterUserItems(true)

    val (hitRate, ndcg) = KNNSpec.toHitRateAndNDCG(
      knn.recommendForUserSubset(testquerydata, traindata, 50).join(testdata, "userid"))

    hitRate should be >= 0.32
    ndcg should be > 0.26
  }

  "TfIdfKNN" should "train and save a model" in {
    val traindata = KNNSpec.load(s, traindataPath)

    val model = new TfIdfKNN().setK(150).setFilterUserItems(true).fit(traindata)

    model.save(modelPath)
    FileSystem.get(s.sparkContext.hadoopConfiguration).exists(new Path(modelPath)) shouldBe true
    modelCreated = true
  }

  it should "load a model" in {
    if (!modelCreated) {
      pending
    }

    val model = TfIdfKNNModel.load(modelPath)
    model.getUserCol shouldBe "userid"
    model.getItemCol shouldBe "itemid"
  }

  it should "have a high enough hit rate and ndcg for the top 50 recommendations" in {
    if (!modelCreated) {
      pending
    }

    val testdata = KNNSpec.load(s, testdataPath)
    val testquerydata = KNNSpec.load(s, testquerydataPath)

    val model = TfIdfKNNModel.load(modelPath)

    val (hitRate, ndcg) = KNNSpec.toHitRateAndNDCG(
      model.recommendForUserSubset(testquerydata, 50).join(testdata, "userid"))

    hitRate should be >= 0.35
    ndcg should be > 0.17
  }
}
