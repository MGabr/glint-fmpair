package org.apache.spark.ml.recommendation

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.scalatest.{BeforeAndAfterAll, FlatSpec, Inspectors, Matchers}
import org.scalatest.concurrent.ScalaFutures

import scala.collection.mutable

object SAGHSpec {

  /**
   * Helper function to load preprocessed csv data from path as dataframe
   */
  def load(s: SparkSession, dataPath: String): DataFrame = {
    s.read.format("csv").option("header", "true").option("inferSchema", "true").load(dataPath)
      .select(col("pid").as("userid"), col("traid").as("itemid"), col("artid"))
  }
}

class SAGHSpec extends FlatSpec with ScalaFutures with BeforeAndAfterAll with Matchers with Inspectors{

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

  "SAGH" should "train and save a model" in {
    val traindata = SAGHSpec.load(s, traindataPath)

    val model = new SAGH().fit(traindata)

    model.save(modelPath)
    FileSystem.get(s.sparkContext.hadoopConfiguration).exists(new Path(modelPath)) shouldBe true
    modelCreated = true
  }

  it should "load a model" in {
    if (!modelCreated) {
      pending
    }

    val model = SAGHModel.load(modelPath)
    model.getUserCol shouldBe "userid"
    model.getItemCol shouldBe "itemid"
  }

  it should "have a high enough hit rate for the top 50 recommendations" in {
    if (!modelCreated) {
      pending
    }

    val testdata = SAGHSpec.load(s, testdataPath)
    val testquerydata = SAGHSpec.load(s, testquerydataPath)

    val model = SAGHModel.load(modelPath)

    val hits = model.recommendForUserSubset(testquerydata, 50)
      .join(testdata, "userid")
      .rdd
      .map(row => (row.getAs[Int]("itemid"), row.getAs[mutable.WrappedArray[Row]]("recommendations")))
      .map { case (item, recs) => recs.map(_.getAs[Int]("itemid")).contains(item) }
      .collect()

    val hitRate = hits.count(t => t).toDouble / hits.length
    hitRate should be >= 0.1
  }
}
