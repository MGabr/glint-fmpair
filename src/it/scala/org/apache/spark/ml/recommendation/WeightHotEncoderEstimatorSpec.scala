package org.apache.spark.ml.recommendation

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.ml.feature.{WeightHotEncoderEstimator, WeightHotEncoderModel}
import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.scalatest.{BeforeAndAfterAll, FlatSpec, Inspectors, Matchers}

object WeightHotEncoderEstimatorSpec {

  /**
   * Helper function to load csv data from path as dataframe
   */
  def load(s: SparkSession, dataPath: String): DataFrame = {
    s.read.format("csv").option("header", "true").option("inferSchema", "true").load(dataPath)
  }
}

class WeightHotEncoderEstimatorSpec extends FlatSpec with BeforeAndAfterAll with Matchers with Inspectors {

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
   * The Spark session to use
   */
  private lazy val s: SparkSession = SparkSession.builder().appName(getClass.getSimpleName).getOrCreate()

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

    s.stop()
  }

  "WeightHotEncoderEstimator" should "train and save a model" in {
    val traindata = WeightHotEncoderEstimatorSpec.load(s, traindataPath)

    val model = new WeightHotEncoderEstimator()
      .setWeights(Array(1.0, 0.3, 1.0, 0.5, 0.1))
      .setInputCols(Array("pid", "userid", "traid", "albid", "artid"))
      .setOutputCols(Array("pid_encoded", "userid_encoded", "traid_encoded", "albid_encoded", "artid_encoded"))
      .fit(traindata)

    model.save(modelPath)
    FileSystem.get(s.sparkContext.hadoopConfiguration).exists(new Path(modelPath)) shouldBe true
    modelCreated = true
  }

  it should "load a model" in {
    if (!modelCreated) {
      pending
    }

    val model = WeightHotEncoderModel.load(modelPath)
    model.getWeights shouldEqual Array(1.0, 0.3, 1.0, 0.5, 0.1)
  }

  it should "one-hot encode with weight instead of one" in {
    if (!modelCreated) {
      pending
    }

    val testdata = WeightHotEncoderEstimatorSpec.load(s, testdataPath)

    val model = WeightHotEncoderModel.load(modelPath)

    val transformedRow = model.transform(testdata).head()

    val weights = Array("pid_encoded", "userid_encoded", "traid_encoded", "albid_encoded", "artid_encoded")
      .foldLeft(Array[Double]())((values, outputCol) => values ++ transformedRow.getAs[SparseVector](outputCol).values)
    weights shouldEqual Array(1.0, 0.3, 1.0, 0.5, 0.1)
  }
}