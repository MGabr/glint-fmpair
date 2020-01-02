package org.apache.spark.ml.recommendation

import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.param.{BooleanParam, Param, ParamMap, Params}
import org.apache.spark.ml.param.shared.HasPredictionCol
import org.apache.spark.ml.util.{DefaultParamsReader, DefaultParamsWritable, DefaultParamsWriter, Identifiable, MLReadable, MLReader, MLWritable, MLWriter, SchemaUtils}
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{FloatType, IntegerType, StructType}

private[recommendation] trait SAGHParams extends Params with HasPredictionCol {

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
   * The name of the artist id column of integers from 0 to number of artists in training dataset
   * Default: "artid"
   *
   * @group param
   */
  final val artistCol = new Param[String](this, "artistCol", "the name of the artist id column")
  setDefault(artistCol -> "artid")

  /** @group getParam */
  def getArtistCol: String = $(artistCol)


  /**
   * Whether the items of a user should be filtered from the recommendations for the user
   * Default: false
   */
  final val filterUserItems = new BooleanParam(this, "filterUserItems", "whether the items of a user should be filtered from the recommendations for the user")
  setDefault(filterUserItems -> false)

  /** @group getParam */
  def getFilterUserItems: Boolean = $(filterUserItems)
}

class SAGH(override val uid: String) extends Estimator[SAGHModel] with SAGHParams with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("sagh"))

  /** @group setParam */
  def setUserCol(value: String): this.type = set(userCol, value)

  /** @group setParam */
  def setItemCol(value: String): this.type = set(itemCol, value)

  /** @group setParam */
  def setArtistCol(value: String): this.type = set(artistCol, value)

  /** @group setParam */
  def setPredictionCol(value: String): this.type = set(predictionCol, value)

  override def fit(dataset: Dataset[_]): SAGHModel = {
    val itemCounts = dataset.select(getItemCol, getArtistCol, getUserCol)
      .distinct()
      .groupBy(getItemCol, getArtistCol)
      .agg(count(getUserCol).as("score"))
      .orderBy(desc("score"), asc(getItemCol))

    copyValues(new SAGHModel(this.uid, itemCounts).setParent(this))
  }

  override def copy(extra: ParamMap): Estimator[SAGHModel] = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, getUserCol, IntegerType)
    SchemaUtils.checkColumnType(schema, getItemCol, IntegerType)
    schema
  }
}

class SAGHModel private[ml](override val uid: String, val itemCounts: DataFrame)
  extends Model[SAGHModel] with SAGHParams with MLWritable {

  /** @group setParam */
  def setFilterUserItems(value: Boolean): this.type = set(filterUserItems, value)

  override def transform(dataset: Dataset[_]): DataFrame = {
    throw new NotImplementedError() // TODO
  }

  override def transformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, getUserCol, IntegerType)
    SchemaUtils.appendColumn(schema, getPredictionCol, FloatType)
  }

  override def copy(extra: ParamMap): SAGHModel = {
    val copied = new SAGHModel(uid, itemCounts)
    copyValues(copied, extra).setParent(parent)
  }

  override def write: MLWriter = new SAGHModel.SAGHModelWriter(this)

  /**
   * Returns top numItems items recommended for each user id in the input data set
   *
   * @param dataset The dataset containing a column of user ids and a column of artist ids.
   *                The column names must match userCol and artistCol
   * @param numItems The maximum number of recommendations for each user
   * @return A dataframe of (userCol: Int, recommendations), where recommendations are stored
   *         as an array of (score: Float, itemCol: Int) rows.
   */
  def recommendForUserSubset(dataset: Dataset[_], numItems: Int): DataFrame = {

    val recommendDf = if (getFilterUserItems) {
      dataset
        .select(col(getUserCol), col(getArtistCol), col(getItemCol).as("useritemid"))
        .join(itemCounts, getArtistCol)
        .filter(col("useritemid").notEqual(col(getItemCol)))
        .dropDuplicates(getUserCol, getItemCol)
    } else {
      dataset
        .select(getUserCol, getArtistCol)
        .distinct()
        .join(itemCounts, getArtistCol)
    }

    val window = Window.partitionBy(getUserCol).orderBy(desc("score"), asc("itemid"))

    recommendDf
      // get top numItems
      .select(col(getUserCol), col(getItemCol), col("score"), row_number().over(window).as("rowno"))
      .filter(col("rowno").leq(numItems))
      .select(getUserCol, "score", getItemCol)
      // convert to rows with recommendation arrays
      .withColumn("recommendation", struct("score", getItemCol))  // score first for sort
      .groupBy(getUserCol)
      .agg(sort_array(collect_list("recommendation"), asc=false).as("recommendations"))
      .select(getUserCol, "recommendations")
  }
}

object SAGHModel extends MLReadable[SAGHModel] {

  private[SAGHModel] class SAGHModelWriter(instance: SAGHModel) extends MLWriter {

    override protected def saveImpl(path: String): Unit = {
      DefaultParamsWriter.saveMetadata(instance, path, sc)
      instance.itemCounts.write.save(path + "/itemcounts")
    }
  }

  private class SAGHModelReader extends MLReader[SAGHModel] {

    private val className = classOf[SAGHModel].getName

    override def load(path: String): SAGHModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc, className)
      val itemCounts = sparkSession.read.load(path + "/itemcounts")
      val model = new SAGHModel(metadata.uid, itemCounts)
      metadata.getAndSetParams(model)
      model
    }
  }

  override def read: MLReader[SAGHModel] = new SAGHModelReader

  /**
   * Loads a [[org.apache.spark.ml.recommendation.SAGHModel SAGHModel]]
   *
   * @param path The path
   * @return The [[org.apache.spark.ml.recommendation.SAGHModel SAGHModel]]
   */
  override def load(path: String): SAGHModel = super.load(path)
}