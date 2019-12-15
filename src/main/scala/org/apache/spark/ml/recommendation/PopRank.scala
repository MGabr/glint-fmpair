package org.apache.spark.ml.recommendation

import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.param.{Param, ParamMap, Params}
import org.apache.spark.ml.param.shared.HasPredictionCol
import org.apache.spark.ml.util.{DefaultParamsReader, DefaultParamsWritable, DefaultParamsWriter, Identifiable, MLReadable, MLReader, MLWritable, MLWriter, SchemaUtils}
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.functions.{row_number, _}
import org.apache.spark.sql.types.{FloatType, IntegerType, StructType}

private[recommendation] trait PopRankParams extends Params with HasPredictionCol {

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
}

class PopRank(override val uid: String) extends Estimator[PopRankModel] with PopRankParams with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("poprank"))

  /** @group setParam */
  def setUserCol(value: String): this.type = set(userCol, value)

  /** @group setParam */
  def setItemCol(value: String): this.type = set(itemCol, value)

  /** @group setParam */
  def setPredictionCol(value: String): this.type = set(predictionCol, value)

  override def fit(dataset: Dataset[_]): PopRankModel = {
    val itemCounts = dataset.select(getItemCol, getUserCol)
      .distinct()
      .groupBy(getItemCol)
      .agg(count(getUserCol).as("score"))
      .orderBy(desc("score"), asc("itemid"))

    copyValues(new PopRankModel(this.uid, itemCounts).setParent(this))
  }

  override def copy(extra: ParamMap): Estimator[PopRankModel] = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, getUserCol, IntegerType)
    SchemaUtils.checkColumnType(schema, getItemCol, IntegerType)
    schema
  }
}

class PopRankModel private[ml](override val uid: String, val itemCounts: DataFrame)
  extends Model[PopRankModel] with PopRankParams with MLWritable {

  override def transform(dataset: Dataset[_]): DataFrame = {
    throw new NotImplementedError() // TODO
  }

  override def transformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, getUserCol, IntegerType)
    SchemaUtils.appendColumn(schema, getPredictionCol, FloatType)
  }

  override def copy(extra: ParamMap): PopRankModel = {
    val copied = new PopRankModel(uid, itemCounts)
    copyValues(copied, extra).setParent(parent)
  }

  override def write: MLWriter = new PopRankModel.PopRankModelWriter(this)

  /**
   * Returns top numItems items recommended for each user id in the input data set
   *
   * @param dataset  The dataset containing a column of user ids. The column name must match userCol
   * @param numItems The maximum number of recommendations for each user
   * @param explode Whether the resulting dataframe should be exploded or contain only a single row for each user
   * @return A dataframe of (userCol: Int, recommendations), where recommendations are stored
   *         as an array of (score: Float, itemCol: Int) rows. Or if exploded
   *         a dataframe of (userCol: Int, score: Float, itemCol: Int) rows.
   */
  def recommendForUserSubset(dataset: Dataset[_], numItems: Int, explode: Boolean = false): DataFrame = {

    // get top numItems, limit(numItems) not working correctly
    val window = Window.orderBy(desc("score"), asc("itemid"))
    val topItemCountsDf = itemCounts
      .select(col(getItemCol), col("score"), row_number().over(window).as("rowno"))
      .filter(col("rowno").leq(numItems))

    val recommendDf = dataset.select(getUserCol).distinct()
      .crossJoin(topItemCountsDf)
      .select(getUserCol, "score", getItemCol)

    if (explode) {
      recommendDf
    } else {
      // convert to rows with recommendation arrays
      recommendDf
        .withColumn("recommendation", struct("score", getItemCol))  // score first for sort
        .groupBy(getUserCol)
        .agg(sort_array(collect_list("recommendation"), asc=false).as("recommendations"))
        .select(getUserCol, "recommendations")
    }
  }
}

object PopRankModel extends MLReadable[PopRankModel] {

  private[PopRankModel] class PopRankModelWriter(instance: PopRankModel) extends MLWriter {

    override protected def saveImpl(path: String): Unit = {
      DefaultParamsWriter.saveMetadata(instance, path, sc)
      instance.itemCounts.write.save(path + "/itemcounts")
    }
  }

  private class PopRankModelReader extends MLReader[PopRankModel] {

    private val className = classOf[PopRankModel].getName

    override def load(path: String): PopRankModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc, className)
      val itemCounts = sparkSession.read.load(path + "/itemcounts")
      val model = new PopRankModel(metadata.uid, itemCounts)
      metadata.getAndSetParams(model)
      model
    }
  }

  override def read: MLReader[PopRankModel] = new PopRankModelReader

  /**
   * Loads a [[org.apache.spark.ml.recommendation.PopRankModel PopRankModel]]
   *
   * @param path The path
   * @return The [[org.apache.spark.ml.recommendation.PopRankModel PopRankModel]]
   */
  override def load(path: String): PopRankModel = super.load(path)
}
