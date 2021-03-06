package org.apache.spark.ml.recommendation

import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.param.{BooleanParam, Param, ParamMap, Params}
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

  /**
   * Whether the overall frequency of items should be used instead of the frequency of items per user
   * Default: false
   *
   * @group param
   */
  final val allItemsFrequency = new BooleanParam(this, "allItemsFrequency",
    "whether the overall frequency of items should be used instead of the frequency of items per user")
  setDefault(allItemsFrequency -> false)

  /** @group getParam */
  def getAllItemsFrequency: Boolean = $(allItemsFrequency)

  /**
   * Whether the items of a user should be filtered from the recommendations for the user
   * Default: false
   */
  final val filterUserItems = new BooleanParam(this, "filterUserItems", "whether the items of a user should be filtered from the recommendations for the user")
  setDefault(filterUserItems -> false)

  /** @group getParam */
  def getFilterUserItems: Boolean = $(filterUserItems)
}

/**
 * Popularity ranking recommender.
 */
class PopRank(override val uid: String) extends Estimator[PopRankModel] with PopRankParams with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("poprank"))

  /** @group setParam */
  def setUserCol(value: String): this.type = set(userCol, value)

  /** @group setParam */
  def setItemCol(value: String): this.type = set(itemCol, value)

  /** @group setParam */
  def setPredictionCol(value: String): this.type = set(predictionCol, value)

  /** @group setParam */
  def setAllItemsFrequency(value: Boolean): this.type = set(allItemsFrequency, value)

  override def fit(dataset: Dataset[_]): PopRankModel = {
    var itemCounts = dataset.select(getItemCol, getUserCol)
    if (!getAllItemsFrequency) {
      itemCounts = itemCounts.distinct()
    }
    itemCounts = itemCounts
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

/**
 * Model fitted by [[org.apache.spark.ml.recommendation.PopRank PopRank]].
 */
class PopRankModel private[ml](override val uid: String, val itemCounts: DataFrame)
  extends Model[PopRankModel] with PopRankParams with MLWritable {


  /** @group setParam */
  def setFilterUserItems(value: Boolean): this.type = set(filterUserItems, value)


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
   * @return A dataframe of (userCol: Int, recommendations), where recommendations are stored
   *         as an array of (score: Float, itemCol: Int) rows. Or if exploded
   *         a dataframe of (userCol: Int, score: Float, itemCol: Int) rows.
   */
  def recommendForUserSubset(dataset: Dataset[_], numItems: Int): DataFrame = {

    val maxNumItems = if (getFilterUserItems) {
      numItems + dataset
        .groupBy(getUserCol)
        .agg(count(getItemCol).as("count"))
        .agg(max("count").as("max"))
        .first()
        .getAs[Long]("max").toInt
    } else {
      numItems
    }

    // get top maxNumItems, limit(maxNumItems) not working correctly
    val window = Window.orderBy(desc("score"), asc("itemid"))
    val topItemCountsDf = itemCounts
      .select(col(getItemCol), col("score"), row_number().over(window).as("rowno"))
      .filter(col("rowno").leq(maxNumItems))

    val recommendDf = if (getFilterUserItems) {
      val userWindow = Window.partitionBy(getUserCol).orderBy(desc("score"), asc("itemid"))
      dataset
        .select(col(getUserCol), col(getItemCol))
        .groupBy(getUserCol)
        .agg(collect_set(getItemCol).alias("filterItems"))
        .crossJoin(topItemCountsDf)
        .filter(not(array_contains(col("filterItems"), col(getItemCol))))
        // get top numItems
        .select(col(getUserCol), col(getItemCol), col("score"), row_number().over(userWindow).as("userrowno"))
        .filter(col("userrowno").leq(numItems))
    } else {
      dataset
        .select(getUserCol)
        .distinct()
        .crossJoin(topItemCountsDf)
        .select(getUserCol, "score", getItemCol)
    }

    // convert to rows with recommendation arrays
    recommendDf
      .withColumn("recommendation", struct("score", getItemCol))  // score first for sort
      .groupBy(getUserCol)
      .agg(sort_array(collect_list("recommendation"), asc=false).as("recommendations"))
      .select(getUserCol, "recommendations")
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
