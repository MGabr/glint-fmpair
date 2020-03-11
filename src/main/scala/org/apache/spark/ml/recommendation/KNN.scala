package org.apache.spark.ml.recommendation

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared.HasPredictionCol
import org.apache.spark.ml.util._
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{FloatType, IntegerType, StructType}
import org.apache.spark.sql.{DataFrame, Dataset}

private[recommendation] trait KNNParams extends Params with HasPredictionCol {

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
   * The number of nearest neighbours
   * Default: 150
   */
  final val k = new IntParam(this, "k", "the number of nearest neighbours")
  setDefault(k -> 150)

  /** @group getParam */
  def getK: Int = $(k)

  /**
   * Whether the items of a user should be filtered from the recommendations for the user
   * Default: false
   */
  final val filterUserItems = new BooleanParam(this, "filterUserItems",
    "whether the items of a user should be filtered from the recommendations for the user")
  setDefault(filterUserItems -> false)

  /** @group getParam */
  def getFilterUserItems: Boolean = $(filterUserItems)
}


class KNN(override val uid: String) extends Transformer with KNNParams {

  def this() = this(Identifiable.randomUID("knn"))

  /** @group setParam */
  def setK(value: Int): this.type = set(k, value)

  /** @group setParam */
  def setFilterUserItems(value: Boolean): this.type = set(filterUserItems, value)


  override def transform(dataset: Dataset[_]): DataFrame = {
    throw new NotImplementedError() // TODO
  }

  override def transformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, getUserCol, IntegerType)
    SchemaUtils.appendColumn(schema, getPredictionCol, FloatType)
  }

  override def copy(extra: ParamMap): KNN = {
    val copied = new KNN(uid)
    copyValues(copied, extra)
  }

  /**
   * Returns top numItems items recommended for each user id in the input data set
   *
   * @param queryDataset The dataset containing a column of user ids. The column name must match userCol
   * @param fitDataset
   * @param numItems The maximum number of recommendations for each user
   * @return A dataframe of (userCol: Int, recommendations), where recommendations are stored
   *         as an array of (score: Float, itemCol: Int) rows. Or if exploded
   *         a dataframe of (userCol: Int, score: Float, itemCol: Int) rows.
   */
  def recommendForUserSubset(queryDataset: Dataset[_], fitDataset: Dataset[_], numItems: Int): DataFrame = {

    val normWindow = Window.partitionBy(getUserCol)
    val otherUserWindow = Window.partitionBy(getUserCol).orderBy(desc("similarity"), asc("otherUserid"))
    val itemWindow = Window.partitionBy(getUserCol).orderBy(desc("score"), asc(getItemCol))

    val userDf = fitDataset
      .select(
        col(getUserCol).alias("otherUserid"),
        col(getItemCol),
        count(getItemCol).over(normWindow).alias("otherUsernorm"))

    var recommendDf = queryDataset
      .select(col(getUserCol), col(getItemCol), count(getItemCol).over(normWindow).alias("usernorm"))
      // get top k nearest users
      .join(userDf, getItemCol)
      .groupBy(getUserCol, "otherUserid")
      .agg(sum(lit(1).divide(sqrt("usernorm")).multiply(sqrt("otherUsernorm"))).alias("similarity"))
      .select(col(getUserCol), col("otherUserid"), col("similarity"), row_number().over(otherUserWindow).as("rowno"))
      .filter(col("rowno").leq(getK))
      .select(getUserCol, "otherUserid", "similarity")
      // get items of k nearest users
      .join(fitDataset.select(col(getUserCol).alias("otherUserid"), col(getItemCol)), "otherUserid")

    if (getFilterUserItems) {
      // filter items of user him/herself
      val filterDf = queryDataset
        .select(col(getUserCol), col(getItemCol))
        .groupBy(getUserCol)
        .agg(collect_set(getItemCol).alias("filterItems"))
      recommendDf = recommendDf
        .join(filterDf, getUserCol)
        .filter(not(array_contains(col("filterItems"), col(getItemCol))))
    }

    // get top items
    recommendDf = recommendDf.groupBy(getUserCol, getItemCol)
      .agg(sum("similarity").alias("score"))
      .select(col(getUserCol), col("score"), col(getItemCol), row_number().over(itemWindow).as("rowno"))
      .filter(col("rowno").leq(numItems))
      .select(getUserCol, getItemCol, "score")

    // convert to rows with recommendation arrays
    recommendDf
      .withColumn("recommendation", struct("score", getItemCol))  // score first for sort
      .groupBy(getUserCol)
      .agg(sort_array(collect_list("recommendation"), asc=false).as("recommendations"))
      .select(getUserCol, "recommendations")
  }
}
