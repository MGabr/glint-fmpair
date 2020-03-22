package org.apache.spark.ml.recommendation

import org.apache.spark.ml.{Estimator, Model}
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


private[recommendation] trait KNNRecommender extends KNNParams {

  /** @group setParam */
  def setK(value: Int): this.type = set(k, value)

  /** @group setParam */
  def setFilterUserItems(value: Boolean): this.type = set(filterUserItems, value)

  def recommendFromNearestUsers(queryDf: DataFrame, nnDf: DataFrame, numItems: Int): DataFrame = {

    val recommendDf = if (getFilterUserItems) {
      // filter items of user him/herself
      val filterDf = queryDf
        .select(col(getUserCol), col(getItemCol))
        .groupBy(getUserCol)
        .agg(collect_set(getItemCol).as("filterItems"))
      nnDf
        .join(filterDf, getUserCol)
        .filter(not(array_contains(col("filterItems"), col(getItemCol))))
    } else {
      nnDf
    }

    val itemWindow = Window.partitionBy(getUserCol).orderBy(desc("score"), asc(getItemCol))

    // get top items
    recommendDf
      .groupBy(getUserCol, getItemCol)
      .agg(sum("similarity").as("score"))
      .select(col(getUserCol), col("score"), col(getItemCol), row_number().over(itemWindow).as("rowno"))
      .filter(col("rowno").leq(numItems))
      .select(getUserCol, getItemCol, "score")
      // convert to rows with recommendation arrays
      .withColumn("recommendation", struct("score", getItemCol))  // score first for sort
      .groupBy(getUserCol)
      .agg(sort_array(collect_list("recommendation"), asc=false).as("recommendations"))
      .select(getUserCol, "recommendations")
  }
}


class KNN(override val uid: String) extends KNNRecommender {

  def this() = this(Identifiable.randomUID("knn"))

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

    val userWindow = Window.partitionBy(getUserCol)
    val otherUserWindow = Window.partitionBy(getUserCol).orderBy(desc("similarity"), asc("otherUserid"))

    val userDf = fitDataset
      .select(
        col(getUserCol).alias("otherUserid"),
        col(getItemCol),
        count(getItemCol).over(userWindow).alias("otherUsernorm"))

    val nnDf = queryDataset
      .select(col(getUserCol), col(getItemCol), count(getItemCol).over(userWindow).alias("usernorm"))
      // get top k nearest users
      .join(userDf, getItemCol)
      .groupBy(getUserCol, "otherUserid")
      .agg(sum(lit(1) / (sqrt("usernorm") * sqrt("otherUsernorm"))).as("similarity"))
      .select(col(getUserCol), col("otherUserid"), col("similarity"), row_number().over(otherUserWindow).as("rowno"))
      .filter(col("rowno").leq(getK))
      .select(getUserCol, "otherUserid", "similarity")
      // get items of k nearest users
      .join(fitDataset.select(col(getUserCol).alias("otherUserid"), col(getItemCol)), "otherUserid")

    recommendFromNearestUsers(queryDataset.toDF(), nnDf, numItems)
  }
}


private[recommendation] trait _TfIdfKNN extends KNNParams {

  def toTfIdf(idfDf: DataFrame): DataFrame = {
    val userWindow = Window.partitionBy(getUserCol)

    idfDf
      .groupBy(getUserCol, getItemCol, "idf")
      .agg(count(getItemCol).as("tf"))
      .select(col(getUserCol), col(getItemCol), ((log2(col("tf")) + 1) * col("idf")).as("tfidf"))
      .withColumn("tfidf2", col("tfidf") * col("tfidf"))
      .withColumn("tfidfl2", sum("tfidf2").over(userWindow))
      .select(col(getUserCol), col(getItemCol), (col("tfidf") / sqrt("tfidfl2")).as("tfidf"))
  }
}


class TfIdfKNN(val uid: String) extends Estimator[TfIdfKNNModel] with _TfIdfKNN with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("tfidf-knn"))

  /** @group setParam */
  def setK(value: Int): this.type = set(k, value)

  /** @group setParam */
  def setFilterUserItems(value: Boolean): this.type = set(filterUserItems, value)

  override def fit(dataset: Dataset[_]): TfIdfKNNModel = {
    // tf-idf = log2(1 + tf) * (log2((1 + n) / (1 + df)) + 1)
    val n = dataset.dropDuplicates(getUserCol).count()
    val idfDf = dataset
      .groupBy(getItemCol)
      .agg(countDistinct(getUserCol).as("df"))
      .select(col(getItemCol), (log2(lit(1 + n) / (col("df") + 1)) + 1).as("idf"))
    val userDf = toTfIdf(dataset.join(idfDf, getItemCol))
      .select(col(getUserCol).as("otherUserid"), col(getItemCol), col("tfidf").as("otherUserTfidf"))

    new TfIdfKNNModel(this.uid, idfDf, userDf).setParent(this)
  }

  override def copy(extra: ParamMap): Estimator[TfIdfKNNModel] = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, getUserCol, IntegerType)
    SchemaUtils.checkColumnType(schema, getItemCol, IntegerType)
    schema
  }
}


class TfIdfKNNModel(override val uid: String, val idfDf: DataFrame, val userDf: DataFrame)
  extends Model[TfIdfKNNModel] with _TfIdfKNN with KNNRecommender with MLWritable {

  override def transform(dataset: Dataset[_]): DataFrame = {
    throw new NotImplementedError() // TODO
  }

  override def transformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, getUserCol, IntegerType)
    SchemaUtils.appendColumn(schema, getPredictionCol, FloatType)
  }

  override def copy(extra: ParamMap): TfIdfKNNModel = {
    val copied = new TfIdfKNNModel(uid, idfDf, userDf)
    copyValues(copied, extra).setParent(parent)
  }

  override def write: MLWriter = new TfIdfKNNModel.TfIdfKNNModelWriter(this)

  /**
   * Returns top numItems items recommended for each user id in the input data set
   *
   * @param queryDataset The dataset containing a column of user ids. The column name must match userCol
   * @param numItems The maximum number of recommendations for each user
   * @return A dataframe of (userCol: Int, recommendations), where recommendations are stored
   *         as an array of (score: Float, itemCol: Int) rows. Or if exploded
   *         a dataframe of (userCol: Int, score: Float, itemCol: Int) rows.
   */
  def recommendForUserSubset(queryDataset: Dataset[_], numItems: Int): DataFrame = {

    val otherUserWindow = Window.partitionBy(getUserCol).orderBy(desc("similarity"), asc("otherUserid"))

    val nnDf = toTfIdf(queryDataset.join(idfDf, getItemCol))
      // get top k nearest users
      .join(userDf, getItemCol)
      .select(col(getUserCol), col("otherUserid"), (col("tfidf") * col("otherUserTfidf")).as("similarity"))
      .groupBy(getUserCol, "otherUserid")
      .agg(sum("similarity").as("similarity"))
      .select(col(getUserCol), col("otherUserid"), col("similarity"), row_number().over(otherUserWindow).as("rowno"))
      .filter(col("rowno").leq(getK))
      .select(getUserCol, "otherUserid", "similarity")
      // get items of k nearest users
      .join(userDf.select("otherUserid", getItemCol), "otherUserid")

    recommendFromNearestUsers(queryDataset.toDF(), nnDf, numItems)
  }
}

object TfIdfKNNModel extends MLReadable[TfIdfKNNModel] {

  private[TfIdfKNNModel] class TfIdfKNNModelWriter(instance: TfIdfKNNModel) extends MLWriter {

    override protected def saveImpl(path: String): Unit = {
      DefaultParamsWriter.saveMetadata(instance, path, sc)
      instance.idfDf.write.save(path + "/idf")
      instance.userDf.write.save(path + "/user")
    }
  }

  private class TfIdfKNNModelReader extends MLReader[TfIdfKNNModel] {

    private val className = classOf[TfIdfKNNModel].getName

    override def load(path: String): TfIdfKNNModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc, className)
      val idfDf = sparkSession.read.load(path + "/idf")
      val userDf = sparkSession.read.load(path + "/user")
      val model = new TfIdfKNNModel(metadata.uid, idfDf, userDf)
      metadata.getAndSetParams(model)
      model
    }
  }

  override def read: MLReader[TfIdfKNNModel] = new TfIdfKNNModelReader

  /**
   * Loads a [[org.apache.spark.ml.recommendation.TfIdfKNNModel TfIdfKNNModel]]
   *
   * @param path The path
   * @return The [[org.apache.spark.ml.recommendation.TfIdfKNNModel TfIdfKNNModel]]
   */
  override def load(path: String): TfIdfKNNModel = super.load(path)
}