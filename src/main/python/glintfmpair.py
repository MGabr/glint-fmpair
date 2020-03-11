import sys
import types
if sys.version > '3':
    basestring = str

from pyspark import keyword_only
from pyspark.ml import feature
from pyspark.ml import recommendation
from pyspark.ml.param.shared import *
from pyspark.ml.util import JavaMLReadable, JavaMLWritable
from pyspark.ml.wrapper import JavaEstimator, JavaModel, JavaTransformer
from pyspark.ml.common import inherit_doc

__all__ = ['GlintFMPair', 'GlintFMPairModel', 'PopRank', 'PopRankModel', 'SAGH', 'SAGHModel', 'KNN',
           'WeightHotEncoderEstimator', 'WeightHotEncoderModel']


class _GlintFMPairParams(HasStepSize, HasMaxIter, HasSeed, HasPredictionCol):

    __module__ = "pyspark.ml.recommendation"

    userCol = Param(Params._dummy(),
                    "userCol", "the name of the user id column")
    itemCol = Param(Params._dummy(),
                    "itemCol", "the name of the item id column")
    userctxfeaturesCol = Param(Params._dummy(), "userctxfeaturesCol",
                               "the name of the user and context features column")
    itemfeaturesCol = Param(Params._dummy(), "itemfeaturesCol",
                            "the name of the item features column")
    samplingCol = Param(Params._dummy(), "samplingCol",
                        "the name of the column to use for acceptance sampling, usually same as itemCol")
    filterItemsCol = Param(Params._dummy(), "filterItemsCol",
                           "the name of the column to use for recommendation filtering")

    sampler = Param(Params._dummy(), "sampler",
                    "the sampler to use, one of uniform, exp and crossbatch")
    rho = Param(Params._dummy(), "rho",
                "the rho value to use for the exp sampler",
                typeConverter=TypeConverters.toFloat)
    batchSize = Param(Params._dummy(), "batchSize",
                      "the worker mini-batch size",
                      typeConverter=TypeConverters.toInt)
    numDims = Param(Params._dummy(), "numDims",
                    "the number of dimensions (k)",
                    typeConverter=TypeConverters.toInt)
    linearReg = Param(Params._dummy(), "linearReg",
                      "the regularization rate for the linear weights",
                      typeConverter=TypeConverters.toFloat)
    factorsReg = Param(Params._dummy(), "factorsReg",
                       "the regularization rate for the factor weights",
                       typeConverter=TypeConverters.toFloat)

    numParameterServers = Param(Params._dummy(), "numParameterServers",
                                "the number of parameter servers",
                                typeConverter=TypeConverters.toInt)
    parameterServerHost = Param(Params._dummy(), "parameterServerHost",
                                "the master host of the running parameter servers. "
                                "If this is not set a standalone parameter server cluster is started in this Spark application.")

    loadMetadata = Param(Params._dummy(), "loadMetadata",
                         "Whether the meta data of the data frame to fit should be loaded from HDFS. " +
                         "This allows skipping the meta data computation stages when fitting on the same data frame " +
                         "with different parameters. Meta data for \"cross-batch\" and \"uniform\" sampling is intercompatible " +
                         "but \"exp\" requires its own meta data",
                         typeConverter=TypeConverters.toBoolean)
    saveMetadata = Param(Params._dummy(), "saveMetadata",
                         "Whether the meta data of the fitted data frame should be saved to HDFS",
                         typeConverter=TypeConverters.toBoolean)
    metadataPath = Param(Params._dummy(), "metadataPath",
                         "The HDFS path to load meta data for the fit data frame from or to save the fitted meta data to")

    treeDepth = Param(Params._dummy(), "treeDepth",
                      "The depth to use for tree reduce when computing the meta data. " +
                      "To avoid OOM errors, this has to be set sufficiently large but lower depths might lead to faster runtimes",
                      typeConverter=TypeConverters.toInt)

    def getUserCol(self):
        return self.getOrDefault(self.userCol)

    def getItemCol(self):
        return self.getOrDefault(self.itemCol)

    def getUserctxfeaturesCol(self):
        return self.getOrDefault(self.userctxfeaturesCol)

    def getItemfeaturesCol(self):
        return self.getOrDefault(self.itemfeaturesCol)

    def getSamplingCol(self):
        return self.getOrDefault(self.samplingCol)

    def getFilterItemsCol(self):
        return self.getOrDefault(self.filterItemsCol)

    def getRho(self):
        return self.getOrDefault(self.rho)

    def getBatchSize(self):
        return self.getOrDefault(self.batchSize)

    def getNumDims(self):
        return self.getOrDefault(self.numDims)

    def getLinearReg(self):
        return self.getOrDefault(self.linearReg)

    def getFactorsReg(self):
        return self.getOrDefault(self.factorsReg)

    def getNumParameterServers(self):
        return self.getOrDefault(self.numParameterServers)

    def getParameterServerHost(self):
        return self.getOrDefault(self.parameterServerHost)

    def getLoadMetadata(self):
        return self.getOrDefault(self.loadMetadata)

    def getSaveMetadata(self):
        return self.getOrDefault(self.saveMetadata)

    def getMetadataPath(self):
        return self.getOrDefault(self.metadataPath)

    def getTreeDepth(self):
        return self.getOrDefault(self.treeDepth)


@inherit_doc
class GlintFMPair(JavaEstimator, _GlintFMPairParams, JavaMLReadable, JavaMLWritable):

    @keyword_only
    def __init__(self, userCol="userid", itemCol="itemid", userctxfeaturesCol="userctxfeatures",
                 itemfeaturesCol="itemfeatures", samplingCol="", filterItemsCol="",
                 maxIter=1000, stepSize=0.1, seed=1, sampler="uniform", rho=1.0,
                 batchSize=256, numDims=150, linearReg=0.01, factorsReg=0.001,
                 numParameterServers=3, parameterServerHost="",
                 loadMetadata=False, saveMetadata=False, metadataPath="", treeDepth=2):
        super(GlintFMPair, self).__init__()
        self._java_obj = self._new_java_obj("org.apache.spark.ml.recommendation.GlintFMPair", self.uid)
        self._setDefault(userCol="userid", itemCol="itemid", userctxfeaturesCol="userctxfeatures",
                         itemfeaturesCol="itemfeatures", samplingCol="", filterItemsCol="",
                         maxIter=1000, stepSize=0.1, seed=1, sampler="uniform", rho=1.0,
                         batchSize=256, numDims=150, linearReg=0.01, factorsReg=0.001,
                         numParameterServers=3, parameterServerHost="",
                         loadMetadata=False, saveMetadata=False, metadataPath="", treeDepth=2)
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, userCol="userid", itemCol="itemid", userctxfeaturesCol="userctxfeatures",
                  itemfeaturesCol="itemfeatures", samplingCol="", filterItemsCol="",
                  maxIter=1000, stepSize=0.1, seed=1, sampler="uniform", rho=1.0,
                  batchSize=256, numDims=150, linearReg=0.01, factorsReg=0.001,
                  numParameterServers=3, parameterServerHost="",
                  loadMetadata=False, saveMetadata=False, metadataPath="", treeDepth=2):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def setUserCol(self, value):
        return self._set(userCol=value)

    def setItemCol(self, value):
        return self._set(itemCol=value)

    def setUserctxfeaturesCol(self, value):
        return self._set(userctxfeaturesCol=value)

    def setItemfeaturesCol(self, value):
        return self._set(itemfeaturesCol=value)

    def setSamplingCol(self, value):
        return self._set(samplingCol=value)

    def setFilterItemsCol(self, value):
        return self._set(filterItemsCol=value)

    def setRho(self, value):
        return self._set(rho=value)

    def setBatchSize(self, value):
        return self._set(batchSize=value)

    def setNumDims(self, value):
        return self._set(numDims=value)

    def setLinearReg(self, value):
        return self._set(linearReg=value)

    def setFactorsReg(self, value):
        return self._set(factorsReg=value)

    def setNumParameterServers(self, value):
        return self._set(numParameterServers=value)

    def setParameterServerHost(self, value):
        return self._set(parameterServerHost=value)

    def setLoadMetadata(self, value):
        return self._set(loadMetadata=value)

    def setSaveMetadata(self, value):
        return self._set(saveMetadata=value)

    def setMetadataPath(self, value):
        return self._set(metadataPath=value)

    def setTreeDepth(self, value):
        return self._set(treeDepth=value)

    def _create_model(self, java_model):
        return GlintFMPairModel(java_model)


recommendation.GlintFMPair = GlintFMPair


class GlintFMPairModel(JavaModel, _GlintFMPairParams, JavaMLReadable, JavaMLWritable):
    """
    Model fitted by :py:class:`GlintFMPair`.
    """

    __module__ = "pyspark.ml.recommendation"

    def setUserCol(self, value):
        return self._set(userCol=value)

    def setUserctxfeaturesCol(self, value):
        return self._set(userctxfeaturesCol=value)

    def setFilterItemsCol(self, value):
        return self._set(filterItemsCol=value)

    def recommendForUserSubset(self, dataset, numItems):
        """
        Returns top numItems items recommended for each user id in the input data set

        :param dataset: The dataset containing a column of user ids and user context features.
            The column names must match userCol, userctxFeaturesCol and,
            if filtering should be used, also filterItemsCol.
        :param numItems: The maximum number of recommendations for each user
        :return: A dataframe of (userCol: Int, recommendations),
            where recommendations are stored as an array of (itemCol: Int, score: Float) rows.
        """
        self._transfer_params_to_java()
        return self._call_java("recommendForUserSubset", dataset, numItems)

    @classmethod
    def load(cls, path, parameterServerHost=""):
        """
        Loads a :py:class:`GlintFMPairModel`

        :param path: The path
        :param parameterServerHost: The master host of the running parameter servers.
            If this is not set a standalone parameter server cluster is started in this Spark application.
        """
        reader = cls.read()

        def readerLoad(self, path, parameterServerHost):
            if not isinstance(path, basestring):
                raise TypeError("path should be a basestring, got type %s" % type(path))
            java_obj = self._jread.load(path, parameterServerHost)
            if not hasattr(self._clazz, "_from_java"):
                raise NotImplementedError("This Java ML type cannot be loaded into Python currently: %s" % self._clazz)
            return self._clazz._from_java(java_obj)

        reader.load = types.MethodType(readerLoad, reader)
        return reader.load(path, parameterServerHost)

    def destroy(self, terminateOtherClients=False):
        """
        Destroys the model and releases the underlying distributed models and broadcasts.
        This model can't be used anymore afterwards.

        :param terminateOtherClients: If other clients should be terminated. This is necessary if a glint cluster
            in another Spark application should be terminated.
        """
        self._call_java("destroy", terminateOtherClients)

recommendation.GlintFMPairModel = GlintFMPairModel


class _PopRankParams(HasPredictionCol):

    __module__ = "pyspark.ml.recommendation"

    userCol = Param(Params._dummy(), "userCol", "the name of the user id column",
                    typeConverter=TypeConverters.toString)

    itemCol = Param(Params._dummy(), "itemCol", "the name of the item id column",
                    typeConverter=TypeConverters.toString)

    filterUserItems = Param(Params._dummy(), "filterUserItems",
                            "whether the items of a user should be filtered from the recommendations for the user",
                            typeConverter=TypeConverters.toBoolean)

    def getUserCol(self):
        return self.getOrDefault(self.userCol)

    def getItemCol(self):
        return self.getOrDefault(self.itemCol)

    def getFilterUserItems(self):
        return self.getOrDefault(self.filterUserItems)


@inherit_doc
class PopRank(JavaEstimator, _PopRankParams, JavaMLReadable, JavaMLWritable):

    __module__ = "pyspark.ml.recommendation"

    @keyword_only
    def __init__(self, userCol="userid", itemCol="itemid", filterUserItems=False):
        super(PopRank, self).__init__()
        self._java_obj = self._new_java_obj(
            "org.apache.spark.ml.recommendation.PopRank", self.uid)
        self._setDefault(userCol="userid", itemCol="itemid", filterUserItems=False)
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, userCol="userid", itemCol="itemid", filterUserItems=False):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def setUserCol(self, value):
        return self._set(userCol=value)

    def setItemCol(self, value):
        return self._set(itemCol=value)

    def setFilterUserItems(self, value):
        return self._set(filterUserItems=value)

    def _create_model(self, java_model):
        return PopRankModel(java_model)


recommendation.PopRank = PopRank


class PopRankModel(JavaModel, _PopRankParams, JavaMLReadable, JavaMLWritable):
    """
    Model fitted by :py:class:`PopRank`.
    """

    __module__ = "pyspark.ml.recommendation"

    def setFilterUserItems(self, value):
        return self._set(filterUserItems=value)

    def recommendForUserSubset(self, dataset, numItems):
        """
        Returns top numItems items recommended for each user id in the input data set

        :param dataset  The dataset containing a column of user ids. The column name must match userCol
        :param numItems The maximum number of recommendations for each user
        :return A dataframe of (userCol: Int, recommendations), where recommendations are stored
                as an array of (score: Float, itemCol: Int) rows. Or if exploded
                a dataframe of (userCol: Int, score: Float, itemCol: Int) rows.
        """
        self._transfer_params_to_java()
        return self._call_java("recommendForUserSubset", dataset, numItems)


recommendation.PopRankModel = PopRankModel


class _SAGHParams(HasPredictionCol):

    __module__ = "pyspark.ml.recommendation"

    userCol = Param(Params._dummy(), "userCol", "the name of the user id column",
                    typeConverter=TypeConverters.toString)

    itemCol = Param(Params._dummy(), "itemCol", "the name of the item id column",
                    typeConverter=TypeConverters.toString)

    artistCol = Param(Params._dummy(), "artistCol", "the name of the artist id column",
                      typeConverter=TypeConverters.toString)

    filterUserItems = Param(Params._dummy(), "filterUserItems",
                            "whether the items of a user should be filtered from the recommendations for the user",
                            typeConverter=TypeConverters.toBoolean)

    def getUserCol(self):
        return self.getOrDefault(self.userCol)

    def getItemCol(self):
        return self.getOrDefault(self.itemCol)

    def getArtistCol(self):
        return self.getOrDefault(self.artistCol)

    def getFilterUserItems(self):
        return self.getOrDefault(self.filterUserItems)


@inherit_doc
class SAGH(JavaEstimator, _SAGHParams, JavaMLReadable, JavaMLWritable):

    __module__ = "pyspark.ml.recommendation"

    @keyword_only
    def __init__(self, userCol="userid", itemCol="itemid", filterUserItems=False):
        super(SAGH, self).__init__()
        self._java_obj = self._new_java_obj(
            "org.apache.spark.ml.recommendation.SAGH", self.uid)
        self._setDefault(userCol="userid", itemCol="itemid", artistCol="artid", filterUserItems=False)
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, userCol="userid", itemCol="itemid", artistCol="artid", filterUserItems=False):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def setUserCol(self, value):
        return self._set(userCol=value)

    def setItemCol(self, value):
        return self._set(itemCol=value)

    def setArtistCol(self, value):
        return self._set(artistCol=value)

    def setFilterUserItems(self, value):
        return self._set(filterUserItems=value)

    def _create_model(self, java_model):
        return SAGHModel(java_model)


recommendation.SAGH = SAGH


class SAGHModel(JavaModel, _SAGHParams, JavaMLReadable, JavaMLWritable):
    """
    Model fitted by :py:class:`SAGH`.
    """

    __module__ = "pyspark.ml.recommendation"

    def setFilterUserItems(self, value):
        return self._set(filterUserItems=value)

    def recommendForUserSubset(self, dataset, numItems):
        """
        Returns top numItems items recommended for each user id in the input data set

        :param dataset  The dataset containing a column of user ids. The column name must match userCol
        :param numItems The maximum number of recommendations for each user
        :return A dataframe of (userCol: Int, recommendations), where recommendations are stored
                as an array of (score: Float, itemCol: Int) rows. Or if exploded
                a dataframe of (userCol: Int, score: Float, itemCol: Int) rows.
        """
        self._transfer_params_to_java()
        return self._call_java("recommendForUserSubset", dataset, numItems)


recommendation.SAGHModel = SAGHModel


class _KNNParams(HasPredictionCol):

    __module__ = "pyspark.ml.recommendation"

    userCol = Param(Params._dummy(), "userCol", "the name of the user id column",
                    typeConverter=TypeConverters.toString)

    itemCol = Param(Params._dummy(), "itemCol", "the name of the item id column",
                    typeConverter=TypeConverters.toString)

    k = Param(Params._dummy(), "k", "the number of nearest neighbours",
              typeConverter=TypeConverters.toInt)

    filterUserItems = Param(Params._dummy(), "filterUserItems",
                            "whether the items of a user should be filtered from the recommendations for the user",
                            typeConverter=TypeConverters.toBoolean)

    def getUserCol(self):
        return self.getOrDefault(self.userCol)

    def getItemCol(self):
        return self.getOrDefault(self.itemCol)

    def getK(self):
        return self.getOrDefault(self.k)

    def getFilterUserItems(self):
        return self.getOrDefault(self.filterUserItems)


class KNN(JavaTransformer, _KNNParams, JavaMLReadable, JavaMLWritable):

    __module__ = "pyspark.ml.recommendation"

    @keyword_only
    def __init__(self, userCol="userid", itemCol="itemid", k=None, filterUserItems=False):
        super(KNN, self).__init__()
        self._java_obj = self._new_java_obj(
            "org.apache.spark.ml.recommendation.KNN", self.uid)
        self._setDefault(userCol="userid", itemCol="itemid", filterUserItems=False)
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, userCol="userid", itemCol="itemid", k=None, filterUserItems=False):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def setUserCol(self, value):
        return self._set(userCol=value)

    def setItemCol(self, value):
        return self._set(itemCol=value)

    def setK(self, value):
        return self._set(k=value)

    def setFilterUserItems(self, value):
        return self._set(filterUserItems=value)

    def recommendForUserSubset(self, dataset, fitDataset, numItems):
        """
        Returns top numItems items recommended for each user id in the input data set

        :param dataset  The dataset containing a column of user ids. The column name must match userCol
        :param numItems The maximum number of recommendations for each user
        :return A dataframe of (userCol: Int, recommendations), where recommendations are stored
                as an array of (score: Float, itemCol: Int) rows. Or if exploded
                a dataframe of (userCol: Int, score: Float, itemCol: Int) rows.
        """
        self._transfer_params_to_java()
        return self._call_java("recommendForUserSubset", dataset, fitDataset, numItems)


recommendation.KNN = KNN


class _WeightHotEncoderParams(HasInputCols, HasOutputCols, HasHandleInvalid):

    __module__ = "pyspark.ml.feature"

    handleInvalid = Param(Params._dummy(), "handleInvalid", "How to handle invalid data during " +
                          "transform(). Options are 'keep' (invalid data presented as an extra " +
                          "categorical feature) or error (throw an error). Note that this Param " +
                          "is only used during transform; during fitting, invalid data will " +
                          "result in an error.",
                          typeConverter=TypeConverters.toString)

    dropLast = Param(Params._dummy(), "dropLast", "whether to drop the last category",
                     typeConverter=TypeConverters.toBoolean)

    weights = Param(Params._dummy(), "weights",
                    "the weight to use instead of 1.0 for hot encoding per column",
                    typeConverter=TypeConverters.toListFloat)

    groupCols = Param(Params._dummy(), "groupCols", "the columns to use for grouping",
                      typeConverter=TypeConverters.toListString)

    groupWeighting = Param(Params._dummy(), "groupWeighting",
                           "the group weighting to use, one of equi, sqrt and one",
                           typeConverter=TypeConverters.toString)

    def getDropLast(self):
        return self.getOrDefault(self.dropLast)

    def getWeights(self):
        return self.getOrDefault(self.weights)

    def getGroupCols(self):
        return self.getOrDefault(self.groupCols)

    def getGroupWeighting(self):
        return self.getOrDefault(self.groupWeighting)


@inherit_doc
class WeightHotEncoderEstimator(JavaEstimator, _WeightHotEncoderParams, JavaMLReadable, JavaMLWritable):

    __module__ = "pyspark.ml.feature"

    @keyword_only
    def __init__(self, inputCols=None, outputCols=None, handleInvalid="error", dropLast=True,
                 weights=[], groupCols=[], groupWeighting="sqrt"):
        super(WeightHotEncoderEstimator, self).__init__()
        self._java_obj = self._new_java_obj(
            "org.apache.spark.ml.feature.WeightHotEncoderEstimator", self.uid)
        self._setDefault(handleInvalid="error", dropLast=True, weights=[], groupCols=[], groupWeighting="sqrt")
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCols=None, outputCols=None, handleInvalid="error", dropLast=True,
                  weights=[], groupCols=[], groupWeighting="sqrt"):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def setDropLast(self, value):
        return self._set(dropLast=value)

    def setInputCols(self, value):
        return self._set(inputCols=value)

    def setOutputCols(self, value):
        return self._set(outputCols=value)

    def setHandleInvalid(self, value):
        return self._set(handleInvalid=value)

    def setWeights(self, value):
        return self._set(weights=value)

    def setGroupCols(self, value):
        return self._set(groupCols=value)

    def setGroupWeighting(self, value):
        return self._set(groupWeighting=value)

    def _create_model(self, java_model):
        return WeightHotEncoderModel(java_model)


feature.WeightHotEncoderEstimator = WeightHotEncoderEstimator


class WeightHotEncoderModel(JavaModel, _WeightHotEncoderParams, JavaMLReadable, JavaMLWritable):

    __module__ = "pyspark.ml.feature"

    def setDropLast(self, value):
        return self._set(dropLast=value)

    def setInputCols(self, value):
        return self._set(inputCols=value)

    def setOutputCols(self, value):
        return self._set(outputCols=value)

    def setHandleInvalid(self, value):
        return self._set(handleInvalid=value)

    def setWeights(self, value):
        return self._set(weights=value)

    def setGroupCols(self, value):
        return self._set(groupCols=value)

    def setGroupWeighting(self, value):
        return self._set(groupWeighting=value)

    @property
    def categorySizes(self):
        """
        Original number of categories for each feature being encoded.
        The array contains one value for each input column, in order.
        """
        return self._call_java("categorySizes")


feature.WeightHotEncoderModel = WeightHotEncoderModel
