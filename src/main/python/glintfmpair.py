import sys
import types
if sys.version > '3':
    basestring = str

from pyspark import keyword_only
from pyspark.ml import recommendation
from pyspark.ml.param.shared import *
from pyspark.ml.util import JavaMLReadable, JavaMLWritable
from pyspark.ml.wrapper import JavaEstimator, JavaModel
from pyspark.ml.common import inherit_doc

__all__ = ['GlintFMPair', 'GlintFMPairModel']


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

    aggregateDepth = Param(Params._dummy(), "aggregateDepth",
                           "The depth to use for tree aggregation when computing the meta data. " +
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

    def getAggregateDepth(self):
        return self.getOrDefault(self.aggregateDepth)


@inherit_doc
class GlintFMPair(JavaEstimator, _GlintFMPairParams, JavaMLReadable, JavaMLWritable):

    @keyword_only
    def __init__(self, userCol="userid", itemCol="itemid", userctxfeaturesCol="userctxfeatures",
                 itemfeaturesCol="itemfeatures", samplingCol="", filterItemsCol="",
                 maxIter=1000, stepSize=0.1, seed=1, sampler="uniform", rho=1.0,
                 batchSize=256, numDims=150, linearReg=0.01, factorsReg=0.001,
                 numParameterServers=3, parameterServerHost="",
                 loadMetadata=False, saveMetadata=False, metadataPath="", aggregateDepth=2):
        super(GlintFMPair, self).__init__()
        self._java_obj = self._new_java_obj("org.apache.spark.ml.recommendation.GlintFMPair", self.uid)
        self._setDefault(userCol="userid", itemCol="itemid", userctxfeaturesCol="userctxfeatures",
                         itemfeaturesCol="itemfeatures", samplingCol="", filterItemsCol="",
                         maxIter=1000, stepSize=0.1, seed=1, sampler="uniform", rho=1.0,
                         batchSize=256, numDims=150, linearReg=0.01, factorsReg=0.001,
                         numParameterServers=3, parameterServerHost="",
                         loadMetadata=False, saveMetadata=False, metadataPath="", aggregateDepth=2)
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, userCol="userid", itemCol="itemid", userctxfeaturesCol="userctxfeatures",
                  itemfeaturesCol="itemfeatures", samplingCol="", filterItemsCol="",
                  maxIter=1000, stepSize=0.1, seed=1, sampler="uniform", rho=1.0,
                  batchSize=256, numDims=150, linearReg=0.01, factorsReg=0.001,
                  numParameterServers=3, parameterServerHost="",
                  loadMetadata=False, saveMetadata=False, metadataPath="", aggregateDepth=2):
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

    def setAggregateDepth(self, value):
        return self._set(aggregateDepth=value)

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
