import sys
if sys.version > '3':
    basestring = str

from pyspark import keyword_only
from pyspark.ml import feature
from pyspark.ml.param.shared import *
from pyspark.ml.util import JavaMLReadable, JavaMLWritable
from pyspark.ml.wrapper import JavaEstimator, JavaModel
from pyspark.ml.common import inherit_doc

__all__ = ['WeightHotEncoderEstimator', 'WeightHotEncoderModel']


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
        self._setDefault(handleInvalid="error", dropLast=True)
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
