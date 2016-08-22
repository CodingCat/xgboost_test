package me.codingcat.titanic

import me.codingcat.base.BasicFeatureExtractor
import ml.dmlc.xgboost4j.scala.DMatrix
import org.apache.spark.SparkContext
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

class OneHotAndNormalFeatureGenerator(
    @transient sc: SparkContext) extends BasicFeatureExtractor(sc) {

  override def generateLabeledPointRDD(datasetPath: String, containsGroundTruth: Boolean):
      RDD[LabeledPoint] = {
    null
  }

  override def generateDMatrix(datasetPath: String, containsGroundTruth: Boolean): DMatrix = {
    null
  }
}
