package me.codingcat.rosseman

import me.codingcat.base.BasicFeatureExtractor
import ml.dmlc.xgboost4j.scala.DMatrix
import org.apache.spark.SparkContext
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.rdd.RDD

/*
class FeatureExtractor(@transient override val sc: SparkContext)
  extends BasicFeatureExtractor(sc) {
  override def generateLabeledPointRDD(datasetPath: String, containsGroundTruth: Boolean):
      RDD[LabeledPoint] = {

  }

  override def generateDMatrix(datasetPath: String, containsGroundTruth: Boolean): DMatrix = {

  }
}
*/