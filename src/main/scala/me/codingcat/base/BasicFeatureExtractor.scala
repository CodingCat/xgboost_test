package me.codingcat.base

import ml.dmlc.xgboost4j.scala.DMatrix
import org.apache.spark.SparkContext
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.rdd.RDD

abstract class BasicFeatureExtractor(@transient val sc: SparkContext) extends Serializable {

  def generateLabeledPointRDD(datasetPath: String, containsGroundTruth: Boolean = true):
      RDD[LabeledPoint]

  def generateDMatrix(datasetPath: String, containsGroundTruth: Boolean): DMatrix
}