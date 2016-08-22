package me.codingcat.titanic

import scala.collection.mutable.ListBuffer
import scala.io.Source

import me.codingcat.base.BasicFeatureExtractor
import ml.dmlc.xgboost4j.scala.DMatrix
import org.apache.spark.SparkContext
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD


case class PassengerInfo(pClass: Float, sex: String, age: Float, sibsp: Float, parch: Float,
                         ticket: String, fare: Float, cabin: String, embarked: String,
                         survived: Int)

case class TransformedPassengerInfo(
    pClass: Array[Float], sex: Array[Float], age: Float, sibsp: Float,
    parch: Float, ticketLength: Float, ticketStart: Float, fare: Float,
    cabin: Array[Float], Embarked: Array[Float])

class OneHotAndNormalFeatureGenerator(
    @transient sc: SparkContext) extends BasicFeatureExtractor(sc) {

  private def parseFileToRawPassengerInfo(filePath: String): List[PassengerInfo] = {
    val list = new ListBuffer[PassengerInfo]
    for (line <- Source.fromFile(filePath).getLines()) {
      val featureStringArray = line.split(",")
      val pClass = {if (featureStringArray(2) != null) featureStringArray(2).toFloat else -1.0f}
      val sex = featureStringArray(4)
      val age = {if (featureStringArray(5) != null) featureStringArray(5).toFloat else -1.0f}
      val sibsp = {if (featureStringArray(6) != null) featureStringArray(6).toFloat else -1.0f}
      val parch = {if (featureStringArray(7) != null) featureStringArray(7).toFloat else -1.0f}
      val ticket = featureStringArray(8)
      val fare = {if (featureStringArray(9) != null) featureStringArray(9).toFloat else -1.0f}
      val cabin = featureStringArray(10)
      val embarked = featureStringArray(11)
      val survived = featureStringArray(1).toInt
      list += PassengerInfo(pClass, sex, age, sibsp, parch, ticket, fare, cabin, embarked, survived)
    }
    list.toList
  }

  private def oneHotEncodingAndNormalization(passengers: List[PassengerInfo]): Unit = {
    // one hot encoding
    val pClassAllVars = passengers.map(_.pClass).distinct.zipWithIndex.toMap
    val sexAllVars = passengers.map(_.sex).distinct.zipWithIndex.toMap
    val cabin = passengers.map(_.cabin).distinct.zipWithIndex.toMap
    val embarked = passengers.map(_.embarked).distinct.zipWithIndex.toMap
    // normalization
    
  }

  override def generateLabeledPointRDD(datasetPath: String, containsGroundTruth: Boolean):
      RDD[LabeledPoint] = {
    null
  }

  override def generateDMatrix(datasetPath: String, containsGroundTruth: Boolean): DMatrix = {
    null
  }
}
