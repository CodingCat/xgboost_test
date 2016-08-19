package me.codingcat.bicycle

import scala.collection.mutable.ListBuffer
import scala.io.Source

import ml.dmlc.xgboost4j.scala.DMatrix
import org.apache.spark.SparkContext
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

case class Feature(date: String, season: Int, holiday: Int, workingDay: Int, weather: Int,
                   temp: Float, atemp: Float, humidity: Int, windspeed: Float,
                   groundTruth: Option[Int])

abstract class BasicFeatureExtractor(@transient sc: SparkContext) extends Serializable {

  protected def generateRawFeatureIterator(rawFilePath: String, containGroundTruth: Boolean = true):
      Iterator[Feature] = {
    val featureList = new ListBuffer[Feature]
    for (line <- Source.fromFile(rawFilePath).getLines()) {
      val featureArray = line.split(",")
      val date = featureArray(0)
      val season = featureArray(1).toInt
      val holiday = featureArray(2).toInt
      val workingday = featureArray(3).toInt
      val weather = featureArray(4).toInt
      val temp = featureArray(5).toFloat
      val atemp = featureArray(6).toFloat
      val humidity = featureArray(7).toInt
      val windspeed = featureArray(8).toFloat
      val count = {
        if (containGroundTruth) {
          Some(featureArray(featureArray.length - 1).toInt)
        } else {
          None
        }
      }
      featureList +=
        Feature(date, season, holiday, workingday, weather, temp, atemp, humidity, windspeed, count)
    }
    featureList.iterator
  }

  protected def generateRawFeatureRDD(rawFilePath: String, containGroundTruth: Boolean = true):
      RDD[Feature] = {
    val textFileRDD = sc.textFile(rawFilePath)
    textFileRDD.map{
      line =>
        val featureArray = line.split(",")
        val date = featureArray(0)
        val season = featureArray(1).toInt
        val holiday = featureArray(2).toInt
        val workingday = featureArray(3).toInt
        val weather = featureArray(4).toInt
        val temp = featureArray(5).toFloat
        val atemp = featureArray(6).toFloat
        val humidity = featureArray(7).toInt
        val windspeed = featureArray(8).toFloat
        val count = {
          if (containGroundTruth) {
            Some(featureArray(featureArray.length - 1).toInt)
          } else {
            None
          }
        }
        Feature(date, season, holiday, workingday, weather, temp, atemp, humidity, windspeed, count)
    }
  }

  def generateFeatureRDD(datasetPath: String): RDD[LabeledPoint]

  def genenerateFeatureDMatrix(datasetPath: String, containsGroundTruth: Boolean): DMatrix
}
