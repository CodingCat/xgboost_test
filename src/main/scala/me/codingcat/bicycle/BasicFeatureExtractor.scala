package me.codingcat.bicycle

import ml.dmlc.xgboost4j.LabeledPoint
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

case class Feature(date: String, season: Int, holiday: Int, workingDay: Int, weather: Int,
                   temp: Float, atemp: Float, humidity: Int, windspeed: Int,
                   groundTruth: Option[Int])

abstract class BasicFeatureExtractor(sc: SparkContext) extends Serializable {

  protected def generateRawFeature(rawFilePath: String, containGroundTruth: Boolean = true):
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
        val windspeed = featureArray(8).toInt
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
}
