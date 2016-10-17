package me.codingcat.bicycle

import java.text.SimpleDateFormat
import java.util.Calendar

import scala.collection.mutable.ListBuffer
import scala.io.Source

import me.codingcat.base.BasicFeatureExtractor
import ml.dmlc.xgboost4j.scala.DMatrix
import ml.dmlc.xgboost4j.{LabeledPoint => XGBoostLabeledPoint}
import org.apache.spark.SparkContext
import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.rdd.RDD

case class Feature(date: String, season: Int, holiday: Int, workingDay: Int, weather: Int,
                   temp: Float, atemp: Float, humidity: Int, windspeed: Float,
                   groundTruth: Option[Int])

class FirstFeatureGenerator(@transient override val sc: SparkContext)
  extends BasicFeatureExtractor(sc) {

  private val c = Calendar.getInstance()

  private val simpleDate = new SimpleDateFormat("yyyy-MM-dd")

  private def generateRawFeatureIterator(rawFilePath: String, containGroundTruth: Boolean = true):
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

  private def generateRawFeatureRDD(rawFilePath: String, containGroundTruth: Boolean = true):
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

  private def translateDateStringToYMDHDOfW(dateString: String): (Int, Int, Int, Int, Int) = {
    val Array(yearMonthDay, hourMinSec) = dateString.split(" ")
    val Array(year, month, day) = yearMonthDay.split("-").map(_.toInt)
    val Array(hour, _, _) = hourMinSec.split(":").map(_.toInt)
    c.setTime(simpleDate.parse(yearMonthDay))
    val dayOfWeek = c.get(Calendar.DAY_OF_WEEK)
    (year, month, day, hour, dayOfWeek)
  }

  private def fromRawFeatureToIndicesAndValuesPair(feature: Feature): (Array[Int], Array[Float])
      = {
    val indices = new Array[Int](13)
    val values = new Array[Float](13)
    val (year, month, day, hour, dayOfWeek) = translateDateStringToYMDHDOfW(feature.date)
    for (i <- 0 until 13) {
      indices(i) = i
    }
    values(0) = year
    values(1) = month
    values(2) = day
    values(3) = hour
    values(4) = dayOfWeek
    values(5) = feature.season
    values(6) = feature.holiday
    values(7) = feature.workingDay
    values(8) = feature.weather
    values(9) = feature.temp
    values(10) = feature.atemp
    values(11) = feature.humidity
    values(12) = feature.windspeed
    (indices, values)
  }

  private def fromRawFeatureToSparseVector(feature: Feature): SparseVector = {
    // expand date feature to y/m/d/h/dayofweek
    val (indices, values) = fromRawFeatureToIndicesAndValuesPair(feature)
    new SparseVector(13, indices, values.map(_.toDouble))
  }

  override def generateLabeledPointRDD(datasetPath: String, containsGroundTruth: Boolean = true):
      RDD[LabeledPoint] = {
    val rawFeatureRDD = generateRawFeatureRDD(datasetPath)
    val featureRdd = rawFeatureRDD.map {
      case f @ Feature(date, season, holiday, workingDay, weather, temp, atemp, humidity, windspeed,
        groundTruth) =>
        LabeledPoint(groundTruth.getOrElse(-1).toDouble, fromRawFeatureToSparseVector(f))
    }
    featureRdd.cache()
  }

  override def generateDMatrix(datasetPath: String, containsGroundTruth: Boolean = true):
      DMatrix = {
    val rawFeatureItr = generateRawFeatureIterator(datasetPath, containsGroundTruth)
    val transformedFeatureItr = rawFeatureItr.map {
      case f @ Feature(date, season, holiday, workingDay, weather, temp, atemp, humidity,
          windspeed, groundTruth) =>
        // expand date feature to y/m/d/h/dayofweek
        val (indices, values) = fromRawFeatureToIndicesAndValuesPair(f)
        XGBoostLabeledPoint.fromSparseVector(groundTruth.getOrElse(-1).toFloat, indices, values)
    }
    new DMatrix(transformedFeatureItr)
  }
}
