package me.codingcat.bicycle

import java.text.SimpleDateFormat
import java.util.Calendar

import ml.dmlc.xgboost4j.scala.DMatrix
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.SparseVector
import ml.dmlc.xgboost4j.{LabeledPoint => XGBoostLabeledPoint}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

class FirstFeatureGenerator(@transient sc: SparkContext) extends BasicFeatureExtractor(sc) {

  private val c = Calendar.getInstance()

  private val simpleDate = new SimpleDateFormat("yyyy-MM-dd")

  private def translateDateStringToYMDHDOfW(dateString: String): (Int, Int, Int, Int, Int) = {
    val Array(yearMonthDay, hourMinSec) = dateString.split(" ")
    val Array(year, month, day) = yearMonthDay.split("-").map(_.toInt)
    val Array(hour, _, _) = hourMinSec.split(":").map(_.toInt)
    c.setTime(simpleDate.parse(yearMonthDay))
    val dayOfWeek = c.get(Calendar.DAY_OF_WEEK)
    (year, month, day, hour, dayOfWeek)
  }

  private def fromRawFeatureToIndicesAndValuesArray(feature: Feature): (Array[Int], Array[Float])
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
    val (indices, values) = fromRawFeatureToIndicesAndValuesArray(feature)
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
        val (indices, values) = fromRawFeatureToIndicesAndValuesArray(f)
        XGBoostLabeledPoint.fromSparseVector(groundTruth.getOrElse(-1).toFloat, indices, values)
    }
    new DMatrix(transformedFeatureItr)
  }
}
