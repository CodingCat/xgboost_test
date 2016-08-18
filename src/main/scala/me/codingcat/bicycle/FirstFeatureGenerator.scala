package me.codingcat.bicycle

import java.text.SimpleDateFormat
import java.util.Calendar

import ml.dmlc.xgboost4j.LabeledPoint
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

class FirstFeatureGenerator(sc: SparkContext) extends BasicFeatureExtractor(sc) {

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

  override def generateFeatureRDD(datasetPath: String): RDD[LabeledPoint] = {
    val rawFeatureRDD = generateRawFeature(datasetPath)
    val featureRdd = rawFeatureRDD.map {
      case Feature(date, season, holiday, workingDay, weather, temp, atemp, humidity, windspeed,
        groundTruth) =>
        // expand date feature to y/m/d/h/dayofweek
        val labeledPoint = new LabeledPoint()
        labeledPoint.indices = new Array[Int](13)
        labeledPoint.values = new Array[Float](13)
        val (year, month, day, hour, dayOfWeek) = translateDateStringToYMDHDOfW(date)
        for (i <- 0 until 13) {
          labeledPoint.indices(i) = i
        }
        labeledPoint.values(0) = year
        labeledPoint.values(1) = month
        labeledPoint.values(2) = day
        labeledPoint.values(3) = hour
        labeledPoint.values(4) = dayOfWeek
        labeledPoint.values(5) = season
        labeledPoint.values(6) = holiday
        labeledPoint.values(7) = workingDay
        labeledPoint.values(8) = weather
        labeledPoint.values(9) = temp
        labeledPoint.values(10) = atemp
        labeledPoint.values(11) = humidity
        labeledPoint.values(12) = windspeed
        if (groundTruth.isDefined) {
          labeledPoint.label = groundTruth.get
        }
        labeledPoint
    }
    featureRdd
  }
}
