package me.codingcat.bicycle

import java.text.SimpleDateFormat
import java.util.Calendar

import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.SparseVector
import org.apache.spark.mllib.regression.LabeledPoint
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
        val indices = new Array[Int](13)
        val values = new Array[Float](13)
        val (year, month, day, hour, dayOfWeek) = translateDateStringToYMDHDOfW(date)
        for (i <- 0 until 13) {
          indices(i) = i
        }
        values(0) = year
        values(1) = month
        values(2) = day
        values(3) = hour
        values(4) = dayOfWeek
        values(5) = season
        values(6) = holiday
        values(7) = workingDay
        values(8) = weather
        values(9) = temp
        values(10) = atemp
        values(11) = humidity
        values(12) = windspeed
        val featureVector = new SparseVector(13, indices, values.map(_.toDouble))
        LabeledPoint(groundTruth.getOrElse(-1).toDouble, featureVector)
    }
    featureRdd
  }
}
