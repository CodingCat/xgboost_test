package me.codingcat.rosseman

import scala.collection.mutable.ListBuffer
import scala.io.Source

import org.apache.spark.sql.SparkSession

object Main {

  private def parseStoreFile(storeFilePath: String): List[Store] = {
    var isHeader = true
    val storeInstances = new ListBuffer[Store]
    var cnt = 0
    for (line <- Source.fromFile(storeFilePath).getLines()) {
      if (isHeader) {
        isHeader = false
      } else {
        try {
          val strArray = line.split(",")
          if (strArray.length == 10) {
            val Array(storeIdStr, storeTypeStr, assortmentStr, competitionDistanceStr,
            competitionOpenSinceMonthStr, competitionOpenSinceYearStr, promo2Str, promo2SinceWeekStr,
            promo2SinceYearStr, promoIntervalStr) = line.split(",")
            storeInstances += Store(storeIdStr.toInt, storeTypeStr, assortmentStr,
              if (competitionDistanceStr == "") -1 else competitionDistanceStr.toInt,
              if (competitionOpenSinceMonthStr == "" ) -1 else competitionOpenSinceMonthStr.toInt,
              if (competitionOpenSinceYearStr == "" ) -1 else competitionOpenSinceYearStr.toInt,
              promo2Str.toInt,
              if (promo2Str == "0") -1 else promo2SinceWeekStr.toInt,
              if (promo2Str == "0") -1 else promo2SinceYearStr.toInt,
              promoIntervalStr.replace("\"", ""))
          } else {
            val Array(storeIdStr, storeTypeStr, assortmentStr, competitionDistanceStr,
            competitionOpenSinceMonthStr, competitionOpenSinceYearStr, promo2Str, promo2SinceWeekStr,
            promo2SinceYearStr, firstMonth, secondMonth, thirdMonth, forthMonth) = line.split(",")
            storeInstances += Store(storeIdStr.toInt, storeTypeStr, assortmentStr,
              if (competitionDistanceStr == "") -1 else competitionDistanceStr.toInt,
              if (competitionOpenSinceMonthStr == "" ) -1 else competitionOpenSinceMonthStr.toInt,
              if (competitionOpenSinceYearStr == "" ) -1 else competitionOpenSinceYearStr.toInt,
              promo2Str.toInt,
              if (promo2Str == "0") -1 else promo2SinceWeekStr.toInt,
              if (promo2Str == "0") -1 else promo2SinceYearStr.toInt,
              firstMonth.replace("\"", "") + "," + secondMonth + "," + thirdMonth + "," +
                forthMonth.replace("\"", ""))
          }
        } catch {
          case e: Exception =>
            e.printStackTrace()
            println(cnt)
            sys.exit(1)
        }
      }
      cnt += 1
    }
    storeInstances.toList
  }

  //"Store","DayOfWeek","Date","Sales","Customers","Open","Promo","StateHoliday","SchoolHoliday"
  // 2 1,5,2015-07-31,5263,555,1,1,"0","1"
  private def parseTrainingFile(trainingPath: String): List[SalesRecord] = {
    var isHeader = true
    val records = new ListBuffer[SalesRecord]
    for (line <- Source.fromFile(trainingPath).getLines()) {
      if (isHeader) {
        isHeader = false
      } else {
        val Array(storeIdStr, daysOfWeekStr, dateStr, salesStr, customerStr, openStr, promoStr,
        stateHolidayStr, schoolHolidayStr) = line.split(",")
        val salesRecord = SalesRecord(storeIdStr.toInt, daysOfWeekStr.toInt, dateStr,
          salesStr.toInt, customerStr.toInt, openStr.toInt, promoStr.toInt, stateHolidayStr,
          schoolHolidayStr)
        records += salesRecord
      }
    }
    records.toList
  }

  def main(args: Array[String]): Unit = {
    val sparkSession = SparkSession.builder().appName("rosseman").getOrCreate()
    import sparkSession.implicits._

    // parse training file to data frame
    val trainingPath = args(0)
    val allSalesRecords = parseTrainingFile(trainingPath)
    // create dataset
    val salesRecordsDF = allSalesRecords.toDF
    salesRecordsDF.show()

    // parse store file to data frame
    val storeFilePath = args(1)
    val allStores = parseStoreFile(storeFilePath)
    val storesDS = allStores.toDF()
    storesDS.show()
  }
}
