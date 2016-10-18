package me.codingcat.rosseman

import scala.collection.mutable
import scala.collection.mutable.ListBuffer
import scala.io.Source

import ml.dmlc.xgboost4j.scala.spark.XGBoost
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer}
import org.apache.spark.sql.{DataFrame, SparkSession}

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

  private def featureEngineering(ds: DataFrame): DataFrame = {
    import org.apache.spark.sql.functions._
    import ds.sparkSession.implicits._
    val stateHolidayIndexer = new StringIndexer()
      .setInputCol("stateHoliday")
      .setOutputCol("stateHolidayIndex")
    val schoolHolidayIndexer = new StringIndexer()
      .setInputCol("schoolHoliday")
      .setOutputCol("schoolHolidayIndex")
    val storeTypeIndexer = new StringIndexer()
      .setInputCol("storeType")
      .setOutputCol("storeTypeIndex")
    val assortmentIndexer = new StringIndexer()
      .setInputCol("assortment")
      .setOutputCol("assortmentIndex")
    val promoInterval = new StringIndexer()
      .setInputCol("promoInterval")
      .setOutputCol("promoIntervalIndex")
    val filteredDS = ds.filter($"sales" > 0).filter($"open" > 0)
    // parse date
    val dsWithDayCol =
      filteredDS.withColumn("day", udf((dateStr: String) => dateStr.split("-")(2).toInt).apply(col("date")))
    val dsWithMonthCol =
      dsWithDayCol.withColumn("month", udf((dateStr: String) => dateStr.split("-")(1).toInt).apply(col("date")))
    val dsWithYearCol =
      dsWithMonthCol.withColumn("year", udf((dateStr: String) => dateStr.split("-")(0).toInt).apply(col("date")))
    val dsWithLogSales = dsWithYearCol.withColumn("logSales",
      udf((sales: Int) => math.log(sales)).apply(col("sales")))

    // fill with mean values
    val meanCompetitionDistance = dsWithLogSales.select(avg("competitionDistance")).first()(0).
      asInstanceOf[Double]
    println("====" + meanCompetitionDistance)
    val finalDS = dsWithLogSales.withColumn("transformedCompetitionDistance",
      udf((distance: Int) => if (distance > 0) distance.toDouble else meanCompetitionDistance).
        apply(col("competitionDistance")))

    val vectorAssembler = new VectorAssembler()
      .setInputCols(Array("storeId", "daysOfWeek", "promo", "competitionDistance", "promo2", "day",
      "month", "year", "transformedCompetitionDistance", "stateHolidayIndex", "schoolHolidayIndex",
      "storeTypeIndex", "assortmentIndex", "promoIntervalIndex"))
      .setOutputCol("features")

    val pipeline = new Pipeline().setStages(
      Array(stateHolidayIndexer, schoolHolidayIndexer, storeTypeIndexer, assortmentIndexer,
        promoInterval, vectorAssembler))

    pipeline.fit(finalDS).transform(finalDS).
      drop("stateHoliday", "schoolHoliday", "storeType", "assortment", "promoInterval", "sales",
      "promo2SinceWeek", "customers", "promoInterval", "competitionOpenSinceYear",
        "competitionOpenSinceMonth", "promo2SinceYear", "competitionDistance", "date")
  }

  def main(args: Array[String]): Unit = {
    val sparkSession = SparkSession.builder().appName("rosseman").getOrCreate()
    import sparkSession.implicits._

    // parse training file to data frame
    val trainingPath = args(0)
    val allSalesRecords = parseTrainingFile(trainingPath)
    // create dataset
    val salesRecordsDF = allSalesRecords.toDF

    // parse store file to data frame
    val storeFilePath = args(1)
    val allStores = parseStoreFile(storeFilePath)
    val storesDS = allStores.toDF()

    val fullDataset = salesRecordsDF.join(storesDS, "storeId")
    val featureEngineeredDF = featureEngineering(fullDataset)
    // prediction
    val params = new mutable.HashMap[String, Any]()
    params += "eta" -> 0.1
    params += "max_depth" -> 6
    params += "silent" -> 0
    params += "ntreelimit" -> 1000
    params += "objective" -> "reg:linear"
    params += "subsample" -> 0.8
    params += "round" -> 100
    val trainedModel = XGBoost.trainWithDataFrame(featureEngineeredDF, params.toMap,
      100, 4, null, null, useExternalMemory = true, labelCol = "logSales")
  }
}