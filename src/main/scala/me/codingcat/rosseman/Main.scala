package me.codingcat.rosseman

import scala.collection.mutable
import scala.collection.mutable.ListBuffer
import scala.io.Source

import ml.dmlc.xgboost4j.scala.spark.{XGBoostEstimator, XGBoost}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, RegressionEvaluator}
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning._
import org.apache.spark.sql.{functions, Dataset, DataFrame, SparkSession}

object Main {

  private def parseStoreFile(storeFilePath: String): List[Store] = {
    var isHeader = true
    val storeInstances = new ListBuffer[Store]
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
            sys.exit(1)
        }
      }
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

    val r = pipeline.fit(finalDS).transform(finalDS).
      drop("stateHoliday", "schoolHoliday", "storeType", "assortment", "promoInterval", "sales",
      "promo2SinceWeek", "customers", "promoInterval", "competitionOpenSinceYear",
        "competitionOpenSinceMonth", "promo2SinceYear", "competitionDistance", "date")

    val average = r.select(avg($"logSales")).first().get(0).asInstanceOf[Double]
    r.withColumn("label",
      udf{(logSales: Double) => if (logSales> average) 0 else 1}.apply(col("logSales")))
  }

  private def crossValidation(
      xgboostParam: Map[String, Any],
      trainingData: Dataset[_]): TrainValidationSplitModel = {
    val xgbEstimator = new XGBoostEstimator(xgboostParam).setFeaturesCol("features").
      setLabelCol("label")
    val paramGrid = new ParamGridBuilder()
      .addGrid(xgbEstimator.round, Array(20, 50))
      .addGrid(xgbEstimator.eta, Array(0.1, 0.4))
      .build()
    val tv = new TrainValidationSplit()
      .setEstimator(xgbEstimator)
      .setEvaluator(new BinaryClassificationEvaluator().setLabelCol("label").
        setRawPredictionCol("probabilities"))
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.8)  // Use 3+ in practice
    tv.fit(trainingData)
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
    params += "silent" -> 1
    params += "ntreelimit" -> 1000
    params += "objective" -> "binary:logistic"
    params += "subsample" -> 0.8
    params += "nworkers" -> 4
    params += "eval_metric" -> "logloss"

    val bestModel = crossValidation(params.toMap, featureEngineeredDF)
    bestModel.transform(featureEngineeredDF).show()
  }
}
