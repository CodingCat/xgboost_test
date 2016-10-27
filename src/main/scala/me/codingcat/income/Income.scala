package me.codingcat.income

import scala.collection.mutable.ListBuffer
import scala.io.Source

import ml.dmlc.xgboost4j.scala.spark.XGBoostEstimator
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer}
import org.apache.spark.ml.tuning.{TrainValidationSplit, ParamGridBuilder}
import org.apache.spark.sql.{DataFrame, SparkSession}

case class IncomeRecord(age: Int, workClass: String, fnlwgt: Int, education: String,
                        educationNum: Int, maritalStatus: String, occupation: String,
                        relationship: String, race: String, sex: String, capitalGain: Int,
                        capitalLoss: Int, hoursPerWeek: Int, nativeCountry: String, label: String)

object Income {

  private def parseTrainingSet(trainingPath: String): List[IncomeRecord] = {
    val ret = new ListBuffer[IncomeRecord]
    var cnt = 0
    for (line <- Source.fromFile(trainingPath).getLines()) {
      cnt += 1
      try {
        val Array(age, workClass, fnlwgt, education, educationNum, martalStatus, occupation,
        relationship, race, sex, capitalGain, capitalLoss, hoursPerWeek, nativeCountry, label) =
          line.split(",")
        val instance = IncomeRecord(age.trim.toInt, workClass, fnlwgt.trim.toInt, education,
          educationNum.trim.toInt, martalStatus, occupation, relationship, race, sex,
          capitalGain.trim.toInt,
          capitalLoss.trim.toInt, hoursPerWeek.trim.toInt, nativeCountry, label)
        ret += instance
      } catch {
        case ex: Exception =>
          println(cnt)
          ex.printStackTrace()
      }
    }
    ret.toList
  }

  def featureTransformation(inputDF: DataFrame): DataFrame = {
    val workClassIndexer = new StringIndexer().setInputCol("workClass").setOutputCol("workClassIdx")
    val educationIndexer = new StringIndexer().setInputCol("education").setOutputCol("educationIdx")
    val maritalStatusIndexer = new StringIndexer().setInputCol("maritalStatus").setOutputCol("maritalStatusIdx")
    val occupationIndexer = new StringIndexer().setInputCol("occupation").setOutputCol("occupationIdx")
    val relationshipIndexer = new StringIndexer().setInputCol("relationship").setOutputCol("relationshipIdx")
    val raceIndexer = new StringIndexer().setInputCol("race").setOutputCol("raceIdx")
    val sexIndexer = new StringIndexer().setInputCol("sex").setOutputCol("sexIdx")
    val nativeCountryIndexer = new StringIndexer().setInputCol("nativeCountry").setOutputCol("nativeCountryIdx")
    val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("labelIdx")

    val vectorAssembler = new VectorAssembler().setInputCols(
      Array("age", "workClassIdx", "fnlwgt", "educationIdx", "educationNum",
        "maritalStatusIdx", "occupationIdx", "relationshipIdx", "raceIdx", "sexIdx", "capitalGain",
      "capitalLoss", "hoursPerWeek", "nativeCountryIdx", "labelIdx")
    ).setOutputCol("features")

    val pipeline = new Pipeline().setStages(Array(workClassIndexer, educationIndexer, maritalStatusIndexer,
      occupationIndexer, relationshipIndexer, raceIndexer, sexIndexer, nativeCountryIndexer,
      labelIndexer, vectorAssembler))
    pipeline.fit(inputDF).transform(inputDF).drop("workClass", "education", "maritalStatus",
      "occupation", "relationship", "race", "sex", "nativeCountry", "label")
  }

  def modelTuning(inputDF: DataFrame): DataFrame = {
    val est = new XGBoostEstimator(Map[String, Any]("num_round" -> 100,
      "objective" -> "binary:logistic", "nworkers" -> 4)).
      setLabelCol("labelIdx").setFeaturesCol("features")
    val paramGrid = new ParamGridBuilder().
      addGrid(est.eta, Array(0.1, 0.6)).
      addGrid(est.lambda, Array(0.5, 1)).build()
    val trainSplitValidator = new TrainValidationSplit().
      setEstimator(est).
      setEstimatorParamMaps(paramGrid).
      setTrainRatio(0.8).
      setEvaluator(new BinaryClassificationEvaluator().
        setRawPredictionCol("probabilities").setLabelCol("labelIdx"))
    trainSplitValidator.fit(inputDF).transform(inputDF)
  }

  def main(args: Array[String]): Unit = {

    val trainingSetPath = args(0)

    val allIncomeRecords = parseTrainingSet(trainingSetPath)

    val sparkSession = SparkSession.builder().getOrCreate()

    import sparkSession.implicits._

    val transformedDF = featureTransformation(allIncomeRecords.toDF())

    modelTuning(transformedDF).show()
  }
}
