package me.codingcat.titanic

import scala.collection.mutable

import me.codingcat.bicycle.RMLSEEval
import ml.dmlc.xgboost4j.scala.DMatrix
import ml.dmlc.xgboost4j.scala.spark.{XGBoostEstimator, XGBoost}
import org.apache.spark.SparkContext
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, RegressionEvaluator}
import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator}
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{Row, SparkSession}

object Main {

  def main(args: Array[String]): Unit = {
    val sc = new SparkContext()
    val trainingPath = args(0)
    val testPath = args(1)
    val iterations = args(2).toInt
    val featureGenerator = new OneHotAndNormalFeatureGenerator(sc)
    val trainingList = featureGenerator.generateLabeledPointRDD(trainingPath, containsGroundTruth = true)
    val sparkSession = SparkSession.builder().getOrCreate()
    import sparkSession.implicits._
    val trainingDS = trainingList.toDS()

    val params = new mutable.HashMap[String, Any]()
    params += "eta" -> 0.1
    params += "max_depth" -> 6
    params += "silent" -> 0
    params += "objective" -> "binary:logitraw"

    val xgbEstimator = new XGBoostEstimator(params.toMap).setFeaturesCol("features").
      setLabelCol("label")

    val paramGrid = new ParamGridBuilder()
      .addGrid(xgbEstimator.round, Array(20, 50))
      .addGrid(xgbEstimator.eta, Array(0.1, 0.4))
      .build()

    val cv = new CrossValidator()
      .setEstimator(xgbEstimator)
      .setEvaluator(new BinaryClassificationEvaluator().setLabelCol("label").
        setRawPredictionCol("probabilities"))
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(3)  // Use 3+ in practice
    val model = cv.fit(trainingDS)



    /*
    val testDMatrix = featureGenerator.generateDMatrix(testPath, containsGroundTruth = true)
    val evalMatries = new Array[DMatrix](1)
    evalMatries(0) = testDMatrix
    val evalMatriesNames = new Array[String](1)
    evalMatriesNames(0) = "test"
    println(xgBooster.booster.evalSet(evalMatries, evalMatriesNames, iterations))

    println("===========")
    println(xgBooster.booster.getModelDump().toList)
    */
  }

}
