package me.codingcat.titanic

import scala.collection.mutable

import me.codingcat.bicycle.RMLSEEval
import ml.dmlc.xgboost4j.scala.DMatrix
import ml.dmlc.xgboost4j.scala.spark.XGBoost
import org.apache.spark.SparkContext

object Main {

  def main(args: Array[String]): Unit = {
    val sc = new SparkContext()
    val trainingPath = args(0)
    val testPath = args(1)
    val iterations = args(2).toInt
    val featureGenerator = new OneHotAndNormalFeatureGenerator(sc)
    val trainingRDD = featureGenerator.generateLabeledPointRDD(trainingPath)

    val params = new mutable.HashMap[String, Any]()
    params += "eta" -> 0.1
    params += "max_depth" -> 6
    params += "silent" -> 0
    params += "ntreelimit" -> 1000
    params += "objective" -> "binary:logistic"

    val xgBooster = XGBoost.train(trainingRDD, params.toMap, round = iterations, nWorkers = 1,
      useExternalMemory = false)

    val testDMatrix = featureGenerator.generateDMatrix(testPath, containsGroundTruth = true)
    val evalMatries = new Array[DMatrix](1)
    evalMatries(0) = testDMatrix
    val evalMatriesNames = new Array[String](1)
    evalMatriesNames(0) = "test"
    println(xgBooster.booster.evalSet(evalMatries, evalMatriesNames, iterations))
  }

}
