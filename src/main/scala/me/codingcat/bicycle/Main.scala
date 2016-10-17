package me.codingcat.bicycle

import scala.collection.mutable

import ml.dmlc.xgboost4j.scala.spark.XGBoost
import org.apache.spark.SparkContext

object Main {

  def main(args: Array[String]): Unit = {
    val sc = new SparkContext()
    val trainingPath = args(0)
    val testPath = args(1)
    val iterations = args(2).toInt
    val featureGenerator = new FirstFeatureGenerator(sc)
    val trainingRDD = featureGenerator.generateLabeledPointRDD(trainingPath)

    val params = new mutable.HashMap[String, Any]()
    params += "eta" -> 0.1
    params += "max_depth" -> 6
    params += "silent" -> 0
    params += "ntreelimit" -> 1000
    params += "objective" -> "reg:linear"
    params += "subsample" -> 0.8
    params += "round" -> 100

    val testMatrix = featureGenerator.generateDMatrix(testPath,
      containsGroundTruth =  true)
    val xgBooster = XGBoost.trainWithRDD(trainingRDD, params.toMap, round = iterations, nWorkers = 4,
      useExternalMemory = true, eval = new RMLSEEval)
    // val predictiveResults = xgBooster.predict(testMatrix)
    // val evalMatries = new Array[DMatrix](1)
    // evalMatries(0) = testMatrix
    // val evalMatriesName = new Array[String](1)
    // evalMatriesName(0) = "test"
    //println(xgBooster.predict(testMatrix).toList.map(_.toList))
    //println(xgBooster.booster.evalSet(evalMatries, evalMatriesName, new RMLSEEval))
    val testsetRDD = featureGenerator.generateLabeledPointRDD(testPath, containsGroundTruth = true)
    println(xgBooster.eval(testsetRDD, evalFunc = new RMLSEEval, evalName = "test",
      useExternalCache = false))
  }
}
