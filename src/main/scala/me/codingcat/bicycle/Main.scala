package me.codingcat.bicycle

import scala.collection.mutable

import ml.dmlc.xgboost4j.scala.DMatrix
import ml.dmlc.xgboost4j.scala.spark.XGBoost
import org.apache.spark.SparkContext

object Main {

  var sc: SparkContext = null

  def main(args: Array[String]): Unit = {
    sc = new SparkContext()
    val trainingPath = args(0)
    val testPath = args(1)
    val iterations = args(2).toInt
    val featureGenerator = new FirstFeatureGenerator(sc)
    val trainingRDD = featureGenerator.generateFeatureRDD(trainingPath)

    val params = new mutable.HashMap[String, Any]()
    params += "eta" -> 0.1
    params += "max_depth" -> 6
    params += "silent" -> 0
    params += "ntreelimit" -> 1000
    params += "objective" -> "reg:linear"

    val testMatrix = new DMatrix(testPath)
    val xgBooster = XGBoost.train(trainingRDD, params.toMap, round = iterations, nWorkers = 1,
      useExternalMemory = true)
    xgBooster.predict(testMatrix)
  }
}
