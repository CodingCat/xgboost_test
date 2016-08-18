package me.codingcat.bicycle

import ml.dmlc.xgboost4j.LabeledPoint
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

object Main {

  var sc: SparkContext = null

  private def generateTrainingRDD(trainingPath: String): RDD[LabeledPoint] = {
    val textFileRDD = sc.textFile(trainingPath)
    textFileRDD.map{
      line =>

    }
  }

  def main(args: Array[String]): Unit = {
    sc = new SparkContext()
    val trainingPath = args(0)
    val testPath = args(1)


  }
}
