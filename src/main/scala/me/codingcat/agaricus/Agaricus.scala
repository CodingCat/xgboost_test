package me.codingcat.agaricus

import ml.dmlc.xgboost4j.scala.DMatrix
import ml.dmlc.xgboost4j.scala.spark.{XGBoostClassificationModel, XGBoost, XGBoostEstimator}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.SparkSession

object Agaricus {

  def main(args: Array[String]): Unit = {
    val trainingPath = args(0)
    val testPath = args(1)

    val sparkSession = SparkSession.builder().getOrCreate()
    import sparkSession.implicits._
    val trainingRDD = MLUtils.loadLibSVMFile(sparkSession.sparkContext, trainingPath).map(
      lp => new LabeledPoint(lp.label, lp.features.asML)
    )
    val testRDD = MLUtils.loadLibSVMFile(sparkSession.sparkContext, testPath).map(
      lp => new LabeledPoint(lp.label, lp.features.asML)
    )
    val trainingDS = trainingRDD.toDF()
    val testDS = testRDD.toDF()

    val estimator = new XGBoostEstimator(Map[String, Any](
      "num_round" -> 2, "objective" -> "binary:logistic", "nworkers" -> 4))

    val paramGrid = new ParamGridBuilder().
      addGrid(estimator.eta, Array(0.2, 0.6)).
      addGrid(estimator.maxDepth, Array(5, 6)).build()

    val trainSplitter = new TrainValidationSplit().setEstimator(estimator).
      setEstimatorParamMaps(paramGrid).setEvaluator(
      new BinaryClassificationEvaluator().setRawPredictionCol("probabilities")).setTrainRatio(0.9)

    val bestModel = trainSplitter.fit(trainingDS)
    bestModel.transform(testDS).show(100)


    val model = XGBoost.train(trainingRDD, Map[String, Any](
      "num_round" -> 100, "objective" -> "binary:logistic", "nworkers" -> 4, "eta" -> 1,
      "max_depth" -> 3, "gamma" -> 1), nWorkers = 1, round = 2)

    println(model.booster.evalSet(Array(new DMatrix(testPath)), Array("test"), -1))

  }

}
