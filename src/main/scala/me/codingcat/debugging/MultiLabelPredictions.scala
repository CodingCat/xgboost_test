package me.codingcat.debugging

import ml.dmlc.xgboost4j.scala.{XGBoost, DMatrix}

object MultiLabelPredictions {
  def main(args: Array[String]): Unit = {
    val trainData = new DMatrix(args(0))
    // define parameters
    val paramMap = List(
      "num_class" -> 10,
      "silent" -> 1,
      "eval_metric" -> args(1),
      "eta" -> 0.3,
      "max_depth" ->3,
      "objective" ->"multi:softprob"
      // "objective" -> "binary:logistic"
    ).toMap
    // number of iterations
    val round = 10
    val watch = Map( "train" -> trainData)
    // train the model
    val model = XGBoost.train(trainData, paramMap, round, watch)

  }
}
