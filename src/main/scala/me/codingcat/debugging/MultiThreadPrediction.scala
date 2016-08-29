package me.codingcat.debugging

import java.util

import ml.dmlc.xgboost4j.java.{DMatrix, XGBoostError, XGBoost}

object MultiThreadPrediction {
  def main(args: Array[String]): Unit = {
    val model = XGBoost.loadModel("/Users/nanzhu/Downloads/001.model")
    for (i <- 0 until 10) {
      new Thread(new Runnable() {
        def run() {
          try {
            val ms = System.currentTimeMillis()
            val predicts2 = model.predict(new DMatrix("/Users/nanzhu/Downloads/test_200.data"))
            System.out.println("##############" +
              Thread.currentThread().getName() +
              " " + ((System.currentTimeMillis() - ms)/1000.0));
          } catch {
            case e: XGBoostError =>
              e.printStackTrace();
          }
        }
      }).start()
    }
  }
}
