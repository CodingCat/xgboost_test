package me.codingcat.agaricus

import java.io.File

import scala.collection.mutable.ListBuffer
import scala.io.Source

import ml.dmlc.xgboost4j.scala.spark._
import ml.dmlc.xgboost4j.scala.{XGBoost => ScalaXGBoost, DMatrix}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.{DenseVector, Vector => SparkVector}
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.SparkSession

object Agaricus {

  protected def loadLabelPoints(filePath: String): List[LabeledPoint] = {
    val file = Source.fromFile(new File(filePath))
    val sampleList = new ListBuffer[LabeledPoint]
    for (sample <- file.getLines()) {
      sampleList += fromSVMStringToLabeledPoint(sample)
    }
    sampleList.toList
  }

  protected def loadLabelAndVector(filePath: String): List[(Double, SparkVector)] = {
    val file = Source.fromFile(new File(filePath))
    val sampleList = new ListBuffer[(Double, SparkVector)]
    for (sample <- file.getLines()) {
      sampleList += fromSVMStringToLabelAndVector(sample)
    }
    sampleList.toList
  }

  protected def fromSVMStringToLabelAndVector(line: String): (Double, SparkVector) = {
    val labelAndFeatures = line.split(" ")
    val label = labelAndFeatures(0).toDouble
    val features = labelAndFeatures.tail
    val denseFeature = new Array[Double](129)
    for (feature <- features) {
      val idAndValue = feature.split(":")
      denseFeature(idAndValue(0).toInt) = idAndValue(1).toDouble
    }
    (label, new DenseVector(denseFeature))
  }

  protected def fromSVMStringToLabeledPoint(line: String): LabeledPoint = {
    val (label, sv) = fromSVMStringToLabelAndVector(line)
    LabeledPoint(label, sv)
  }

  def main(args: Array[String]): Unit = {
    val trainingPath = args(0)
    val testPath = args(1)

    val sparkSession = SparkSession.builder().getOrCreate()
    import sparkSession.implicits._
    /*
    val trainingRDD = MLUtils.loadLibSVMFile(sparkSession.sparkContext, trainingPath).map(
      lp => new LabeledPoint(lp.label, lp.features.asML)
    )
    val testRDD = MLUtils.loadLibSVMFile(sparkSession.sparkContext, testPath).map(
      lp => new LabeledPoint(lp.label, lp.features.asML)
    )*/
    val trainingList = loadLabelPoints(trainingPath)
    val testList = loadLabelPoints(testPath)
    val trainingRDD = sparkSession.sparkContext.parallelize(trainingList, 4)
    val testRDD = sparkSession.sparkContext.parallelize(testList, 4)

    val trainingDS = trainingRDD.toDF()
    val testDS = testRDD.toDF()
/*
    val estimator = new XGBoostEstimator(Map[String, Any](
      "num_round" -> 2, "objective" -> "binary:logistic", "nworkers" -> 4))

    val paramGrid = new ParamGridBuilder().
      addGrid(estimator.eta, Array(0.2, 0.6)).
      addGrid(estimator.maxDepth, Array(5, 6)).build()

    val trainSplitter = new TrainValidationSplit().setEstimator(estimator).
      setEstimatorParamMaps(paramGrid).setEvaluator(
      new BinaryClassificationEvaluator().setRawPredictionCol("probabilities")).setTrainRatio(0.9)

    val bestModel = trainSplitter.fit(trainingDS)
    bestModel.transform(testDS).show()
    */
    // println(model.booster.evalSet(Array(new DMatrix(testPath)), Array("test"), -1))
    /*
    val trainMatrix = new DMatrix(trainingList.iterator)

    val xgboostModel = ScalaXGBoost.train(trainMatrix, map, 5,
      watches = Map[String, DMatrix]("train" -> trainMatrix))*/
    // val predResultFromSeq = xgboostModel.predict(testMatrix)


    val map = Map("eta" -> "1", "max_depth" -> "6", "silent" -> "1",
      "objective" -> "binary:logistic")

    val model = XGBoost.trainWithDataFrame(trainingDS, map, nWorkers = 4, round = 5)
    import DataUtils._
    model.transform(testDS).show(2000)
  }

}
