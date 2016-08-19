package me.codingcat.bicycle

import ml.dmlc.xgboost4j.scala.{DMatrix, EvalTrait}

class RMLSEEval extends EvalTrait with Serializable {
  override def getMetric(): String = "RMLSEEval"

  override def eval(predicts: Array[Array[Float]], dmat: DMatrix): Float = {
    val labels = dmat.getLabel
    var sum = 0.0f
    for (i <- predicts.indices) {
      val pred = if (predicts(i)(0) < 0) 0.0f else predicts(i)(0)
      val r = math.pow(math.log(pred + 1) - math.log(labels(i) + 1), 2)
      sum += r.toFloat
    }
    math.sqrt(sum / labels.length).toFloat
  }
}
