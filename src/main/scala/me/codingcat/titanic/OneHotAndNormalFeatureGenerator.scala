package me.codingcat.titanic

import scala.collection.mutable.ListBuffer
import scala.io.Source

import me.codingcat.base.BasicFeatureExtractor
import ml.dmlc.xgboost4j.scala.DMatrix
import org.apache.spark.SparkContext
import org.apache.spark.ml.linalg.{Vector => SparkVector, DenseVector}
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.rdd.RDD


case class PassengerInfo(pClass: Float, sex: String, age: Float, sibsp: Float, parch: Float,
                         ticket: String, fare: Float, cabin: String, embarked: String,
                         survived: Int)

case class TransformedPassengerInfo(
    pClass: Array[Float], sex: Array[Float], age: Float, sibsp: Float,
    parch: Float, ticketLength: Float, ticketStart: Float, fare: Float,
    cabin: Array[Float], Embarked: Array[Float])

class OneHotAndNormalFeatureGenerator(
    @transient val sc: SparkContext) {

  private def parseFileToRawPassengerInfo(filePath: String): List[PassengerInfo] = {
    val list = new ListBuffer[PassengerInfo]
    for (line <- Source.fromFile(filePath).getLines()) {
      val featureStringArray = line.split(",")
      val pClass = {if (featureStringArray(2).length > 0) featureStringArray(2).toFloat else -1.0f}
      val sex = featureStringArray(5)
      val age = {if (featureStringArray(6).length > 0) featureStringArray(6).toFloat else -1.0f}
      val sibsp = {if (featureStringArray(7).length > 0) featureStringArray(7).toFloat else -1.0f}
      val parch = {if (featureStringArray(8).length > 0) featureStringArray(8).toFloat else -1.0f}
      val ticket = featureStringArray(9)
      val fare = {if (featureStringArray(10).length > 0) featureStringArray(10).toFloat else -1.0f}
      val cabin = featureStringArray(11)
      val embarked = {
        var ret = ""
        if (featureStringArray.length >= 13) {
          ret = featureStringArray(12)
        }
        ret
      }
      val survived = featureStringArray(1).toInt
      list += PassengerInfo(pClass, sex, age, sibsp, parch, ticket, fare, cabin, embarked, survived)
    }
    list.toList
  }

  private def oneHotEncodingAndNormalization(passengers: List[PassengerInfo]):
      List[LabeledPoint] = {
    // one hot encoding
    val pClassAllVars = passengers.map(_.pClass).distinct.zipWithIndex.toMap
    val sexAllVars = passengers.map(_.sex).distinct.zipWithIndex.toMap
    val cabin = passengers.map(_.cabin).distinct.zipWithIndex.toMap
    val embarked = passengers.map(_.embarked).distinct.zipWithIndex.toMap
    // normalization
    // age
    val validAges = passengers.filter(_.age != -1.0f).map(_.age)
    val avrAge = validAges.sum / validAges.length
    val stdevAge = math.sqrt(validAges.map(age => math.pow(age - avrAge, 2)).sum / validAges.length)
      .toFloat
    // sibsp
    val validSibsp = passengers.filter(_.sibsp != -1.0f).map(_.sibsp)
    val avrSibsp = validSibsp.sum / validSibsp.length
    val stdevSibsp = math.sqrt(validSibsp.map(sibsp => math.pow(sibsp - avrSibsp, 2)).sum /
      validSibsp.length).toFloat
    // parch
    val validParch = passengers.filter(_.parch != -1.0f).map(_.parch)
    val avrParch = validParch.sum / validParch.length
    val stdevParch = math.sqrt(validParch.map(parch => math.pow(parch - avrParch, 2)).sum /
      validParch.length).toFloat
    // fare
    val validFare = passengers.filter(_.fare != -1.0f).map(_.fare)
    val avrFare = validFare.sum / validFare.length
    val stdevFare = math.sqrt(validFare.map(fare => math.pow(fare - avrFare, 2)).sum /
      validFare.length).toFloat
    // ignore ticket feature for now
    val labelAndfeatureArrays = passengers.map{
      passenger =>
        val classFeatureArray = new Array[Float](pClassAllVars.size)
        if (passenger.pClass != -1.0f) {
          classFeatureArray(pClassAllVars.get(passenger.pClass).get) = 1.0f
        }
        val sexFeatureArray = new Array[Float](sexAllVars.size)
        if (passenger.sex.length > 0) {
          sexFeatureArray(sexAllVars.get(passenger.sex).get) = 1.0f
        }
        val cabinFeatureArray = new Array[Float](cabin.size)
        if (passenger.cabin.length > 0) {
          cabinFeatureArray(cabin.get(passenger.cabin).get) = 1.0f
        }
        val embarkedFeatureArray = new Array[Float](embarked.size)
        if (passenger.embarked.length > 0) {
          embarkedFeatureArray(embarked.get(passenger.embarked).get) = 1.0f
        }
        val age = (passenger.age - avrAge) / stdevAge
        val sibsp = (passenger.sibsp - avrSibsp) / stdevSibsp
        val parch = (passenger.parch - avrParch) / stdevParch
        val fare = (passenger.fare - avrFare) / stdevFare
        val normalizedFeatureArray = new Array[Float](4)
        normalizedFeatureArray(0) = age
        normalizedFeatureArray(1) = sibsp
        normalizedFeatureArray(2) = parch
        normalizedFeatureArray(3) = fare
        (passenger.survived,
          classFeatureArray ++ sexFeatureArray ++ cabinFeatureArray ++ embarkedFeatureArray ++
          normalizedFeatureArray)
    }
    labelAndfeatureArrays.map{case (label, featureArray) =>
      LabeledPoint(label, new DenseVector(featureArray.map(_.toDouble)))
    }
  }

  def generateLabeledPointRDD(datasetPath: String, containsGroundTruth: Boolean):
      List[LabeledPoint] = {
    val rawFeatures = parseFileToRawPassengerInfo(datasetPath)
    oneHotEncodingAndNormalization(rawFeatures)
  }

  def generateDMatrix(datasetPath: String, containsGroundTruth: Boolean): DMatrix = {
    val rawFeatures = parseFileToRawPassengerInfo(datasetPath)
    val labeledPoints = oneHotEncodingAndNormalization(rawFeatures)
    import ml.dmlc.xgboost4j.scala.spark.DataUtils._
    new DMatrix(labeledPoints.iterator)
  }
}
