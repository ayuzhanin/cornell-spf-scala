package edu.cornell.cs.nlp.spf.scalalearn.simple

import edu.cornell.cs.nlp.spf.base.hashvector.HashVectorFactory
import edu.cornell.cs.nlp.spf.data.collection.IDataCollection
import edu.cornell.cs.nlp.spf.data.sentence.Sentence
import edu.cornell.cs.nlp.spf.data.singlesentence.SingleSentence
import edu.cornell.cs.nlp.spf.learn.ILearner
import edu.cornell.cs.nlp.spf.mr.lambda.LogicalExpression
import edu.cornell.cs.nlp.spf.parser.IParser
import edu.cornell.cs.nlp.spf.parser.ccg.model.Model
import edu.cornell.cs.nlp.utils.log.{ILogger, LoggerFactory}

import scala.collection.JavaConverters._

/**
  * Created by a.ayuzhanin on 19/03/2017.
  */
class SimplePerceptronScalaImpl(numIterations: Integer,
                                trainingData: IDataCollection[SingleSentence],
                                parser: IParser[Sentence, LogicalExpression])
  extends ILearner[Sentence, SingleSentence, Model[Sentence, LogicalExpression]] {

  val log: ILogger = LoggerFactory.create(classOf[SimplePerceptronScalaImpl])

  import SimplePerceptronScalaImpl.toImmutableSeq

  override def train(model: Model[Sentence, LogicalExpression]): Unit = {

    for (iterationIndex <- 0 to numIterations) {

      log.info("=========================")
      log.info(s"Training iteration $iterationIndex")
      log.info("=========================")

      trainingData.asScala.zipWithIndex.foreach { case (singleSentence, itemCounter) =>
        val startTime = System.currentTimeMillis
        log.info(s"$itemCounter : ================== [$iterationIndex]")
        log.info(s"Sample type: ${singleSentence.getClass.getSimpleName}")
        log.info(s"$singleSentence")

        val dataItemModel = model.createDataItemModel(singleSentence.getSample)
        val parserOutput = parser.parse(singleSentence.getSample, dataItemModel)
        val bestParses = toImmutableSeq(parserOutput.getBestDerivations.asScala)

        // Correct parse
        val correctParses = toImmutableSeq {
          parserOutput
            .getMaxDerivations(_.getSemantics == singleSentence.getLabel)
            .asScala
        }

        // Violating parses
        val violatingBadParses = bestParses.filter(parse => !singleSentence.isCorrect(parse.getSemantics))
        violatingBadParses.foreach(parse => log.info(s"Bad parse: ${parse.getSemantics}"))

        // Case we have bad best parses and a correct parse, need to update.
        if (violatingBadParses.nonEmpty && correctParses.nonEmpty) {
          // Create the parameter update
          val update = HashVectorFactory.create()
          // Positive update
          correctParses.foreach(_.getAverageMaxFeatureVector.addTimesInto(1.0 / correctParses.size, update))
          // Negative update
          violatingBadParses.foreach(_.getAverageMaxFeatureVector.addTimesInto(-1.0 / violatingBadParses.size, update))
          update.dropNoise()
          if (!model.isValidWeightVector(update)) new IllegalStateException(s"invalid update $update")
          log.info(s"Update: $update")
          update.addTimesInto(1.0, model.getTheta)
        }
        else if (correctParses.isEmpty) log.info("No correct parses. No update.")
        else log.info("Correct. No update.")
        log.info("Sample processing time %.4f", (System.currentTimeMillis - startTime) / 1000.0)
      }
    }
  }

}

object SimplePerceptronScalaImpl {
  def toImmutableSeq[T](sq: collection.mutable.Seq[T]): collection.immutable.Seq[T] =
    collection.immutable.Seq[T](sq: _*)
}