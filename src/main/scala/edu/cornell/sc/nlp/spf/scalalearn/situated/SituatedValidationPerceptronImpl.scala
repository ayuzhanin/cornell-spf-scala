package edu.cornell.sc.nlp.spf.scalalearn.situated

import edu.cornell.cs.nlp.spf.base.hashvector.{HashVectorFactory, IHashVector}
import edu.cornell.cs.nlp.spf.ccg.categories.ICategoryServices
import edu.cornell.cs.nlp.spf.data.ILabeledDataItem
import edu.cornell.cs.nlp.spf.data.collection.IDataCollection
import edu.cornell.cs.nlp.spf.data.sentence.Sentence
import edu.cornell.cs.nlp.spf.data.situated.ISituatedDataItem
import edu.cornell.cs.nlp.spf.data.utils.IValidator
import edu.cornell.cs.nlp.spf.genlex.ccg.ILexiconGenerator
import edu.cornell.cs.nlp.spf.learn.situated.AbstractSituatedLearner
import edu.cornell.cs.nlp.spf.parser.joint.model.{IJointModelImmutable, JointModel}
import edu.cornell.cs.nlp.spf.parser.joint.{IJointDerivation, IJointOutputLogger, IJointParser}

import scala.collection.JavaConverters._
/**
  * Created by a.ayuzhanin on 19/03/2017.
  */
/**
  * Situated validation-based perceptron learner. See Artzi and Zettlemoyer 2013
  * for detailed description.
  * <p>
  * Parameter update step inspired by: Natasha Singh-Miller and Michael Collins.
  * 2007. Trigger-based Language Modeling using a Loss-sensitive Perceptron
  * Algorithm. In proceedings of ICASSP 2007.
  * </p>
  *
  * @author Yoav Artzi
  * @param < STATE>
  *          Type of initial state.
  * @param < MR>
  *          Meaning representation type.
  * @param < ESTEP>
  *          Type of execution step.
  * @param < ERESULT>
  *          Type of execution result.
  * @param < DI>
  *          Training data item.
  */

class SituatedValidationPerceptronImpl[SAMPLE <: ISituatedDataItem[Sentence, _], MR, ESTEP, ERESULT, DI <: ILabeledDataItem[SAMPLE, _]]
                                        (numIterations: Int,
                                         margin: Double,
                                         trainingData: IDataCollection[DI],
                                         trainingDataDebug: java.util.Map[DI, edu.cornell.cs.nlp.utils.composites.Pair[MR, ERESULT]],
                                         maxSentenceLength: Int,
                                         lexiconGenerationBeamSize: Int,
                                         parser: IJointParser[SAMPLE, MR, ESTEP, ERESULT],
                                         hardUpdates: Boolean,
                                         parserOutputLogger: IJointOutputLogger[MR, ESTEP, ERESULT],
                                         validator: IValidator[DI, ERESULT],
                                         categoryServices: ICategoryServices[MR],
                                         genlex: ILexiconGenerator[DI, MR, IJointModelImmutable[SAMPLE, MR, ESTEP]])
                                          extends AbstractSituatedLearner[SAMPLE, MR, ESTEP, ERESULT, DI] (numIterations,
                                            trainingData,
                                            trainingDataDebug,
                                            maxSentenceLength,
                                            lexiconGenerationBeamSize,
                                            parser,
                                            parserOutputLogger,
                                            categoryServices,
                                            genlex) {

  type JointDerivation = IJointDerivation[MR, ERESULT]
  type Parses = java.util.List[JointDerivation]
  type ParsesPair = edu.cornell.cs.nlp.utils.composites.Pair[Parses, Parses]
  type JModel = JointModel[SAMPLE, MR, ESTEP]

  import edu.cornell.sc.nlp.spf.scalalearn.simple.SimplePerceptronScalaImpl.toImmutableSeq

  private def constructUpdate(violatingValidParses: Parses, violatingInvalidParses: Parses, model: JModel): IHashVector = {

    // Create the parameter update
    val update = HashVectorFactory.create()

    // Get the update for valid violating samples
    violatingValidParses.asScala.foreach(_.getMeanMaxFeatures.addTimesInto(1.0 / violatingValidParses.size(), update))

    // Get the update for the invalid violating samples
    violatingInvalidParses.asScala.foreach(_.getMeanMaxFeatures.addTimesInto(1.0 / violatingInvalidParses.size(), update))

    // Prune small entries from the update
    update.dropNoise()

    // Validate the update
    if (!model.isValidWeightVector(update)) throw new IllegalStateException(s"invalid update: $update")

    update
  }

  private def createValidInvalidSets(dataItem: DI,
                                     parses: java.util.Collection[_ <: JointDerivation]): ParsesPair = {
    val (valids, invalids, _) = parses.asScala.foldLeft((Seq.empty[JointDerivation], Seq.empty[JointDerivation], -java.lang.Double.MAX_VALUE)) {
      case ((validParses, invalidParses, validScore), parse) =>
        if (validate(dataItem, parse.getResult))
        // Case using hard updates, only keep the highest scored
        // valid ones
          if (hardUpdates)
            if (parse.getViterbiScore > validScore)
              (Seq(parse), invalidParses, parse.getViterbiScore)
            else if (parse.getViterbiScore == validScore)
              (validParses :+ parse, invalidParses, validScore)
            else (validParses, invalidParses, validScore)
          else (validParses :+ parse, invalidParses, validScore)
        else (validParses, invalidParses :+ parse, validScore)
    }
    edu.cornell.cs.nlp.utils.composites.Pair.of(valids.asJava, invalids.asJava)
  }

  private def marginViolatingSets(model: JModel, validParses: Parses, invalidParses: Parses): ParsesPair = {
    val validParsesWithIndex = toImmutableSeq(validParses.asScala).zipWithIndex.map { case (elem, index) => (elem, false, index) }
    val invalidParsesWithIndex = toImmutableSeq(invalidParses.asScala).zipWithIndex.map { case (elem, index) => (elem, false, index) }

  }
}
