package edu.cornell.cs.nlp.spf.scalalearn.situated.perceptron

import edu.cornell.cs.nlp.spf.base.hashvector.HashVectorFactory
import edu.cornell.cs.nlp.spf.ccg.categories.ICategoryServices
import edu.cornell.cs.nlp.spf.data.ILabeledDataItem
import edu.cornell.cs.nlp.spf.data.collection.IDataCollection
import edu.cornell.cs.nlp.spf.data.sentence.Sentence
import edu.cornell.cs.nlp.spf.data.situated.ISituatedDataItem
import edu.cornell.cs.nlp.spf.data.utils.IValidator
import edu.cornell.cs.nlp.spf.explat.resources.IResourceObjectCreator
import edu.cornell.cs.nlp.spf.explat.resources.usage.ResourceUsage
import edu.cornell.cs.nlp.spf.explat.{IResourceRepository, ParameterizedExperiment}
import edu.cornell.cs.nlp.spf.genlex.ccg.ILexiconGenerator
import edu.cornell.cs.nlp.spf.learn.situated.AbstractSituatedLearner
import edu.cornell.cs.nlp.spf.parser.IParserOutput
import edu.cornell.cs.nlp.spf.parser.ccg.model.IDataItemModel
import edu.cornell.cs.nlp.spf.parser.joint.model.{IJointDataItemModel, IJointModelImmutable, JointModel}
import edu.cornell.cs.nlp.spf.parser.joint.{IJointDerivation, IJointOutput, IJointOutputLogger, IJointParser}
import edu.cornell.cs.nlp.utils.log.{ILogger, LoggerFactory}

import scala.collection.JavaConverters._

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

class SituatedValidationPerceptronScala[SAMPLE <: ISituatedDataItem[Sentence, _], MR, ESTEP, ERESULT, DI <: ILabeledDataItem[SAMPLE, _]]
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
  type ParsesJava = java.util.List[JointDerivation]
  type JModel = JointModel[SAMPLE, MR, ESTEP]

  val log = LoggerFactory.create(classOf[SituatedValidationPerceptronScala[SAMPLE,MR, ESTEP, ERESULT, DI]])

  private def constructUpdate(violatingValidParses: Seq[JointDerivation], violatingInvalidParses: Seq[JointDerivation], model: JModel) = {

    // Create the parameter update
    val update = HashVectorFactory.create()

    // Get the update for valid violating samples
    violatingValidParses.foreach(_.getMeanMaxFeatures.addTimesInto(1.0 / violatingValidParses.size, update))

    // Get the update for the invalid violating samples
    violatingInvalidParses.foreach(_.getMeanMaxFeatures.addTimesInto(1.0 / violatingInvalidParses.size, update))

    // Prune small entries from the update
    update.dropNoise()

    // Validate the update
    if (!model.isValidWeightVector(update)) throw new IllegalStateException(s"invalid update: $update")

    update
  }

  private def createValidInvalidSets(dataItem: DI,
                                     parses: java.util.Collection[_ <: JointDerivation]) = {
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
    (valids, invalids)
  }

  private def marginViolatingSets(model: JModel, validParses: Seq[JointDerivation], invalidParses: Seq[JointDerivation]) = {
    val valids = validParses
    val invalids = invalidParses.map((_, false))

    val violatingParses = valids.foldLeft((List.empty[JointDerivation], invalids)) {
      case ((violatingValidsInner, invalidsWithFlags), validParse) =>
        var isValidViolating = false
        val invalidsWithUpdatedFlags = invalidsWithFlags.map { case (invalid, flag) =>
          val featureDelta = validParse.getMeanMaxFeatures.addTimes(-1.0, invalid.getMeanMaxFeatures)
          val deltaScore = model score featureDelta
          val threshold = margin * featureDelta.l1Norm()
          if (deltaScore < threshold) {isValidViolating = true; (invalid, true)}
          else (invalid, flag)
        }
        val violatingValidsUpdated =
          if (isValidViolating) violatingValidsInner :+ validParse
          else violatingValidsInner
        (violatingValidsUpdated, invalidsWithUpdatedFlags)
    }

    val (violatingValids, violatingInvalidsCandidates) = violatingParses
    val violatingInvalids = violatingInvalidsCandidates.filter(_._2).map(_._1)

    (violatingValids.toList, violatingInvalids.toList)
  }

  override protected def parameterUpdate(dataItem: DI,
                                dataItemModel: IJointDataItemModel[MR, ESTEP],
                                model: JModel,
                                dataItemNumber: Int,
                                epochNumber: Int): Unit = {

    // Parse with current model
    val parserOutput = parser.parse(dataItem.getSample, dataItemModel)
    stats.mean("model parse", parserOutput.getInferenceTime/ 1000.0, "sec")
    parserOutputLogger.log(parserOutput, dataItemModel, s"$dataItemNumber-update")
    val modelParses = parserOutput.getDerivations()
    val bestModelParses = parserOutput.getMaxDerivations()

    if (modelParses.isEmpty) {
      // Skip the rest of the process if no complete parses available
      log.info(s"No parses for: $dataItem")
      log.info("Skipping parameter update")
      return
    }

    log.info(s"Created ${modelParses.size()} model parses for training sample")
    log.info(s"Model parsing time: ${parserOutput.getInferenceTime / 1000.0}sec")
    log.info(s"Output is ${if (parserOutput.isExact) "exact" else "approximate"}")

    // Split all parses to valid and invalid sets
    val (validParses, invalidParses) = createValidInvalidSets(dataItem, modelParses)
    log.info(s"${validParses.size} valid parses, ${invalidParses.size} invalid parses")
    log.info("Valid parses:")
    validParses.foreach(logParse(dataItem, _, true, true, dataItemModel))

    // Record if the best is the gold standard, if such debug information is available.
    if (bestModelParses.size() == 1 && isGoldDebugCorrect(dataItem, bestModelParses.get(0).getResult)) {
      stats.appendSampleStat(dataItemNumber, epochNumber, AbstractSituatedLearner.GOLD_LF_IS_MAX)
    } else if (validParses.nonEmpty) {
      // Record if a valid parse was found.
      stats.appendSampleStat(dataItemNumber, epochNumber, AbstractSituatedLearner.HAS_VALID_LF)
    }

    if (validParses.nonEmpty) {
      stats.count("valid", epochNumber)
    }

    // Skip update if there are no valid or invalid parses
    if (validParses.isEmpty || invalidParses.isEmpty) {
      log.info("No valid/invalid parses -- skipping")
      return
    }

    // Construct margin violating sets
    val (violatingValidParses, violatingInvalidParses) = marginViolatingSets(model, validParses, invalidParses)
    log.info(s"${violatingValidParses.size} violating valid parses, ${violatingInvalidParses.size} violating invalid parses")
    if (violatingValidParses.isEmpty) {
      log.info("There are no violating valid/invalid parses -- skipping")
      return
    }
    log.info("Violating valid parses: ")
    violatingValidParses.foreach(logParse(dataItem, _, true, true, dataItemModel))

    log.info("Violating invalid parses: ")
    violatingInvalidParses.foreach(logParse(dataItem, _, false, true, dataItemModel))

    // Construct weight update vector
    val update = constructUpdate(violatingValidParses, violatingInvalidParses, model)

    // Update the parameters vector
    log.info(s"Update: $update")
    update.addTimesInto(1.0, model.getTheta)
    stats.appendSampleStat(dataItemNumber, epochNumber, AbstractSituatedLearner.TRIGGERED_UPDATE)
    stats.count("update", epochNumber)
  }

  override protected def validate(dataItem: DI, hypothesis: ERESULT): Boolean = validator.isValid(dataItem, hypothesis)

}

object SituatedValidationPerceptronScala {

  case class Builder[SAMPLE <: ISituatedDataItem[Sentence, _],
                    MR,
                    ESTEP,
                    ERESULT,
                    DI <: ILabeledDataItem[SAMPLE, _]] private(categoryService: ICategoryServices[MR],
                                                               genlex: ILexiconGenerator[DI, MR, IJointModelImmutable[SAMPLE, MR, ESTEP]],
                                                               hardUpdates: Boolean = false,
                                                               lexiconGenerationBeamSize: Int = 20,
                                                               margin: Double = 1.0,
                                                               maxSentenceLength: Int = Integer.MAX_VALUE,
                                                               numIterations: Int = 4,
                                                               parser: IJointParser[SAMPLE, MR, ESTEP, ERESULT],
                                                               trainingData: IDataCollection[DI],
                                                               trainingDataDebug: java.util.Map[DI, edu.cornell.cs.nlp.utils.composites.Pair[MR, ERESULT]] = new java.util.HashMap(),
                                                               validator: IValidator[DI, ERESULT]) {

    /**
      * categoryServices Required for lexical induction.

      * genlex GENLEX procedure. If 'null' skip lexical induction.

      * Use hard updates. Meaning: consider only highest-scored valid parses
      * for parameter updates, instead of all valid parses.

      * Beam size to use when doing loss sensitive pruning with generated
      * lexicon.
      *
      * Margin to scale the relative loss function

      * Max sentence length. Sentence longer than this value will be skipped
      * during training
      *
      *
      * Number of training iterations
      */

    private val parserOutputLogger = new IJointOutputLogger[MR, ESTEP, ERESULT]() {
      // Stub, do nothing.
      override def log(output: IJointOutput[MR, ERESULT], dataItemModel: IJointDataItemModel[MR, ESTEP], tag: String): Unit = ()

      override def log(output: IParserOutput[MR], dataItemModel: IDataItemModel[MR], tag: String): Unit = ()
    }

    object IJointOutputLogger{
      private val serialVersionUID: Long = 4342845964338126692L
    }

    /**
      * Training data
      *
      * Mapping a subset of training samples into their gold label for debug.
      */

    def build(): SituatedValidationPerceptronScala[SAMPLE, MR, ESTEP, ERESULT, DI] =
      new SituatedValidationPerceptronScala(
        numIterations, margin, trainingData, trainingDataDebug,
        maxSentenceLength, lexiconGenerationBeamSize, parser,
        hardUpdates, parserOutputLogger, validator,
        categoryServices, genlex)
  }


  case class Creator[SAMPLE <: ISituatedDataItem[Sentence, _], MR, ESTEP, ERESULT, DI <: ILabeledDataItem [SAMPLE, _]]()
   extends IResourceObjectCreator[SituatedValidationPerceptronScala[SAMPLE, MR, ESTEP, ERESULT, DI]] {

    override def create(params: ParameterizedExperiment#Parameters, repo: IResourceRepository): SituatedValidationPerceptronScala[SAMPLE, MR, ESTEP, ERESULT, DI] = {


      val trainingData = repo.get(params.get("data"))

      val hardUpdates = "true" == (params get "hard")

      val parserOutputLogger =
        if (params contains "parseLogger")
          repo.get(params get "parseLogger").asInstanceOf[IJointOutputLogger[MR, ESTEP, ERESULT]]
        else null

      val (genlex, categoryService) =
        if (params contains "genlex")
          (repo.get(params get "genlex").asInstanceOf[ILexiconGenerator[DI, MR, IJointModelImmutable[SAMPLE, MR, ESTEP]]],
            repo.get(ParameterizedExperiment.CATEGORY_SERVICES_RESOURCE).asInstanceOf[ICategoryServices[MR]])
        else (null, null)

      val lexiconGenerationBeamSize =
        if (params contains "genlexbeam") params.get("genlexbeam").asInstanceOf[Int]
        else 20

      val margin =
        if (params.contains("margin")) params.get("margin").asInstanceOf[Double]
        else 1.0

      val maxSentenceLength =
        if (params contains "maxSentenceLength") params.get("maxSentenceLength").asInstanceOf[Int]
        else Integer.MAX_VALUE

      val numTrainingIterations =
        if (params contains "iter") params.get("iter").asInstanceOf[Int]
        else 4

      val parses = repo.get(ParameterizedExperiment.PARSER_RESOURCE).asInstanceOf[IJointParser[SAMPLE, MR, ESTEP, ERESULT]]

      val validator = repo.get(params.get("validator")).asInstanceOf[IValidator[DI, ERESULT]]

      Builder(
        categoryService = categoryService,
        genlex = genlex,
        hardUpdates = hardUpdates,
        lexiconGenerationBeamSize = lexiconGenerationBeamSize,
        margin = margin,
        maxSentenceLength = maxSentenceLength,
        numIterations = numTrainingIterations,
        parser = parses,
        trainingData = trainingData,
        trainingDataDebug = new java.util.HashMap[DI, edu.cornell.cs.nlp.utils.composites.Pair[MR, ERESULT]](),
        validator = validator
      ).build()

    }

    /**
      * The resource type.
      *
      * @return
      */
    override def `type`(): String = ???

    /**
      * Return a usage objects describing the resource created and how it can be
      * created.
      *
      * @return
      */
    override def usage(): ResourceUsage = ???
  }
}
