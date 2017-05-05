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
import edu.cornell.cs.nlp.spf.parser.IParserOutput
import edu.cornell.cs.nlp.spf.parser.ccg.model.IDataItemModel
import edu.cornell.cs.nlp.spf.parser.joint.model.{IJointDataItemModel, IJointModelImmutable}
import edu.cornell.cs.nlp.spf.parser.joint.{IJointOutput, IJointOutputLogger, IJointParser}
import edu.cornell.cs.nlp.spf.scalalearn.situated.AbstractSituatedLearnerScala
import edu.cornell.cs.nlp.utils.log.LoggerFactory

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
  * @tparam MR      Meaning representation type.
  * @tparam ESTEP   Type of execution step.
  * @tparam ERESULT Type of execution result.
  * @tparam DI      Training data item.
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
  extends AbstractSituatedLearnerScala[SAMPLE, MR, ESTEP, ERESULT, DI] (numIterations,
                                                                        trainingData,
                                                                        trainingDataDebug,
                                                                        maxSentenceLength,
                                                                        lexiconGenerationBeamSize,
                                                                        parser,
                                                                        parserOutputLogger,
                                                                        categoryServices,
                                                                        genlex) {

  override protected def parameterUpdate(dataItem: DI,
                                         dataItemModel: IJointDataItemModel[MR, ESTEP],
                                         model: JModel,
                                         dataItemNumber: Int,
                                         epochNumber: Int): Unit = {

    // Parse with current model
    val parserOutput = parser.parse(dataItem.getSample, dataItemModel)
    stats.mean("model parse", parserOutput.getInferenceTime / 1000.0, "sec")
    parserOutputLogger.log(parserOutput, dataItemModel, s"$dataItemNumber-update")
    val modelParses = parserOutput.getDerivations()
    val bestModelParses = parserOutput.getMaxDerivations()

    if (modelParses.isEmpty) {
      // Skip the rest of the process if no complete parses available
      log.info(s"No parses for: $dataItem")
      log.info("Skipping parameter update")
    }
    else {

      log.info(s"Created ${modelParses.size()} model parses for training sample")
      log.info(s"Model parsing time: ${parserOutput.getInferenceTime / 1000.0}sec")
      log.info(s"Output is ${if (parserOutput.isExact) "exact" else "approximate"}")

      // Split all parses to valid and invalid sets
      val (validParses, invalidParses) = createValidInvalidSets(dataItem, modelParses)
      log.info(s"${validParses.size} valid parses, ${invalidParses.size} invalid parses")
      log.info("Valid parses:")
      validParses.foreach(logParse(dataItem, _, valid = true, verbose = true, dataItemModel))

      // Record if the best is the gold standard, if such debug information is available.
      if (bestModelParses.size() == 1 && isGoldDebugCorrect(dataItem, bestModelParses.get(0).getResult))
        stats.appendSampleStat(dataItemNumber, epochNumber, AbstractSituatedLearnerScala.GOLD_LF_IS_MAX)
      // Record if a valid parse was found.
      else if (validParses.nonEmpty)
        stats.appendSampleStat(dataItemNumber, epochNumber, AbstractSituatedLearnerScala.HAS_VALID_LF)

      if (validParses.nonEmpty) stats.count("valid", epochNumber)

      // Skip update if there are no valid or invalid parses
      if (validParses.isEmpty || invalidParses.isEmpty) log.info("No valid/invalid parses -- skipping")
      else {

        // Construct margin violating sets
        val (violatingValidParses, violatingInvalidParses) = marginViolatingSets(model, validParses, invalidParses)
        log.info(s"${violatingValidParses.size} violating valid parses, ${violatingInvalidParses.size} violating invalid parses")
        if (violatingValidParses.isEmpty) log.info("There are no violating valid/invalid parses -- skipping")
        else {
          log.info("Violating valid parses: ")
          violatingValidParses.foreach(logParse(dataItem, _, valid = true, verbose = true, dataItemModel))

          log.info("Violating invalid parses: ")
          violatingInvalidParses.foreach(logParse(dataItem, _, valid = false, verbose = true, dataItemModel))

          // Construct weight update vector
          val update = constructUpdate(violatingValidParses, violatingInvalidParses, model)

          // Update the parameters vector
          log.info(s"Update: $update")
          update.addTimesInto(1.0, model.getTheta)
          stats.appendSampleStat(dataItemNumber, epochNumber, AbstractSituatedLearnerScala.TRIGGERED_UPDATE)
          stats.count("update", epochNumber)
        }
      }
    }
  }

  override protected def validate(dataItem: DI, hypothesis: ERESULT): Boolean = validator.isValid(dataItem, hypothesis)

  // internal

  private val log = LoggerFactory.create(classOf[SituatedValidationPerceptronScala[SAMPLE, MR, ESTEP, ERESULT, DI]])

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
                                     parses: java.util.Collection[_ <: JointDerivation]): (Seq[JointDerivation], Seq[JointDerivation]) = {
    val (valids, invalids, _) = parses.asScala.foldLeft((Seq.empty[JointDerivation], Seq.empty[JointDerivation], -java.lang.Double.MAX_VALUE)) {
      case ((validParses, invalidParses, validScore), parse) =>
        if (validate(dataItem, parse.getResult))
        // Case using hard updates, only keep the highest scored valid ones
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

  private def marginViolatingSets(model: JModel,
                                  validParses: Seq[JointDerivation],
                                  invalidParses: Seq[JointDerivation]): (List[JointDerivation], List[JointDerivation]) = {
    val valids = validParses
    val invalids = invalidParses.map((_, false))

    val violatingParses = valids.foldLeft((List.empty[JointDerivation], invalids)) {
      case ((violatingValidsInner, invalidsWithFlags), validParse) =>
        var isValidViolating = false
        val invalidsWithUpdatedFlags = invalidsWithFlags.map { case (invalid, flag) =>
          val featureDelta = validParse.getMeanMaxFeatures.addTimes(-1.0, invalid.getMeanMaxFeatures)
          val deltaScore = model score featureDelta
          val threshold = margin * featureDelta.l1Norm()
          if (deltaScore < threshold) { isValidViolating = true; (invalid, true) }
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

}

object SituatedValidationPerceptronScala {

  case class Creator[SAMPLE <: ISituatedDataItem[Sentence, _], MR, ESTEP, ERESULT, DI <: ILabeledDataItem[SAMPLE, _]]()
    extends IResourceObjectCreator[SituatedValidationPerceptronScala[SAMPLE, MR, ESTEP, ERESULT, DI]] {

    override def create(params: ParameterizedExperiment#Parameters, repo: IResourceRepository): SituatedValidationPerceptronScala[SAMPLE, MR, ESTEP, ERESULT, DI] = {

      /**
        * Number of training iterations
        */
      val trainingData = repo.get(params.get("data"))

      val parser = repo.get(ParameterizedExperiment.PARSER_RESOURCE).asInstanceOf[IJointParser[SAMPLE, MR, ESTEP, ERESULT]]

      val validator = repo.get(params.get("validator")).asInstanceOf[IValidator[DI, ERESULT]]

      val hardUpdates = (params get "hard").toBoolean

      val parserOutputLogger: IJointOutputLogger[MR, ESTEP, ERESULT] =
        if (params contains "parseLogger")
          repo.get(params get "parseLogger").asInstanceOf[IJointOutputLogger[MR, ESTEP, ERESULT]]
        else new IJointOutputLogger[MR, ESTEP, ERESULT] {
          // Stub, do nothing.
          override def log(output: IJointOutput[MR, ERESULT], dataItemModel: IJointDataItemModel[MR, ESTEP], tag: String): Unit = ()

          override def log(output: IParserOutput[MR], dataItemModel: IDataItemModel[MR], tag: String): Unit = ()
        }

      /**
        * CategoryServices required for lexical induction.
        * GENLEX procedure. If 'null' skip lexical induction.
        */
      val (genlex, categoryServices) =
        if (params contains "genlex")
          (repo.get(params get "genlex").asInstanceOf[ILexiconGenerator[DI, MR, IJointModelImmutable[SAMPLE, MR, ESTEP]]],
            repo.get(ParameterizedExperiment.CATEGORY_SERVICES_RESOURCE).asInstanceOf[ICategoryServices[MR]])
        else (null, null)

      /**
        * Beam size to use when doing loss sensitive pruning with generated lexicon.
        */
      val lexiconGenerationBeamSize =
        if (params contains "genlexbeam") params.get("genlexbeam").toInt
        else 20

      /**
        * Margin to scale the relative loss function
        */
      val margin =
        if (params.contains("margin")) params.get("margin").toDouble
        else 1.0

      /**
        * Max sentence length. Sentence longer than this value will be skipped during training
        */
      val maxSentenceLength =
        if (params contains "maxSentenceLength") params.get("maxSentenceLength").toInt
        else Integer.MAX_VALUE

      /**
        * Number of training iterations
        */
      val numTrainingIterations =
        if (params contains "iter") params.get("iter").toInt
        else 4

      /**
        * TrainingDataDebug is a mapping a subset of training samples into their gold label for debug.
        */
      new SituatedValidationPerceptronScala(
        numIterations = numTrainingIterations,
        margin = margin,
        trainingData = trainingData,
        maxSentenceLength = maxSentenceLength,
        trainingDataDebug = new java.util.HashMap[DI, edu.cornell.cs.nlp.utils.composites.Pair[MR, ERESULT]],
        lexiconGenerationBeamSize = lexiconGenerationBeamSize,
        parser = parser,
        hardUpdates = hardUpdates,
        validator = validator,
        parserOutputLogger = parserOutputLogger,
        categoryServices = categoryServices,
        genlex = genlex
      )

    }

    /**
      * The resource type.
      */
    override def `type`(): String = ???

    /**
      * Return a usage objects describing the resource created and how it can be
      * created.
      */
    override def usage(): ResourceUsage =
      new ResourceUsage.Builder(`type`(), classOf[SituatedValidationPerceptronScala[SAMPLE, MR, ESTEP, ERESULT, DI]])
        .setDescription("Validation senstive perceptron for situated learning of models with situated inference (cite: Artzi and Zettlemoyer 2013)")
        .addParam("data", "id", "Training data")
        .addParam("hard", "boolean", "Use hard updates (i.e., only use max scoring valid parses/evaluation as positive samples). Options: true, false. Default: false")
        .addParam("parseLogger", "id", "Parse logger for debug detailed logging of parses")
        .addParam("genlex", "ILexiconGenerator", "GENLEX procedure")
        .addParam("genlexbeam", "int", "Beam to use for GENLEX inference (parsing).")
        .addParam("margin", "double", "Margin to use for updates. Updates will be done when this margin is violated.")
        .addParam("maxSentenceLength", "int", "Max sentence length to process")
        .addParam("iter", "int", "Number of training iterations")
        .addParam("validator", "IValidator", "Validation function")
        .build()
  }

}
