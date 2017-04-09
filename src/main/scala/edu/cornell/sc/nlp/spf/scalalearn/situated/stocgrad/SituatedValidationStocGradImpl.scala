package edu.cornell.sc.nlp.spf.scalalearn.situated.stocgrad

import edu.cornell.cs.nlp.spf.base.hashvector.{HashVectorFactory, IHashVector}
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
import edu.cornell.cs.nlp.spf.parser.joint.graph.IJointGraphParser
import edu.cornell.cs.nlp.spf.parser.joint.model.{IJointDataItemModel, IJointModelImmutable}
import edu.cornell.cs.nlp.spf.parser.joint.{IJointOutput, IJointOutputLogger}
import edu.cornell.cs.nlp.utils.composites.Pair
import edu.cornell.cs.nlp.utils.filter.IFilter
import edu.cornell.cs.nlp.utils.log.{ILogger, LoggerFactory}
import edu.cornell.sc.nlp.spf.scalalearn.situated.AbstractSituatedLearnerImpl
import edu.cornell.sc.nlp.spf.scalalearn.situated.AbstractSituatedLearnerImpl._

object SituatedValidationStocGradImpl {

  class Creator[SAMPLE <: ISituatedDataItem[Sentence, _],
                MR,
                ESTEP,
                ERESULT,
                DI <: ISituatedDataItem[SAMPLE, _]](val name: String)
      extends IResourceObjectCreator[SituatedValidationStocGradImpl[SAMPLE, MR, ESTEP, ERESULT, DI]] {

    def this() = {
      this("learner.situated.valid.stocgrad")
    }

    @SuppressWarnings("unchecked")
    override def create(params: ParameterizedExperiment#Parameters,
                        repo: IResourceRepository): SituatedValidationStocGradImpl[SAMPLE, MR, ESTEP, ERESULT, DI] = {

      val numIterations =
        if (params.contains("iter")) params.get("iter").toInt
        else 4

      val trainingData: IDataCollection[DI] = repo.get(params.get("data")).asInstanceOf[IDataCollection[DI]]

      val maxSentenceLength =
        if (params.contains("maxSentenceLength")) params.get("maxSentenceLength").toInt
        else Integer.MAX_VALUE

      val lexiconGenerationBeamSize =
        if (params.contains("genlexbeam")) params.get("genlexbeam").toInt
        else 20

      val parser: IJointGraphParser[SAMPLE, MR, ESTEP, ERESULT] = repo.get(ParameterizedExperiment.PARSER_RESOURCE).asInstanceOf[IJointGraphParser[SAMPLE, MR, ESTEP, ERESULT]]

      val parserOutputLogger =
        if (params.contains("parseLogger")) repo.get(params.get("parseLogger")).asInstanceOf[IJointOutputLogger[MR, ESTEP, ERESULT]]
        else new IJointOutputLogger[MR, ESTEP, ERESULT] {
          override def log(output: IJointOutput[MR, ERESULT], dataItemModel: IJointDataItemModel[MR, ESTEP], tag: String): Unit = ()
          override def log(output: IParserOutput[MR], dataItemModel: IDataItemModel[MR], tag: String): Unit = ()
        }

      val alpha0 =
        if (params.contains("alpha0")) params.get("alpha0").toDouble
        else 0.1

      val validator = repo.get(params.get("validator")).asInstanceOf[IValidator[DI, ERESULT]]

      val c =
        if (params.contains("c")) params.get("c").toDouble
        else 0.0001

      val (categoryServices, genlex) =
        if (params.contains("genlex"))
          (repo.get(ParameterizedExperiment.CATEGORY_SERVICES_RESOURCE).asInstanceOf[ICategoryServices[MR]],
            repo.get(params.get("genlex")).asInstanceOf[ILexiconGenerator[DI, MR, IJointModelImmutable[SAMPLE, MR, ESTEP]]])
        else (null, null)

      new SituatedValidationStocGradImpl[SAMPLE, MR, ESTEP, ERESULT, DI](
        numIterations,
        trainingData,
        new java.util.HashMap[DI, Pair[MR, ERESULT]],
        maxSentenceLength,
        lexiconGenerationBeamSize,
        parser,
        parserOutputLogger,
        alpha0,
        c,
        validator,
        categoryServices,
        genlex
      )
    }

    override def `type`: String = name

    override def usage: ResourceUsage = new ResourceUsage.Builder(`type`, classOf[SituatedValidationStocGradImpl[SAMPLE, MR, ESTEP, ERESULT, DI]])
      .setDescription("Validation senstive stochastic gradient for situated learning of models with situated inference (cite: Artzi and Zettlemoyer 2013)")
      .addParam("c", "double", "Learing rate c parameter, temperature=alpha_0/(1+c*tot_number_of_training_instances)")
      .addParam("alpha0", "double", "Learing rate alpha0 parameter, temperature=alpha_0/(1+c*tot_number_of_training_instances)")
      .addParam("validator", "IValidator", "Validation function")
      .addParam("data", "id", "Training data")
      .addParam("genlex", "ILexiconGenerator", "GENLEX procedure")
      .addParam("parseLogger", "id", "Parse logger for debug detailed logging of parses")
      .addParam("genlexbeam", "int", "Beam to use for GENLEX inference (parsing).")
      .addParam("maxSentenceLength", "int", "Max sentence length to process")
      .addParam("iter", "int", "Number of training iterations")
      .build
  }

}

/**
  * Since the logical form is marginalized for computing the normalization
  * constant and probabilities, the validator has access only to the final
  * result of the execution. This is in contrast to the validator in
  * {@link SituatedValidationPerceptron}.
  */

class SituatedValidationStocGradImpl[SAMPLE <: ISituatedDataItem[Sentence, _],
                                    MR,
                                    ESTEP,
                                    ERESULT,
                                    DI <: ILabeledDataItem[SAMPLE, _]] private(
                                      val numIterations: Int,
                                      override val trainingData: IDataCollection[DI],
                                      override val trainingDataDebug: java.util.Map[DI, Pair[MR, ERESULT]],
                                      override val maxSentenceLength: Int,
                                      override val lexiconGenerationBeamSize: Int,
                                      val graphParser: IJointGraphParser[SAMPLE, MR, ESTEP, ERESULT],
                                      override val parserOutputLogger: IJointOutputLogger[MR, ESTEP, ERESULT],
                                      val alpha0: Double,
                                      val c: Double,
                                      val validator: IValidator[DI, ERESULT],
                                      override val categoryServices: ICategoryServices[MR],
                                      override val genlex: ILexiconGenerator[DI, MR, IJointModelImmutable[SAMPLE, MR, ESTEP]])
  extends AbstractSituatedLearnerImpl[SAMPLE, MR, ESTEP, ERESULT, DI](numIterations,
                                                                  trainingData,
                                                                  trainingDataDebug,
                                                                  maxSentenceLength,
                                                                  lexiconGenerationBeamSize,
                                                                  graphParser,
                                                                  parserOutputLogger,
                                                                  categoryServices,
                                                                  genlex) {

  override val log: ILogger = LoggerFactory.create(classOf[SituatedValidationStocGradImpl[SAMPLE, MR, ESTEP, ERESULT, DI]])

  log.info(s"Init SituatedValidationSensitiveStocGrad: numIterations=$numIterations, trainingData.size=${trainingData.size}, trainingDataDebug.size=${trainingDataDebug.size}, maxSentenceLength=$maxSentenceLength ...")
  log.info(s"Init SituatedValidationSensitiveStocGrad: ... lexiconGenerationBeamSize=$lexiconGenerationBeamSize, alpah0=$alpha0, c=$c")

  private var stocGradientNumUpdates: Int = 0


  override def train(model: JModel): Unit = {
    stocGradientNumUpdates = 0
    super.train(model)
  }

  override protected def parameterUpdate(dataItem: DI, dataItemModel: IJointDataItemModel[MR, ESTEP], model: JModel, dataItemNumber: Int, epochNumber: Int): Unit = { // Parse with current model
    // Parse with current model
    val parserOutput = graphParser.parse(dataItem.getSample, dataItemModel)
    stats.mean("model parse", parserOutput.getInferenceTime / 1000.0, "sec")
    parserOutputLogger.log(parserOutput, dataItemModel, s"$dataItemNumber-update")
    val modelParses = parserOutput.getDerivations
    val bestModelParses = parserOutput.getMaxDerivations

    if (modelParses.isEmpty) { // Skip the rest of the process if no complete parses
      // available
      log.info(s"No parses for: $dataItem")
      log.info("Skipping parameter update")
      return
    }

    log.info(s"Created ${modelParses.size} model parses for training sample")
    log.info(s"Model parsing time: ${parserOutput.getInferenceTime / 1000.0}sec")
    log.info(s"Output is ${if (parserOutput.isExact) "exact" else "approximate"}")

    // Create the update
    val update: IHashVector = HashVectorFactory.create

    // Step A: Compute the positive half of the update: conditioned on getting successful validation
    val filter: IFilter[ERESULT] = (e: ERESULT) => validate(dataItem, e)

    val logConditionedNorm: Double = parserOutput.logNorm(filter)
    if (logConditionedNorm == java.lang.Double.NEGATIVE_INFINITY) {
      // No positive update, skip the update.
      return
    }
    else {
      // Case have complete valid parses.
      val expectedFeatures = parserOutput.logExpectedFeatures(filter)
      expectedFeatures.add(-logConditionedNorm)
      expectedFeatures.applyFunction((value: Double) => java.lang.Math.exp(value))
      expectedFeatures.dropNoise()
      expectedFeatures.addTimesInto(1.0, update)
      log.info(s"Positive update: $expectedFeatures")

      // Record if the best is the gold standard, if such debug information is available
      stats.count("valid", epochNumber)
      if (bestModelParses.size == 1 && isGoldDebugCorrect(dataItem, bestModelParses.get(0).getResult))
        stats.appendSampleStat(dataItemNumber, epochNumber, GOLD_LF_IS_MAX)
      else stats.appendSampleStat(dataItemNumber, epochNumber, HAS_VALID_LF)
    }

    // Step B: Compute the negative half of the update: expectation under the current model
    val logNorm = parserOutput.logNorm
    if (logNorm == java.lang.Double.NEGATIVE_INFINITY) log.info("No negative update")
    else {
      // Case have complete parses.
      val expectedFeatures = parserOutput.logExpectedFeatures
      expectedFeatures.add(-logNorm)
      expectedFeatures.applyFunction((value: Double) => Math.exp(value))
      expectedFeatures.dropNoise()
      expectedFeatures.addTimesInto(-1.0, update)
      log.info(s"Negative update: $expectedFeatures")
    }

    // Step C: Apply the update Validate the update
    if (!model.isValidWeightVector(update)) throw new IllegalStateException(s"invalid update: $update")

    // Scale the update
    val scale = alpha0 / (1.0 + c * stocGradientNumUpdates)
    update.multiplyBy(scale)
    update.dropNoise()
    stocGradientNumUpdates += 1
    log.info("Scale: %f", scale)
    if (update.size == 0) {
      log.info("No update")
      return
    }
    else {
      log.info(s"Update: $update")
      stats.appendSampleStat(dataItemNumber, epochNumber, TRIGGERED_UPDATE)
      stats.count("update", epochNumber)
    }

    // Check for NaNs and super large updates
    if (update.isBad) {
      log.error(s"Bad update: $update -- log-norm: $logNorm -- feats: ${null}")
      log.error(model.getTheta.printValues(update))
      throw new IllegalStateException("bad update")
    }
    else {
      if (!update.valuesInRange(-100, 100)) log.warn("Large update")
      // Do the update
      update.addTimesInto(1, model.getTheta)
    }
  }

  override protected def validate(dataItem: DI, hypothesis: ERESULT): Boolean = validator.isValid(dataItem, hypothesis)
}

