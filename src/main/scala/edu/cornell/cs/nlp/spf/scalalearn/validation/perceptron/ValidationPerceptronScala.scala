/** *****************************************************************************
  * Copyright (C) 2011 - 2015 Yoav Artzi, All rights reserved.
  * <p>
  * This program is free software; you can redistribute it and/or modify it under
  * the terms of the GNU General Public License as published by the Free Software
  * Foundation; either version 2 of the License, or any later version.
  * <p>
  * This program is distributed in the hope that it will be useful, but WITHOUT
  * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
  * FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
  * details.
  * <p>
  * You should have received a copy of the GNU General Public License along with
  * this program; if not, write to the Free Software Foundation, Inc., 51
  * Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
  * ******************************************************************************/
package edu.cornell.cs.nlp.spf.scalalearn.validation.perceptron

import java.util.function.Predicate

import scala.collection.JavaConverters._
import edu.cornell.cs.nlp.spf.base.hashvector.{HashVectorFactory, IHashVector}
import edu.cornell.cs.nlp.spf.ccg.categories.ICategoryServices
import edu.cornell.cs.nlp.spf.ccg.lexicon.ILexiconImmutable
import edu.cornell.cs.nlp.spf.data.{IDataItem, ILabeledDataItem}
import edu.cornell.cs.nlp.spf.data.collection.IDataCollection
import edu.cornell.cs.nlp.spf.data.utils.IValidator
import edu.cornell.cs.nlp.spf.explat.{IResourceRepository, ParameterizedExperiment}
import edu.cornell.cs.nlp.spf.explat.resources.IResourceObjectCreator
import edu.cornell.cs.nlp.spf.explat.resources.usage.ResourceUsage
import edu.cornell.cs.nlp.spf.genlex.ccg.ILexiconGenerator
import edu.cornell.cs.nlp.spf.parser._
import edu.cornell.cs.nlp.spf.parser.ccg.model.{IDataItemModel, IModelImmutable, Model}
import edu.cornell.cs.nlp.spf.parser.filter.{IParsingFilterFactory, StubFilterFactory}
import edu.cornell.cs.nlp.spf.scalalearn.validation.AbstractLearnerScala
import edu.cornell.cs.nlp.utils.filter.IFilter
import edu.cornell.cs.nlp.utils.log.{ILogger, LoggerFactory}

/**
  * Validation-based perceptron learner. See Artzi and Zettlemoyer 2013 for
  * detailed description.
  * <p>
  * The learner is insensitive to the syntactic category generated by the
  * inference procedure -- only the semantic portion is being validated. However,
  * parsers can be constrained to output only specific syntactic categories, see
  * the parser builders.
  * </p>
  * <p>
  * Parameter update step inspired by: Natasha Singh-Miller and Michael Collins.
  * 2007. Trigger-based Language Modeling using a Loss-sensitive Perceptron
  * Algorithm. In proceedings of ICASSP 2007.
  * </p>
  *
  * @tparam SAMPLE  Data item to use for inference.
  * @tparam DI       Data item for learning.
  * @tparam MR       Meaning representation.
  */

class ValidationPerceptronScala[SAMPLE <: IDataItem[_],
                                DI <: ILabeledDataItem[SAMPLE, _],
                                MR] private(val numIterations: Int,
                                            override val trainingData: IDataCollection[DI],
                                            override val trainingDataDebug: java.util.Map[DI, MR],
                                            override val lexiconGenerationBeamSize: Int,
                                            val parser: IParser[SAMPLE, MR],
                                            override val parserOutputLogger: IOutputLogger[MR],
                                            override val conflateGenlexAndPrunedParses: Boolean,
                                            override val errorDriven: Boolean,
                                            override val categoryServices: ICategoryServices[MR],
                                            override val genlex: ILexiconGenerator[DI, MR, IModelImmutable[SAMPLE, MR]],
                                            val margin: Double,
                                            val hardUpdates: Boolean,
                                            val validator: IValidator[DI, MR],
                                            override val processingFilter: IFilter[DI],
                                            override val parsingFilterFactory: IParsingFilterFactory[DI, MR])
  extends AbstractLearnerScala[SAMPLE, DI, IParserOutput[MR], MR](numIterations,
                                                            trainingData,
                                                            trainingDataDebug.asScala.toMap,
                                                            lexiconGenerationBeamSize,
                                                            parserOutputLogger,
                                                            conflateGenlexAndPrunedParses,
                                                            errorDriven,
                                                            categoryServices,
                                                            genlex,
                                                            processingFilter,
                                                            parsingFilterFactory) {

  import AbstractLearnerScala._

  log.info(s"Init ValidationStocGrad: numIterations=$numIterations, margin=$margin, trainingData.size()=${trainingData.size}, trainingDataDebug.size()=${trainingDataDebug.size}  ...")
  log.info(s"Init ValidationStocGrad: ... lexiconGenerationBeamSize=$lexiconGenerationBeamSize")
  log.info(s"Init ValidationStocGrad: ... conflateParses=${if (conflateGenlexAndPrunedParses) "true" else "false"}, errorDriven=${if (errorDriven) "true" else "false"}")
  log.info(s"Init ValidationStocGrad: ... parsingFilterFactory=$parsingFilterFactory")

  override protected def parameterUpdate(dataItem: DI,
                                         realOutput: IParserOutput[MR],
                                         goodOutput: IParserOutput[MR],
                                         model: Model[SAMPLE, MR],
                                         itemCounter: Int,
                                         epochNumber: Int): Unit = {
    val dataItemModel = model.createDataItemModel(dataItem.getSample)

    // Split all parses to valid and invalid sets
    val (validParses, invalidParses) = createValidInvalidSets(dataItem, realOutput, goodOutput)

    log.info(s"${validParses.size} valid parses, ${invalidParses.size} invalid parses")
    log.info("Valid parses:")

    validParses.foreach(logParse(dataItem, _, valid = true, verbose = true, dataItemModel))

    // Record if the output LF equals the available gold LF (if one is available), otherwise, record using validation signal.
    if (realOutput.getBestDerivations.size == 1 && isGoldDebugCorrect(dataItem, realOutput.getBestDerivations.get(0).getSemantics))
      stats.appendSampleStat(itemCounter, epochNumber, GOLD_LF_IS_MAX)
    // Record if a valid parse was found.
    else if (validParses.nonEmpty) stats.appendSampleStat(itemCounter, epochNumber, HAS_VALID_LF)

    if (validParses.nonEmpty) stats.count("Valid", epochNumber)

    // Skip update if there are no valid or invalid parses
    if (validParses.isEmpty || invalidParses.isEmpty) {
      log.info("No valid/invalid parses -- skipping")
    } else {

      val (violatingValidParses, violatingInvalidParses) = marginViolatingSets(model, margin, validParses, invalidParses)
      log.info(s"${violatingValidParses.size} violating valid parses, ${violatingInvalidParses.size} violating invalid parses")
      if (violatingValidParses.isEmpty) {
        log.info("There are no violating valid/invalid parses -- skipping")
      } else {

        log.info("Violating valid parses: ")
        violatingValidParses.foreach(logParse(dataItem, _, valid = true, verbose = true, dataItemModel))

        log.info("Violating invalid parses: ")
        violatingInvalidParses.foreach(logParse(dataItem, _, valid = false, verbose = true, dataItemModel))

        // Construct weight update vector
        val update = constructUpdate(violatingValidParses, violatingInvalidParses, model)
        // Update the parameters vector
        log.info(s"Update: $update")
        update.addTimesInto(1.0, model.getTheta)
        stats.appendSampleStat(itemCounter, epochNumber, TRIGGERED_UPDATE)
      }
    }
  }

  override protected def parse(dataItem: DI, dataItemModel: IDataItemModel[MR]): IParserOutput[MR] =
    parser.parse(dataItem.getSample, dataItemModel)

  override protected def parse(dataItem: DI,
                               pruningFilter: Predicate[ParsingOp[MR]],
                               dataItemModel: IDataItemModel[MR]): IParserOutput[MR] =
    parser.parse(dataItem.getSample, pruningFilter, dataItemModel)

  override protected def parse(dataItem: DI,
                               pruningFilter: Predicate[ParsingOp[MR]],
                               dataItemModel: IDataItemModel[MR],
                               generatedLexicon: ILexiconImmutable[MR],
                               beamSize: Integer): IParserOutput[MR] =
    parser.parse(dataItem.getSample, pruningFilter, dataItemModel, false, generatedLexicon, beamSize)

  override protected def validate(dataItem: DI, hypothesis: MR): Boolean = validator.isValid(dataItem, hypothesis)

  // internal

  /**
    * Collect valid and invalid parses.
    *
    * @param dataItem
    * @param realOutput
    * @param goodOutput
    * @return
    */
  private def createValidInvalidSets(dataItem: DI,
                                     realOutput: IParserOutput[MR],
                                     goodOutput: IParserOutput[MR]): (List[IDerivation[MR]], List[IDerivation[MR]]) = {

    // Collect invalid parses from readlOutput
    val invalidParses: scala.List[IDerivation[MR]] = realOutput.getAllDerivations.asScala.filter(parse => !validate(dataItem, parse.getSemantics)).toList

    // Track invalid parses, so we won't aggregate a parse more than once this is an approximation, but it's a best effort
    val invalidSemantics = invalidParses.toSet

    // Collect valid and invalid parses from goodOutput
    val starting = (-java.lang.Double.MAX_VALUE, scala.List.empty[IDerivation[MR]], invalidParses)
    val (_, valids, invalids) = goodOutput.getAllDerivations.asScala.foldLeft(starting){ (accumulated, parse) =>
      val (validScore, valids, invalids) = accumulated
      if (validate(dataItem, parse.getSemantics))
        if (hardUpdates)
          if (parse.getScore > validScore) (parse.getScore, List(parse), invalids)
          else if (parse.getScore == validScore) (validScore, valids :+ parse, invalids)
          else accumulated
        else (validScore, valids :+ parse, invalids)
      else if (!invalidSemantics.contains(parse)) (validScore, valids, invalids :+ parse)
      else accumulated
    }

    (valids, invalids)
  }

  private def constructUpdate[MR, P <: IDerivation[MR],MODEL <: IModelImmutable[_, MR]]
                             (violatingValidParses: List[P], violatingInvalidParses: List[P], model: MODEL): IHashVector = {
    // Create the parameter update
    val update = HashVectorFactory.create

    // Get the update for valid violating samples
    violatingValidParses.foreach(_.getAverageMaxFeatureVector.addTimesInto(1.0 / violatingValidParses.size, update))

    // Get the update for the invalid violating samples
    violatingInvalidParses.foreach(_.getAverageMaxFeatureVector.addTimesInto(-1.0 * (1.0 / violatingInvalidParses.size), update))

    // Prune small entries from the update
    update.dropNoise()
    // Validate the update
    if (!model.isValidWeightVector(update)) throw new IllegalStateException(s"invalid update: $update")
    update
  }

  private def marginViolatingSets[LF, P <: IDerivation[LF], MODEL <: IModelImmutable[_, LF]]
                                 (model: MODEL, margin: Double, validParses: List[P], invalidParses: List[P]): (List[P], List[P]) = {
    // Construct margin violating sets
    val invalids = invalidParses.map((_, false))

    val violatingParses = validParses.foldLeft((List.empty[P], invalids)){
      case ((violatingValidsInner, invalidsWithFlags), valid) =>
        var isValidViolating = false
        val invalidsWithUpdatedFlags = invalidsWithFlags.map { case (invalid, flag) =>
          val featureDelta = valid.getAverageMaxFeatureVector.addTimes(-1.0, invalid.getAverageMaxFeatureVector)
          val deltaScore = model.score(featureDelta)
          val threshold = margin * featureDelta.l1Norm()
          if (deltaScore < threshold) {isValidViolating = true; (invalid, true)}
          else (invalid, flag)
        }
        val violatingValidsUpdated =
          if (isValidViolating) violatingValidsInner :+ valid
          else violatingValidsInner
        (violatingValidsUpdated, invalidsWithUpdatedFlags)
    }

    val (violatingValids, violatingInvalidsCandidates) = violatingParses
    val violatingInvalids = violatingInvalidsCandidates.filter(_._2).map(_._1)

    (violatingValids, violatingInvalids)
  }

  private val log: ILogger = LoggerFactory.create(classOf[ValidationPerceptronScala[_ <: IDataItem[_], _ <: ILabeledDataItem[_, _], _]])
}

object ValidationPerceptronScala {

  class Creator[SAMPLE <: IDataItem[_], DI <: ILabeledDataItem[SAMPLE, _], MR](val name: String)
    extends IResourceObjectCreator[ValidationPerceptronScala[SAMPLE, DI, MR]] {

    def this() = {
      this("learner.validation.perceptron")
    }

    @SuppressWarnings("unchecked")
    override def create(params: ParameterizedExperiment#Parameters, repo: IResourceRepository): ValidationPerceptronScala[SAMPLE, DI, MR] = {

      val numIterations =
        if (params.contains("iter")) params.get("iter").toInt
        else 4

      val trainingData = repo.get(params.get("data"))

      val trainingDataDebug = new java.util.HashMap[DI, MR]

      val lexiconGenerationBeamSize =
        if (params.contains("genlexbeam")) params.get("genlexbeam").toInt
        else 20

      val parser = repo.get(ParameterizedExperiment.PARSER_RESOURCE).asInstanceOf[IParser[SAMPLE, MR]]

      val parserOutputLogger =
        if (params.contains("parseLogger")) repo.get(params.get("parseLogger")).asInstanceOf[IOutputLogger[MR]]
        else new IOutputLogger[MR]() {
          override def log(output: IParserOutput[MR], dataItemModel: IDataItemModel[MR], tag: String): Unit = ()
        }

      val conflateGenlexAndPrunedParses =
        if (params.contains("conflateParses")) params.get("conflateParses").toBoolean
        else false

      val errorDriven =
        if (params.contains("errorDriven")) params.get("errorDriven").toBoolean
        else false

      val (genlex, categoryServices) =
        if (params.contains("genlex"))
          (repo.get(params.get("genlex")).asInstanceOf[ILexiconGenerator[DI, MR, IModelImmutable[SAMPLE, MR]]],
            repo.get(ParameterizedExperiment.CATEGORY_SERVICES_RESOURCE).asInstanceOf[ICategoryServices[MR]])
        else (null, null)

      val margin =
        if (params.contains("margin")) params.get("margin").toDouble
        else 1.0

      val hardUpdates = "true" == params.get("hard")

      val validator = repo.get(params.get("validator")).asInstanceOf[IValidator[DI, MR]]

      val processingFilter: IFilter[DI] =
        if (params.contains("filter")) repo.get(params.get("filter")).asInstanceOf[IFilter[DI]]
        else { (_: DI) => true }

      val parsingFilterFactory =
        if (params.contains("filterFactory")) repo.get(params.get("filterFactory")).asInstanceOf[IParsingFilterFactory[DI, MR]]
        else new StubFilterFactory[DI, MR]

      new ValidationPerceptronScala[SAMPLE, DI, MR](numIterations,
        trainingData,
        trainingDataDebug,
        lexiconGenerationBeamSize,
        parser,
        parserOutputLogger,
        conflateGenlexAndPrunedParses,
        errorDriven,
        categoryServices,
        genlex,
        margin,
        hardUpdates,
        validator,
        processingFilter,
        parsingFilterFactory)
    }

    override def `type`: String = name

    override def usage: ResourceUsage =
      new ResourceUsage.Builder(`type`, classOf[ValidationPerceptronScala[_ <: IDataItem[_], _ <: ILabeledDataItem[_, _], _]])
        .setDescription("Validation-based perceptron")
        .addParam("data", "id", "Training data")
        .addParam("genlex", "ILexiconGenerator", "GENLEX procedure")
        .addParam("filterFactory", classOf[IParsingFilterFactory[_, _]], "Factory to create parsing filters (optional).")
        .addParam("hard", "boolean", "Use hard updates (i.e., only use max scoring valid parses/evaluation as positive samples). Options: true, false. Default: false")
        .addParam("parseLogger", "id", "Parse logger for debug detailed logging of parses")
        .addParam("tester", "ITester", "Intermediate tester to use between epochs")
        .addParam("genlexbeam", "int", "Beam to use for GENLEX inference (parsing).")
        .addParam("margin", "double", "Margin to use for updates. Updates will be done when this margin is violated.")
        .addParam("filter", "IFilter", "Processing filter")
        .addParam("iter", "int", "Number of training iterations")
        .addParam("validator", "IValidator", "Validation function")
        .addParam("conflateParses", "boolean", "Recyle lexical induction parsing output as pruned parsing output")
        .addParam("errorDriven", "boolean", "Error driven lexical generation, if the can generate a valid parse, skip lexical induction")
        .build
  }

}
