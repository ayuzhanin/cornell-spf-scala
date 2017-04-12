package edu.cornell.cs.nlp.spf.scalalearn.validation

import java.util
import java.util.{LinkedList, List, Map}
import java.util.function.Predicate

import edu.cornell.cs.nlp.spf.ccg.categories.ICategoryServices
import edu.cornell.cs.nlp.spf.ccg.lexicon.ILexiconImmutable
import edu.cornell.cs.nlp.spf.data.{IDataItem, ILabeledDataItem}
import edu.cornell.cs.nlp.spf.data.collection.IDataCollection
import edu.cornell.cs.nlp.spf.genlex.ccg.{ILexiconGenerator, LexiconGenerationServices}
import edu.cornell.cs.nlp.spf.learn.{ILearner, LearningStats}
import edu.cornell.cs.nlp.spf.parser.{IDerivation, IOutputLogger, IParserOutput, ParsingOp}
import edu.cornell.cs.nlp.spf.parser.ccg.model.{IDataItemModel, IModelImmutable, Model}
import edu.cornell.cs.nlp.spf.parser.filter.IParsingFilterFactory
import edu.cornell.cs.nlp.utils.filter.IFilter
import edu.cornell.cs.nlp.utils.log.{ILogger, LoggerFactory}
import edu.cornell.cs.nlp.utils.system.MemoryReport

import scala.collection.JavaConverters._

object AbstractLearnerScala {
  protected val GOLD_LF_IS_MAX: String = "G"
  protected val HAS_VALID_LF: String = "V"
  protected val TRIGGERED_UPDATE: String = "U"
}

abstract class AbstractLearnerScala[SAMPLE <: IDataItem[_],
                                    DI <: ILabeledDataItem[SAMPLE, _],
                                    PO <: IParserOutput[MR],
                                    MR] protected(val epochs: Int,
                                                   val trainingData: IDataCollection[DI],
                                                   val trainingDataDebug: util.Map[DI, MR],
                                                   val lexiconGenerationBeamSize: Integer,
                                                   val parserOutputLogger: IOutputLogger[MR],
                                                   val conflateGenlexAndPrunedParses: Boolean,
                                                   val errorDriven: Boolean,
                                                   val categoryServices: ICategoryServices[MR],
                                                   val genlex: ILexiconGenerator[DI, MR, IModelImmutable[SAMPLE, MR]],
                                                   val processingFilter: IFilter[DI],
                                                   val parsingFilterFactory: IParsingFilterFactory[DI, MR])
  extends ILearner[SAMPLE, DI, Model[SAMPLE, MR]] {

  import AbstractLearnerScala._

  val log: ILogger = LoggerFactory.create(classOf[AbstractLearnerScala[SAMPLE, DI, PO, MR]])
  /**
    * Learning statistics.
    */
  protected val stats: LearningStats = new LearningStats.Builder(trainingData.size)
    .addStat(HAS_VALID_LF, "Has a valid parse")
    .addStat(TRIGGERED_UPDATE, "Sample triggered update")
    .addStat(GOLD_LF_IS_MAX, "The best-scoring LF equals the provided GOLD debug LF")
    .setNumberStat("Number of new lexical entries added").build

  override def train(model: Model[SAMPLE, MR]): Unit = {

    // Init GENLEX.
    log.info("Initializing GENLEX ...")
    genlex.init(model)

    // Epochs
    (1 to epochs).foreach { epochNumber =>

      // Training epoch, iterate over all training samples
      log.info("=========================")
      log.info(s"Training epoch $epochNumber")
      log.info("=========================")

      // Iterating over training data
      trainingData.asScala.zipWithIndex.foreach { case (dataItem, itemCounter) =>
        // Process a single training sample

        // Record start time
        val startTime: Long = System.currentTimeMillis

        // Log sample header
        log.info(s"$itemCounter : ================== [$epochNumber]")
        log.info(s"Sample type: ${dataItem.getClass.getSimpleName}")
        log.info(dataItem.toString)

        // Skip sample, if over the length limit
        if (!processingFilter.test(dataItem)) {
          log.info("Skipped training sample, due to processing filter")
          continue //todo: continue is not supported
        }

        stats.count("Processed", epochNumber)

        try
          // Data item model
          val dataItemModel: IDataItemModel[MR] = model.createDataItemModel(dataItem.getSample)

          // ///////////////////////////
          // Step I: Parse with current model. If we get a valid
          // parse, update parameters.
          // ///////////////////////////

          // Parse with current model and record some statistics
          val parserOutput: PO = parse(dataItem, dataItemModel)
          stats.mean("Model parse", parserOutput.getParsingTime / 1000.0, "sec")
          parserOutputLogger.log(parserOutput, dataItemModel, s"train-$epochNumber-$itemCounter")

          val modelParses = parserOutput.getAllDerivations.asScala
          log.info(s"Model parsing time: ${parserOutput.getParsingTime / 1000.0}")
          log.info(s"Output is ${if (parserOutput.isExact) "exact" else "approximate"}")
          log.info(s"Created ${modelParses.size} model parses for training sample:")

          modelParses.foreach(parse => logParse(dataItem, parse, validate(dataItem, parse.getSemantics), verbose = true, dataItemModel = dataItemModel))

          // Create a list of all valid parses
          val validParses = getValidParses(parserOutput, dataItem).asScala

          // If has a valid parse, call parameter update procedure and continue
          if (validParses.nonEmpty && errorDriven) {
            parameterUpdate(dataItem, parserOutput, parserOutput, model, itemCounter, epochNumber)
            continue //todo: continue is not supported
          }


          // ///////////////////////////
          // Step II: Generate new lexical entries, prune and update
          // the model. Keep the parser output for Step III.
          // ///////////////////////////

          if (genlex == null) {
            // Skip the example if not doing lexicon learning
            continue //todo: continue is not supported
          }

          val generationParserOutput = lexicalInduction(dataItem, itemCounter, dataItemModel, model, epochNumber)

          // ///////////////////////////
          // Step III: Update parameters
          // ///////////////////////////

          if (conflateGenlexAndPrunedParses && generationParserOutput != null)
            parameterUpdate(dataItem, parserOutput, generationParserOutput, model, itemCounter, epochNumber)
          else {
            val prunedParserOutput: PO = parse(dataItem, parsingFilterFactory.create(dataItem), dataItemModel)
            log.info("Conditioned parsing time: %.4fsec", prunedParserOutput.getParsingTime / 1000.0)
            parserOutputLogger.log(prunedParserOutput, dataItemModel, String.format("train-%d-%d-conditioned", epochNumber, itemCounter))
            parameterUpdate(dataItem, parserOutput, prunedParserOutput, model, itemCounter, epochNumber)
          }

        finally {
          // Record statistics.
          stats.mean("Sample processing", (System.currentTimeMillis - startTime) / 1000.0, "sec")
          log.info(s"Total sample handling time: ${(System.currentTimeMillis - startTime) / 1000.0}sec")
        }

        // Output epoch statistics
        log.info(s"System memory: ${MemoryReport.generate}")
        log.info("Epoch stats:")
        log.info(stats)
      }
    }
  }

  // Use validation function to prune generation parses. Syntax is not used to distinguish between derivations.
  private def getValidParses(parserOutput: PO, dataItem: DI) =
    parserOutput.getAllDerivations.asScala.filter(e => validate(dataItem, e.getSemantics)).toList

  private def lexicalInduction(dataItem: DI, dataItemNumber: Int, dataItemModel: IDataItemModel[MR], model: Model[SAMPLE, MR], epochNumber: Int): PO = { // Generate lexical entries
    val generatedLexicon = genlex.generate(dataItem, model, categoryServices)
    log.info(s"Generated lexicon size = ${generatedLexicon.size}")

    if (generatedLexicon.size > 0) {
      // Case generated lexical entries

      // Create pruning filter, if the data item fits
      val pruningFilter = parsingFilterFactory.create(dataItem)

      // Parse with generated lexicon
      val parserOutput = parse(dataItem, pruningFilter, dataItemModel, generatedLexicon, lexiconGenerationBeamSize)

      // Log lexical generation parsing time
      stats.mean("genlex parse", parserOutput.getParsingTime / 1000.0, "sec")
      log.info(s"Lexicon induction parsing time: ${parserOutput.getParsingTime / 1000.0}sec")
      log.info(s"Output is ${if (parserOutput.isExact) "exact" else "approximate"}")

      // Log generation parser output
      parserOutputLogger.log(parserOutput, dataItemModel, s"train-$epochNumber-$dataItemNumber-genlex")
      log.info(s"Created ${parserOutput.getAllDerivations.size} lexicon generation parses for training sample")

      // Get valid lexical generation parses
      val validParses = getValidParses(parserOutput, dataItem)
      log.info(s"Removed ${parserOutput.getAllDerivations.size - validParses.size} invalid parses")

      // Collect max scoring valid generation parses
      val bestGenerationParses: util.List[IDerivation[MR]] = new util.LinkedList[IDerivation[MR]]
      var currentMaxModelScore = -java.lang.Double.MAX_VALUE

      for (parse <- validParses) {
        if (parse.getScore > currentMaxModelScore) {
          currentMaxModelScore = parse.getScore
          bestGenerationParses.clear()
          bestGenerationParses.add(parse)
        }
        else if (parse.getScore == currentMaxModelScore) bestGenerationParses.add(parse)
      }

      log.info(s"${bestGenerationParses.size} valid best parses for lexical generation:")

      bestGenerationParses.asScala.foreach(logParse(dataItem, _, valid = true, verbose = true, dataItemModel = dataItemModel))
      // Update the model's lexicon with generated lexical entries from the max scoring valid generation parses
      var newLexicalEntries: Int = 0

      for (parse <- bestGenerationParses.asScala) {
        for (entry <- parse.getMaxLexicalEntries.asScala) {
          if (genlex.isGenerated(entry)) {
            if (model.addLexEntry(LexiconGenerationServices.unmark(entry))) {
              newLexicalEntries += 1
              log.info(s"Added LexicalEntry to model: $entry [${model.getTheta.printValues(model.computeFeatures(entry))}]")
            }
            // Lexical generators might link related lexical
            // entries, so if we add the original one, we
            // should also add all its linked ones

            for (linkedEntry <- entry.getLinkedEntries.asScala) {
              if (model.addLexEntry(LexiconGenerationServices.unmark(linkedEntry))) {
                newLexicalEntries += 1
                log.info(s"Added (linked) LexicalEntry to model: $linkedEntry [${model.getTheta.printValues(model.computeFeatures(linkedEntry))}]")
              }
            }
          }
        }
      }
      // Record statistics
      if (newLexicalEntries > 0) stats.appendSampleStat(dataItemNumber, epochNumber, newLexicalEntries)
      parserOutput
    }
    else { // Skip lexical induction
      log.info("Skipped GENLEX step. No generated lexical items.")
      null
    }
  }

  protected def isGoldDebugCorrect(dataItem: DI, label: MR): Boolean = if (trainingDataDebug.containsKey(dataItem)) trainingDataDebug.get(dataItem) == label
  else false

  protected def logParse(dataItem: DI, parse: IDerivation[MR], valid: Boolean, verbose: Boolean, dataItemModel: IDataItemModel[MR]): Unit = {
    logParse(dataItem, parse, valid, verbose, null, dataItemModel)
  }

  protected def logParse(dataItem: DI, parse: IDerivation[MR], valid: Boolean, verbose: Boolean, tag: String, dataItemModel: IDataItemModel[MR]): Unit = {
    var isGold: Boolean = false
    if (isGoldDebugCorrect(dataItem, parse.getSemantics)) isGold = true
    else isGold = false
    log.info("%s%s[%.2f%s] %s", if (isGold) "* "
    else "  ", if (tag == null) ""
    else tag + " ", parse.getScore, if (valid == null) ""
    else if (valid) ", V"
    else ", X", parse)
    if (verbose) {
      import scala.collection.JavaConversions._
      for (step <- parse.getMaxSteps) {
        log.info("\t%s", step.toString(false, false, dataItemModel.getTheta))
      }
    }
  }

  /**
    * Parameter update method.
    */
  protected def parameterUpdate(dataItem: DI,
                                realOutput: PO,
                                goodOutput: PO,
                                model: Model[SAMPLE, MR],
                                itemCounter: Int,
                                epochNumber: Int): Unit

  /**
    * Unconstrained parsing method.
    */
  protected def parse(dataItem: DI, dataItemModel: IDataItemModel[MR]): PO

  /**
    * Constrained parsing method.
    */
  protected def parse(dataItem: DI, pruningFilter: Predicate[ParsingOp[MR]], dataItemModel: IDataItemModel[MR]): PO

  /**
    * Constrained parsing method for lexical generation.
    */
  protected def parse(dataItem: DI,
                      pruningFilter: Predicate[ParsingOp[MR]],
                      dataItemModel: IDataItemModel[MR],
                      generatedLexicon: ILexiconImmutable[MR],
                      beamSize: Integer): PO

  /**
    * Validation method.
    */
  protected def validate(dataItem: DI, hypothesis: MR): Boolean
}
