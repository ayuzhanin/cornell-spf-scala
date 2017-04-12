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
import edu.cornell.cs.nlp.utils.collections.CollectionUtils
import edu.cornell.cs.nlp.utils.filter.IFilter
import edu.cornell.cs.nlp.utils.log.{ILogger, LoggerFactory}
import edu.cornell.cs.nlp.utils.system.MemoryReport


object AbstractLearnerScala {
  protected val GOLD_LF_IS_MAX: String = "G"
  protected val HAS_VALID_LF: String = "V"
  protected val TRIGGERED_UPDATE: String = "U"
}

abstract class AbstractLearnerScala[SAMPLE <: IDataItem[_],
                                    DI <: ILabeledDataItem[SAMPLE, _],
                                    PO <: IParserOutput[MR], MR] protected(val epochs: Int,
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

  /**
    * Learning statistics.
    */
  protected val stats: LearningStats = new LearningStats.Builder(trainingData.size)
    .addStat(HAS_VALID_LF, "Has a valid parse")
    .addStat(TRIGGERED_UPDATE, "Sample triggered update")
    .addStat(GOLD_LF_IS_MAX, "The best-scoring LF equals the provided GOLD debug LF")
    .setNumberStat("Number of new lexical entries added").build

  override def train(model: Model[SAMPLE, MR]): Unit = { // Init GENLEX.
    AbstractLearner.LOG.info("Initializing GENLEX ...")
    genlex.init(model)
    // Epochs
    var epochNumber: Int = 0
    while ( {
      epochNumber < epochs
    }) { // Training epoch, iterate over all training samples
      AbstractLearner.LOG.info("=========================")
      AbstractLearner.LOG.info("Training epoch %d", epochNumber)
      AbstractLearner.LOG.info("=========================")
      var itemCounter: Int = -1
      // Iterating over training data
      import scala.collection.JavaConversions._
      for (dataItem <- trainingData) { // Process a single training sample
        // Record start time
        val startTime: Long = System.currentTimeMillis
        // Log sample header
        AbstractLearner.LOG.info("%d : ================== [%d]", {
          itemCounter += 1; itemCounter
        }, epochNumber)
        AbstractLearner.LOG.info("Sample type: %s", dataItem.getClass.getSimpleName)
        AbstractLearner.LOG.info("%s", dataItem)
        // Skip sample, if over the length limit
        if (!processingFilter.test(dataItem)) {
          AbstractLearner.LOG.info("Skipped training sample, due to processing filter")
          continue //todo: continue is not supported
        }
        stats.count("Processed", epochNumber)
        try { // Data item model
          val dataItemModel: IDataItemModel[MR] = model.createDataItemModel(dataItem.getSample)
          // ///////////////////////////
          // Step I: Parse with current model. If we get a valid
          // parse, update parameters.
          // ///////////////////////////
          // Parse with current model and record some statistics
          val parserOutput: PO = parse(dataItem, dataItemModel)
          stats.mean("Model parse", parserOutput.getParsingTime / 1000.0, "sec")
          parserOutputLogger.log(parserOutput, dataItemModel, String.format("train-%d-%d", epochNumber, itemCounter))
          val modelParses: util.List[_ <: IDerivation[MR]] = parserOutput.getAllDerivations
          AbstractLearner.LOG.info("Model parsing time: %.4fsec", parserOutput.getParsingTime / 1000.0)
          AbstractLearner.LOG.info("Output is %s", if (parserOutput.isExact) "exact"
          else "approximate")
          AbstractLearner.LOG.info("Created %d model parses for training sample:", modelParses.size)
          import scala.collection.JavaConversions._
          for (parse <- modelParses) {
            logParse(dataItem, parse, validate(dataItem, parse.getSemantics), true, dataItemModel)
          }
          // Create a list of all valid parses
          val validParses: util.List[_ <: IDerivation[MR]] = getValidParses(parserOutput, dataItem)
          // If has a valid parse, call parameter update procedure
          // and continue
          if (!validParses.isEmpty && errorDriven) {
            parameterUpdate(dataItem, parserOutput, parserOutput, model, itemCounter, epochNumber)
            continue //todo: continue is not supported
          }
          // Step II: Generate new lexical entries, prune and update
          // the model. Keep the parser output for Step III.
          if (genlex == null) { // Skip the example if not doing lexicon learning
            continue //todo: continue is not supported
          }
          val generationParserOutput: PO = lexicalInduction(dataItem, itemCounter, dataItemModel, model, epochNumber)
          // Step III: Update parameters
          if (conflateGenlexAndPrunedParses && generationParserOutput != null) parameterUpdate(dataItem, parserOutput, generationParserOutput, model, itemCounter, epochNumber)
          else {
            val prunedParserOutput: PO = parse(dataItem, parsingFilterFactory.create(dataItem), dataItemModel)
            AbstractLearner.LOG.info("Conditioned parsing time: %.4fsec", prunedParserOutput.getParsingTime / 1000.0)
            parserOutputLogger.log(prunedParserOutput, dataItemModel, String.format("train-%d-%d-conditioned", epochNumber, itemCounter))
            parameterUpdate(dataItem, parserOutput, prunedParserOutput, model, itemCounter, epochNumber)
          }
        } finally {
          // Record statistics.
          stats.mean("Sample processing", (System.currentTimeMillis - startTime) / 1000.0, "sec")
          AbstractLearner.LOG.info("Total sample handling time: %.4fsec", (System.currentTimeMillis - startTime) / 1000.0)
        }
      }
      // Output epoch statistics
      AbstractLearner.LOG.info("System memory: %s", MemoryReport.generate)
      AbstractLearner.LOG.info("Epoch stats:")
      AbstractLearner.LOG.info(stats)

      {
        epochNumber += 1; epochNumber
      }
    }
  }

  private def getValidParses(parserOutput: PO, dataItem: DI): util.List[_ <: IDerivation[MR]] = {
    val parses: util.List[_ <: IDerivation[MR]] = new util.LinkedList[IDerivation[MR]](parserOutput.getAllDerivations)
    // Use validation function to prune generation parses. Syntax is not
    // used to distinguish between derivations.
    CollectionUtils.filterInPlace(parses, (e: (_$1) forSome {type _$1 <: IDerivation[MR]}) => validate(dataItem, e.getSemantics))
    parses
  }

  private def lexicalInduction(dataItem: DI, dataItemNumber: Int, dataItemModel: IDataItemModel[MR], model: Model[SAMPLE, MR], epochNumber: Int): PO = { // Generate lexical entries
    val generatedLexicon: ILexiconImmutable[MR] = genlex.generate(dataItem, model, categoryServices)
    AbstractLearner.LOG.info("Generated lexicon size = %d", generatedLexicon.size)
    if (generatedLexicon.size > 0) { // Case generated lexical entries
      // Create pruning filter, if the data item fits
      val pruningFilter: Predicate[ParsingOp[MR]] = parsingFilterFactory.create(dataItem)
      // Parse with generated lexicon
      val parserOutput: PO = parse(dataItem, pruningFilter, dataItemModel, generatedLexicon, lexiconGenerationBeamSize)
      // Log lexical generation parsing time
      stats.mean("genlex parse", parserOutput.getParsingTime / 1000.0, "sec")
      AbstractLearner.LOG.info("Lexicon induction parsing time: %.4fsec", parserOutput.getParsingTime / 1000.0)
      AbstractLearner.LOG.info("Output is %s", if (parserOutput.isExact) "exact"
      else "approximate")
      // Log generation parser output
      parserOutputLogger.log(parserOutput, dataItemModel, String.format("train-%d-%d-genlex", epochNumber, dataItemNumber))
      AbstractLearner.LOG.info("Created %d lexicon generation parses for training sample", parserOutput.getAllDerivations.size)
      // Get valid lexical generation parses
      val validParses: util.List[_ <: IDerivation[MR]] = getValidParses(parserOutput, dataItem)
      AbstractLearner.LOG.info("Removed %d invalid parses", parserOutput.getAllDerivations.size - validParses.size)
      // Collect max scoring valid generation parses
      val bestGenerationParses: util.List[IDerivation[MR]] = new util.LinkedList[IDerivation[MR]]
      var currentMaxModelScore: Double = -Double.MAX_VALUE
      import scala.collection.JavaConversions._
      for (parse <- validParses) {
        if (parse.getScore > currentMaxModelScore) {
          currentMaxModelScore = parse.getScore
          bestGenerationParses.clear()
          bestGenerationParses.add(parse)
        }
        else if (parse.getScore == currentMaxModelScore) bestGenerationParses.add(parse)
      }
      AbstractLearner.LOG.info("%d valid best parses for lexical generation:", bestGenerationParses.size)
      import scala.collection.JavaConversions._
      for (parse <- bestGenerationParses) {
        logParse(dataItem, parse, true, true, dataItemModel)
      }
      // Update the model's lexicon with generated lexical
      // entries from the max scoring valid generation parses
      var newLexicalEntries: Int = 0
      import scala.collection.JavaConversions._
      for (parse <- bestGenerationParses) {
        import scala.collection.JavaConversions._
        for (entry <- parse.getMaxLexicalEntries) {
          if (genlex.isGenerated(entry)) {
            if (model.addLexEntry(LexiconGenerationServices.unmark(entry))) {
              newLexicalEntries += 1
              AbstractLearner.LOG.info("Added LexicalEntry to model: %s [%s]", entry, model.getTheta.printValues(model.computeFeatures(entry)))
            }
            // Lexical generators might link related lexical
            // entries, so if we add the original one, we
            // should also add all its linked ones
            import scala.collection.JavaConversions._
            for (linkedEntry <- entry.getLinkedEntries) {
              if (model.addLexEntry(LexiconGenerationServices.unmark(linkedEntry))) {
                newLexicalEntries += 1
                AbstractLearner.LOG.info("Added (linked) LexicalEntry to model: %s [%s]", linkedEntry, model.getTheta.printValues(model.computeFeatures(linkedEntry)))
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
      AbstractLearner.LOG.info("Skipped GENLEX step. No generated lexical items.")
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
    AbstractLearner.LOG.info("%s%s[%.2f%s] %s", if (isGold) "* "
    else "  ", if (tag == null) ""
    else tag + " ", parse.getScore, if (valid == null) ""
    else if (valid) ", V"
    else ", X", parse)
    if (verbose) {
      import scala.collection.JavaConversions._
      for (step <- parse.getMaxSteps) {
        AbstractLearner.LOG.info("\t%s", step.toString(false, false, dataItemModel.getTheta))
      }
    }
  }

  /**
    * Parameter update method.
    */
  protected def parameterUpdate(dataItem: DI, realOutput: PO, goodOutput: PO, model: Model[SAMPLE, MR], itemCounter: Int, epochNumber: Int): Unit

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
  protected def parse(dataItem: DI, pruningFilter: Predicate[ParsingOp[MR]], dataItemModel: IDataItemModel[MR], generatedLexicon: ILexiconImmutable[MR], beamSize: Integer): PO

  /**
    * Validation method.
    */
  protected def validate(dataItem: DI, hypothesis: MR): Boolean
}
