package edu.cornell.cs.nlp.spf.scalalearn.situated

import edu.cornell.cs.nlp.spf.ccg.categories.ICategoryServices
import edu.cornell.cs.nlp.spf.data.ILabeledDataItem
import edu.cornell.cs.nlp.spf.data.collection.IDataCollection
import edu.cornell.cs.nlp.spf.data.sentence.Sentence
import edu.cornell.cs.nlp.spf.data.situated.ISituatedDataItem
import edu.cornell.cs.nlp.spf.genlex.ccg.{ILexiconGenerator, LexiconGenerationServices}
import edu.cornell.cs.nlp.spf.learn.situated.AbstractSituatedLearner
import edu.cornell.cs.nlp.spf.learn.{ILearner, LearningStats}
import edu.cornell.cs.nlp.spf.parser.ccg.model.IDataItemModel
import edu.cornell.cs.nlp.spf.parser.joint.model.{IJointDataItemModel, IJointModelImmutable, JointModel}
import edu.cornell.cs.nlp.spf.parser.joint.{IJointDerivation, IJointOutputLogger, IJointParser}
import edu.cornell.cs.nlp.utils.log.LoggerFactory
import edu.cornell.cs.nlp.utils.system.MemoryReport

import scala.collection.JavaConverters._


object AbstractSituatedLearnerScala {
  val GOLD_LF_IS_MAX: String = "G"
  val HAS_VALID_LF: String = "V"
  val TRIGGERED_UPDATE: String = "U"
}

abstract class AbstractSituatedLearnerScala[SAMPLE <: ISituatedDataItem[Sentence, _],
                                        MR,
                                        ESTEP,
                                        ERESULT,
                                        DI <: ILabeledDataItem[SAMPLE, _]] private(
                                           val epochs: Int,
                                           val trainingData: IDataCollection[DI],
                                           val trainingDataDebug: java.util.Map[DI, edu.cornell.cs.nlp.utils.composites.Pair[MR, ERESULT]],
                                           val maxSentenceLength: Int,
                                           val lexiconGenerationBeamSize: Int,
                                           val parser: IJointParser[SAMPLE, MR, ESTEP, ERESULT],
                                           val parserOutputLogger: IJointOutputLogger[MR, ESTEP, ERESULT],
                                           val categoryServices: ICategoryServices[MR],
                                           val genlex: ILexiconGenerator[DI, MR, IJointModelImmutable[SAMPLE, MR, ESTEP]])
  extends ILearner[SAMPLE, DI, JointModel[SAMPLE, MR, ESTEP]] {

  type JModel = JointModel[SAMPLE, MR, ESTEP]
  type JointDerivation = IJointDerivation[MR, ERESULT]

  import AbstractSituatedLearnerScala._

  protected val stats: LearningStats = new LearningStats.Builder(trainingData.size())
    .addStat(HAS_VALID_LF, "Has a valid parse")
    .addStat(TRIGGERED_UPDATE, "Sample triggered update")
    .addStat(GOLD_LF_IS_MAX, "The best-scoring LF equals the provided GOLD debug LF")
    .setNumberStat("Number of new lexical entries added for sample")
    .build()

  val log = LoggerFactory.create(classOf[AbstractSituatedLearnerScala[SAMPLE, MR, ESTEP, ERESULT, DI]])

  override def train(model: JModel): Unit = {
    // Init GENLEX.
    log.info("Initializing GENLEX ...")
    genlex.init(model)

    // Epochs
    (1 to epochs).foreach { epochNumber =>
        // Training epoch, iterate over all training samples
        log.info("=========================")
        log.info(s"Training epoch $epochNumber")
        log.info("=========================")
        var itemCounter = -1

        // Iterating over training data
        for (dataItem <- trainingData.asScala) {
          // Process a single training sample

          // Record start time
          val startTime = System.currentTimeMillis

          // Log sample header
          itemCounter = itemCounter + 1
          log.info(s"[$itemCounter]: ================== [$epochNumber]")
          log.info("Sample type: %s", classOf[DI].getSimpleName)
          log.info("%s", dataItem)

          // Skip sample, if over the length limit
          if (dataItem.getSample.getSample.getTokens.size > maxSentenceLength)
            log.warn("Training sample too long, skipping")
          else {
            // Sample data item model
            val dataItemModel = model.createJointDataItemModel(dataItem.getSample)

            // ///////////////////////////
            // Step I: Generate a large number of potential lexical entries,
            // parse to prune them and update the lexicon.
            // ///////////////////////////
            lexicalInduction(dataItem, dataItemModel, model, itemCounter, epochNumber)

            // ///////////////////////////
            // Step II: Update model parameters.
            // ///////////////////////////
            parameterUpdate(dataItem, dataItemModel, model, itemCounter, epochNumber)

            // Record statistics
            stats.mean("sample processing", (System.currentTimeMillis - startTime) / 1000.0, "sec")
            stats.count("processed", epochNumber)
            log.info(s"Total sample handling time: ${(System.currentTimeMillis - startTime) / 1000.0}sec")
          }
        }

        // Output epoch statistics
        log.info(s"System memory: ${MemoryReport.generate}")
        log.info("Epoch stats:")
        log.info(stats)
    }
  }

  private def lexicalInduction(dataItem: DI,
                               dataItemModel: IJointDataItemModel[MR, ESTEP],
                               model: JModel,
                               dataItemNumber: Int,
                               epochNumber: Int): Unit = {
    // Generate lexical entries
    val generatedLexicon = genlex.generate(dataItem, model, categoryServices)
    log.info("Generated lexicon size = %d", generatedLexicon.size)

    if (generatedLexicon.size > 0) {
      // Case generated lexical entries

      // Parse with generated lexicon
      val generateLexiconParserOutput = parser.parse(dataItem.getSample, dataItemModel, false, generatedLexicon, lexiconGenerationBeamSize)

      // Log lexical generation parsing time
      stats.mean("genlex parse", generateLexiconParserOutput.getInferenceTime / 1000.0, "sec")
      log.info("Lexicon induction parsing time: %.4fsec", generateLexiconParserOutput.getInferenceTime / 1000.0)
      val output = if (generateLexiconParserOutput.isExact) "exact" else "approximate"
      log.info(s"Output is $output")

      // Log generation parser output
      parserOutputLogger.log(generateLexiconParserOutput, dataItemModel, s"$dataItemNumber-genlex")

      // Get lexical generation parses
      val generationParses = generateLexiconParserOutput.getDerivations.asScala
      log.info(s"Created ${generationParses.size} lexicon generation parses for training sample")

      // Use validation function to prune generation parses
      val generationParsesValidated = generationParses.filter(e => validate(dataItem, e.getResult))

      log.info(s"Removed ${generateLexiconParserOutput.getDerivations.size - generationParsesValidated.size} invalid parses")

      // Collect max scoring valid generation parses
      val bestGenerationParses = generationParsesValidated.foldLeft((List.empty[JointDerivation], -java.lang.Double.MAX_VALUE)) { (tuple, parse) =>
        val (acc, currentScore) = tuple
        if (parse.getViterbiScore > currentScore) (List(parse), parse.getViterbiScore)
        else if (parse.getViterbiScore == currentScore) (acc :+ parse, currentScore)
        else (acc, currentScore)
      }._1

      AbstractSituatedLearner.LOG.info(s"${bestGenerationParses.size} valid best parses for lexical generation:")
      bestGenerationParses.foreach(logParse(dataItem, _, valid = true, verbose = true, dataItemModel))

      // Проверить на эквивалетность джавовского код и переписать красивее
      // Update the model's lexicon with generated lexical
      // entries from the max scoring valid generation parses
      val newLexicalEntries = bestGenerationParses.foldLeft(0){ (newLexicalEntriesCounter, parse) =>
        newLexicalEntriesCounter + parse.getMaxLexicalEntries.asScala.foldLeft(0){ (innerCounter, entry) =>
          val linkedEntries = entry.getLinkedEntries.asScala.filter(linkedEntry => model.addLexEntry(LexiconGenerationServices.unmark(linkedEntry)))
          linkedEntries.foreach { linkedEntry =>
            AbstractSituatedLearner.LOG.info(s"Added (linked) LexicalEntry to model: $linkedEntry [${model.getTheta.printValues(model.computeFeatures(linkedEntry))}]")
          }
          if (model.addLexEntry(LexiconGenerationServices.unmark(entry))) {
            AbstractSituatedLearner.LOG.info(s"Added LexicalEntry to model: $entry [${model.getTheta.printValues(model.computeFeatures(entry))}]")
            innerCounter + linkedEntries.size + 1
          } else innerCounter + linkedEntries.size
        }
      }

      // Record statistics
      if (newLexicalEntries > 0) stats.appendSampleStat(dataItemNumber, epochNumber, newLexicalEntries)
    }
    else {
      // Skip lexical induction
      AbstractSituatedLearner.LOG.info("Skipped GENLEX step. No generated lexical items.")
    }
  }

  protected def isGoldDebugCorrect(dataItem: DI, label: ERESULT): Boolean =
    if (trainingDataDebug containsKey dataItem) (trainingDataDebug get dataItem) == label
    else false

  protected def logParse(dataItem: DI,
                         parse: JointDerivation,
                         valid: Boolean,
                         verbose: Boolean,
                         dataItemModel: IDataItemModel[MR]): Unit =
    logParse(dataItem, parse, valid, verbose, "", dataItemModel)

  protected def logParse(dataItem: DI,
                         parse: JointDerivation,
                         valid: Boolean,
                         verbose: Boolean,
                         tag: String,
                         dataItemModel: IDataItemModel[MR]): Unit = {
    val isGold = isGoldDebugCorrect(dataItem, parse.getResult)
    AbstractSituatedLearner.LOG.info(s"${if (isGold) "* " else "  "}${tag + " "}[${parse.getViterbiScore}${if (valid) ", V" else ", X"}] $parse")

    if (verbose) {
      parse.getMaxSteps.asScala.foreach { step =>
        AbstractSituatedLearner.LOG.info(s"\t${step.toString(false, false, dataItemModel.getTheta)}")
      }
    }
  }

  /**
    * Parameter update method.
    */
  protected def parameterUpdate(dataItem: DI, dataItemModel: IJointDataItemModel[MR, ESTEP], model: JointModel[SAMPLE, MR, ESTEP], itemCounter: Int, epochNumber: Int)

  protected def validate(dataItem: DI, hypothesis: ERESULT): Boolean
}