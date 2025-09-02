// --- File Path: app/src/main/java/com/example/momentum/model/FederatedLearner.kt (FINAL VERSION) ---
package com.example.momentum.model

import android.content.Context
import android.util.Base64
import android.util.Log
import com.example.momentum.data.LabeledWindow
import com.example.momentum.data.SensorRepository
import com.example.momentum.net.NetworkService
import com.google.gson.Gson
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import kotlin.system.measureTimeMillis

data class FLState(
    val isRunning: Boolean = false,
    val log: String = "Federated Learning is idle. Press start to begin."
)

// Private helper data class for training results
private data class TrainingResult(
    val trainingTimeMs: Long,
    val sampleCount: Int
)

/**
 * A class dedicated to managing the entire Federated Learning lifecycle.
 * It is created and managed by the MainViewModel but contains no UI logic itself.
 */
class FederatedLearner(
    private val context: Context,
    private val repository: SensorRepository,
    private val clientId: String
) {
    private val harness = TFLiteHarness(context)
    private val gson = Gson()

    private val _flState = MutableStateFlow(FLState())
    val flState = _flState.asStateFlow()

    private val logMessages = mutableListOf<String>()
    private var flJob: Job? = null

    // --- REMOVED ---
    // The static testSamples list and loadTestDataForDiagnostics() are no longer needed.

    private fun log(message: String) {
        synchronized(logMessages) {
            Log.d("FederatedLearner", message)
            logMessages.add(message)
            _flState.value = _flState.value.copy(log = logMessages.joinToString("\n"))
        }
    }

    suspend fun runFullFLProcess() {
        if (flJob?.isActive == true) return
        _flState.value = FLState(isRunning = true, log = "Starting...")
        logMessages.clear()

        log("--- Starting Federated Learning Process ---")
        log("Client ID: $clientId")

        while (_flState.value.isRunning) {
            val checkInResult = NetworkService.checkIn(clientId)
            if (checkInResult.isFailure) {
                log("❌ Failed to communicate with server. Halting. Error: ${checkInResult.exceptionOrNull()?.message}")
                break
            }

            val serverResponse = checkInResult.getOrThrow()
            when (serverResponse.action) {
                "TRAIN" -> {
                    log("\n== Round ${serverResponse.round}: Training ==")
                    val weightsB64 = serverResponse.data?.weights
                    if (weightsB64 == null) {
                        log("Error: Server sent no weights. Retrying in 15s...")
                        delay(15000)
                        continue
                    }
                    harness.setWeights(decodeWeights(weightsB64))
                    log("   > Global model received and set.")

                    val localEpochs = serverResponse.data?.config?.local_epochs ?: 1 // Default to 1 if not provided
                    log("   > Server configured training for $localEpochs local epoch(s).")

                    // --- MODIFIED: The entire training and evaluation flow is now self-contained ---
                    val (scalerMean, scalerScale) = try {
                        parseScaler(context.assets.open("scaler.json").bufferedReader().readText())
                    } catch (e: Exception) {
                        log("   > CRITICAL FAILED: Could not load scaler.json. Halting round.")
                        delay(15000)
                        continue
                    }

                    val allLocalData = repository.getLabeledWindows().shuffled().take(1000)
                    if (allLocalData.size < 10) { // Require a minimum amount of data
                        log("   > Not enough local data (found ${allLocalData.size}). Waiting...")
                        delay(10000)
                        continue
                    }

                    val splitIndex = (allLocalData.size * 0.8).toInt()
                    val trainingData = allLocalData.subList(0, splitIndex)
                    val evaluationData = allLocalData.subList(splitIndex, allLocalData.size)
                    log("   > Using ${allLocalData.size} local samples for this round.")
                    log("     - Training on:   ${trainingData.size} samples")
                    log("     - Evaluating on: ${evaluationData.size} samples")

                    val (preLoss, preAccuracy) = evaluateOnLocalData(evaluationData, scalerMean, scalerScale)
                    log(String.format("   > Pre-Training Eval:  Loss=%.4f, Acc=%.2f%%", preLoss, preAccuracy * 100))


                    // --- MODIFICATION: Pass localEpochs to the training function ---
                    val trainingResult = trainOnLocalData(trainingData, scalerMean, scalerScale, localEpochs)

                    log("   > Local training on ${trainingResult.sampleCount} samples complete in ${trainingResult.trainingTimeMs}ms.")

                    val (postLoss, postAccuracy) = evaluateOnLocalData(evaluationData, scalerMean, scalerScale)
                    log(String.format("   > Post-Training Eval: Loss=%.4f, Acc=%.2f%%", postLoss, postAccuracy * 100))

                    val metrics = ClientMetrics(
                        num_samples = trainingResult.sampleCount,
                        training_time_ms = trainingResult.trainingTimeMs,
                        pre_eval_loss = preLoss,
                        pre_eval_accuracy = preAccuracy,
                        post_eval_loss = postLoss,
                        post_eval_accuracy = postAccuracy
                    )

                    val submitResult = NetworkService.submitWeights(clientId, harness.getWeights(), metrics)
                    if (submitResult.isSuccess) {
                        log("   > Updated weights and metrics submitted.")
                    } else {
                        log("❌ Failed to submit weights. Retrying check-in...")
                    }
                }
                "WAIT" -> {
                    log("Server is busy. Checking again in 15s...")
                    delay(15000)
                }
                "COMPLETE" -> {
                    log("\n🏁 Federated Learning process complete!")
                    break
                }
                else -> {
                    log("Unknown server action: ${serverResponse.action}. Halting.")
                    break
                }
            }
        }
        _flState.value = FLState(isRunning = false, log = logMessages.joinToString("\n"))
    }

    private fun trainOnLocalData(
        trainingData: List<LabeledWindow>,
        mean: FloatArray,
        scale: FloatArray,
        epochs: Int // <-- NEW
    ): TrainingResult {
        if (trainingData.isEmpty() || epochs < 1) return TrainingResult(0L, 0)

        val labelMap = mapOf("A" to 0, "B" to 1, "C" to 2, "D" to 3, "E" to 4)
        val time = measureTimeMillis {
            // --- MODIFICATION: Loop for the specified number of epochs ---
            repeat(epochs) { epoch ->
                log("     > Starting local epoch ${epoch + 1}/$epochs...")
                trainingData.forEach { window ->
                    val windowData = try {
                        gson.fromJson(window.window_data_json, Array<Array<Float>>::class.java)
                    } catch (e: Exception) {
                        log("   > TRAIN Warning: Skipping malformed window data (ID: ${window.id})")
                        return@forEach
                    }
                    labelMap[window.label]?.let { labelIndex ->
                        val (xBuffer, yArray) = preprocess(windowData, labelIndex, mean, scale)
                        harness.train(xBuffer, yArray)
                    }
                }
            }
        }
        return TrainingResult(time, trainingData.size)
    }

    // --- NEW: Replaces the old runEvaluationOnTestData function ---
    private fun evaluateOnLocalData(
        evaluationData: List<LabeledWindow>,
        mean: FloatArray,
        scale: FloatArray
    ): Pair<Float, Float> {
        if (evaluationData.isEmpty()) return Pair(Float.MAX_VALUE, 0f)

        var totalLoss = 0.0f
        var correctPredictions = 0
        val labelMap = mapOf("A" to 0, "B" to 1, "C" to 2, "D" to 3, "E" to 4)

        evaluationData.forEach { window ->
            val windowData = try {
                gson.fromJson(window.window_data_json, Array<Array<Float>>::class.java)
            } catch (e: Exception) {
                log("   > EVAL Warning: Skipping malformed window data (ID: ${window.id})")
                return@forEach
            }
            labelMap[window.label]?.let { trueLabel ->
                val (inputBuffer, _) = preprocess(windowData, trueLabel, mean, scale)
                val probabilities = harness.getLogits(inputBuffer)

                val correctClassProbability = probabilities.getOrElse(trueLabel) { 0f }
                totalLoss += -kotlin.math.log((correctClassProbability + 1e-9f).toDouble(), kotlin.math.E).toFloat()

                val predictedClass = probabilities.indices.maxByOrNull { probabilities[it] } ?: -1
                if (predictedClass == trueLabel) correctPredictions++
            }
        }
        val averageLoss = if (evaluationData.isNotEmpty()) totalLoss / evaluationData.size else 0f
        val accuracy = if (evaluationData.isNotEmpty()) correctPredictions.toFloat() / evaluationData.size else 0f
        return Pair(averageLoss, accuracy)
    }

    // --- Utility and Helper functions (unchanged) ---
    private fun decodeWeights(base64String: String): FloatArray {
        val bytes = Base64.decode(base64String, Base64.DEFAULT)
        val byteBuffer = ByteBuffer.wrap(bytes).order(ByteOrder.nativeOrder())
        val floatBuffer = byteBuffer.asFloatBuffer()
        val floatArray = FloatArray(floatBuffer.remaining())
        floatBuffer.get(floatArray)
        return floatArray
    }

    private fun preprocess(data: Array<Array<Float>>, label: Int, mean: FloatArray, scale: FloatArray): Pair<FloatBuffer, IntArray> {
        val windowSize = 60
        val numFeatures = 6
        val xBuffer = ByteBuffer.allocateDirect(windowSize * numFeatures * 4)
            .order(ByteOrder.nativeOrder()).asFloatBuffer()
        for (i in 0 until windowSize) {
            for (j in 0 until numFeatures) {
                val scaledValue = (data[i][j] - mean[j]) / scale[j]
                xBuffer.put(scaledValue)
            }
        }
        xBuffer.rewind()
        return Pair(xBuffer, intArrayOf(label))
    }

    private fun parseScaler(jsonString: String): Pair<FloatArray, FloatArray> {
        val json = org.json.JSONObject(jsonString)
        val meanArray = json.getJSONArray("mean")
        val scaleArray = json.getJSONArray("scale")
        val mean = FloatArray(meanArray.length()) { meanArray.getDouble(it).toFloat() }
        val scale = FloatArray(scaleArray.length()) { scaleArray.getDouble(it).toFloat() }
        return Pair(mean, scale)
    }

    fun cancel() {
        _flState.value = FLState(isRunning = false, log = "Process cancelled by user.")
        flJob?.cancel()
        flJob = null
    }

    fun close() {
        harness.close()
    }
}