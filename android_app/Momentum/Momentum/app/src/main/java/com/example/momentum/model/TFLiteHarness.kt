// --- File Path: app/src/main/java/com/example/momentum/model/TFLiteHarness.kt ---
package com.example.momentum.model

import android.content.Context
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.flex.FlexDelegate
import org.tensorflow.lite.support.common.FileUtil
import java.nio.FloatBuffer

/**
 * A harness specifically for the trainable Float32 TFLite model.
 * It manages loading the model and its signatures for training and weight transfer.
 */
class TFLiteHarness(context: Context) {

    private val interpreter: Interpreter

    companion object {
        private const val TRAINABLE_MODEL_NAME = "tcn_v2_ABCDE_trainable.tflite"
        private const val INFER_SIGNATURE = "infer"
        private const val TRAIN_SIGNATURE = "train_step"
        private const val SET_WEIGHTS_SIGNATURE = "set_weights_flat"
        private const val GET_WEIGHTS_SIGNATURE = "get_weights_flat"
        const val NUM_CLASSES = 5 // Corresponds to A, B, C, D, E
        const val NUM_FLAT_WEIGHTS = 791569  // IMPORTANT: Must match your model's flattened weight count
    }

    init {
        val modelBuffer = FileUtil.loadMappedFile(context, TRAINABLE_MODEL_NAME)
        val options = Interpreter.Options().apply {
            addDelegate(FlexDelegate())
        }
        interpreter = Interpreter(modelBuffer, options)
    }

    fun getLogits(xBatch: FloatBuffer): FloatArray {
        xBatch.rewind()
        val inputs = mapOf("x_input" to xBatch)
        val outputs = mutableMapOf<String, Any>(
            "logits" to Array(1) { FloatArray(NUM_CLASSES) }
        )
        interpreter.runSignature(inputs, outputs, INFER_SIGNATURE)
        return (outputs["logits"] as Array<FloatArray>)[0]
    }

    fun setWeights(flatWeights: FloatArray) {
        val inputs = mapOf("flat_weights" to flatWeights)
        interpreter.runSignature(inputs, mutableMapOf(), SET_WEIGHTS_SIGNATURE)
    }

    fun getWeights(): FloatArray {
        val inputs = mapOf("dummy_input" to floatArrayOf(0.0f))
        val outputs = mutableMapOf<String, Any>("weights" to FloatArray(NUM_FLAT_WEIGHTS))
        interpreter.runSignature(inputs, outputs, GET_WEIGHTS_SIGNATURE)
        return outputs["weights"] as FloatArray
    }

    fun train(xBatch: FloatBuffer, yBatch: IntArray): Float {
        xBatch.rewind()
        val inputs = mapOf("x_input" to xBatch, "y_batch" to yBatch)
        val outputs = mutableMapOf<String, Any>("loss" to FloatArray(1))
        interpreter.runSignature(inputs, outputs, TRAIN_SIGNATURE)
        return (outputs["loss"] as FloatArray)[0]
    }

    fun close() {
        interpreter.close()
    }
}