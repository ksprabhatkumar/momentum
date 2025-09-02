// --- File Path: app/src/main/java/com/example/momentum/model/ClientMetrics.kt ---
package com.example.momentum.model

/**
 * Data class holding the performance metrics of a local training round,
 * to be sent to the federation server.
 */
data class ClientMetrics(
    val num_samples: Int,
    val training_time_ms: Long,
    val pre_eval_loss: Float?,
    val pre_eval_accuracy: Float?,
    val post_eval_loss: Float?,
    val post_eval_accuracy: Float?
)