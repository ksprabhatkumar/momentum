// --- File Path: app/src/main/java/com/example/momentum/net/NetworkService.kt ---
package com.example.momentum.net

import android.content.Context
import android.util.Base64
import android.util.Log
import com.example.momentum.model.ClientMetrics
import io.ktor.client.*
import io.ktor.client.call.*
import io.ktor.client.engine.cio.*
import io.ktor.client.plugins.HttpTimeout
import io.ktor.client.plugins.contentnegotiation.*
import io.ktor.client.request.*
import io.ktor.client.statement.*
import io.ktor.http.*
import io.ktor.serialization.gson.*
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

// ... (Data Models like ClientCheckInRequest, etc., are unchanged) ...
data class ClientCheckInRequest(val client_id: String)
data class TrainingConfig(val local_epochs: Int?)
data class ServerResponse(val action: String, val round: Int?, val data: ServerData?)
data class ServerData(val weights: String?, val config: TrainingConfig?)
data class SubmitWeightsRequest(val client_id: String, val weights: String, val metrics: ClientMetrics)
data class ModelInfo(val version: String, val timestamp: Long)
data class LivePredictionPayload(val prediction: String)


object NetworkService {

    // Default values
    private var federationServerUrl = "http://10.0.2.2:8080"
    private var privateNodeUrl = "http://192.168.1.100:5000"

    private val client = HttpClient(CIO) {
        install(HttpTimeout) {
            requestTimeoutMillis = 60000
            connectTimeoutMillis = 10000
        }
        install(ContentNegotiation) {
            gson()
        }
    }

    // =================================================================================
    //                    NEW: ROBUST URL BUILDER HELPER FUNCTION
    // =================================================================================
    /**
     * Intelligently constructs a full URL from user input.
     * - If the input is a full URL, it's used as is.
     * - If it's a domain name (contains letters), it prepends "https://" and uses no port.
     * - If it's an IP address (only numbers and dots), it prepends "http://" and adds the default port.
     */
    private fun buildUrl(input: String, defaultPort: Int): String {
        val trimmedInput = input.trim()
        if (trimmedInput.isBlank()) return ""

        // Case 1: User provided a full URL. Use it directly.
        if (trimmedInput.startsWith("http://") || trimmedInput.startsWith("https://")) {
            return trimmedInput
        }

        // Heuristic: Check if the input contains any letters to differentiate a domain from an IP.
        val isDomain = trimmedInput.any { it.isLetter() }

        return if (isDomain) {
            // Case 2: Input is a domain (e.g., "my-tunnel.ngrok-free.app", "localhost").
            // Default to HTTPS for security, which is standard for services like ngrok.
            // Do not append a port, as standard ports (80/443) are implicit.
            "https://$trimmedInput"
        } else {
            // Case 3: Input is an IP address (e.g., "192.168.1.10", "10.0.2.2").
            // Default to HTTP for local network connections and append the default port.
            "http://$trimmedInput:$defaultPort"
        }
    }

    // =================================================================================
    //                    MODIFIED: Public Setter Functions
    // =================================================================================

    fun setFederationServerIp(urlOrIp: String) {
        if (urlOrIp.isNotBlank()) {
            federationServerUrl = buildUrl(urlOrIp, 8080)
            Log.d("NetworkService", "Federation Server URL set to: $federationServerUrl")
        }
    }

    fun setPrivateNodeIp(urlOrIp: String) {
        if (urlOrIp.isNotBlank()) {
            privateNodeUrl = buildUrl(urlOrIp, 5000)
            Log.d("NetworkService", "Private Node URL set to: $privateNodeUrl")
        }
    }


    // =================================================================================
    //                    ALL OTHER NETWORKING FUNCTIONS ARE UNCHANGED
    // =================================================================================

    suspend fun checkIn(clientId: String): Result<ServerResponse> {
        return try {
            val response: ServerResponse = client.post("$federationServerUrl/client-check-in") {
                contentType(ContentType.Application.Json)
                setBody(ClientCheckInRequest(client_id = clientId))
            }.body()
            Result.success(response)
        } catch (e: Exception) {
            Log.e("NetworkService", "FL Check-in failed", e)
            Result.failure(e)
        }
    }

    suspend fun submitWeights(clientId: String, weights: FloatArray, metrics: ClientMetrics): Result<Unit> {
        return try {
            val byteBuffer = ByteBuffer.allocate(weights.size * 4).order(ByteOrder.nativeOrder())
            byteBuffer.asFloatBuffer().put(weights)
            val weightsB64 = Base64.encodeToString(byteBuffer.array(), Base64.NO_WRAP)
            val requestBody = SubmitWeightsRequest(clientId, weightsB64, metrics)
            client.post("$federationServerUrl/submit-weights") {
                contentType(ContentType.Application.Json)
                setBody(requestBody)
            }
            Result.success(Unit)
        } catch (e: Exception) {
            Log.e("NetworkService", "FL Submit Weights failed", e)
            Result.failure(e)
        }
    }

    suspend fun getLatestModelInfo(): ModelInfo? {
        return try {
            client.get("$federationServerUrl/model_info").body<ModelInfo>()
        } catch (e: Exception) {
            Log.e("NetworkService", "Failed to get latest model info", e)
            null
        }
    }

    // ... (downloadLatestModel and postLivePrediction are also unchanged) ...
    suspend fun downloadLatestModel(context: Context): Boolean {
        return try {
            val modelResponse: HttpResponse = client.get("$federationServerUrl/download_model")
            if (!modelResponse.status.isSuccess()) return false
            val modelFile = File(context.filesDir, "custom_inference_model.tflite")
            modelFile.writeBytes(modelResponse.body())

            val scalerResponse: HttpResponse = client.get("$federationServerUrl/download_scaler_for_app")
            if (!scalerResponse.status.isSuccess()) { modelFile.delete(); return false }
            val scalerFile = File(context.filesDir, "custom_scaler.json")
            scalerFile.writeBytes(scalerResponse.body<ByteArray>())
            true
        } catch (e: Exception) {
            Log.e("NetworkService", "Model package download failed.", e)
            false
        }
    }

    suspend fun postLivePrediction(prediction: String) {
        try {
            client.post("$privateNodeUrl/live_prediction") {
                contentType(ContentType.Application.Json)
                setBody(LivePredictionPayload(prediction))
            }
        } catch (e: Exception) {
            Log.w("NetworkService", "Failed to post live prediction: ${e.message}")
        }
    }
}