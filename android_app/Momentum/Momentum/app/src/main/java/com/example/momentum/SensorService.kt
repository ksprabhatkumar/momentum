// --- File Path: app/src/main/java/com/example/momentum/SensorService.kt ---
package com.example.momentum

import android.app.*
import android.content.Context
import android.content.Intent
import android.content.SharedPreferences
import android.content.pm.ServiceInfo
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Build
import android.os.IBinder
import android.util.Log
import androidx.core.app.NotificationCompat
import androidx.localbroadcastmanager.content.LocalBroadcastManager
import com.example.momentum.data.SensorData
import com.example.momentum.data.SensorDatabase
import com.example.momentum.net.NetworkService
import kotlinx.coroutines.*
import org.json.JSONObject
import org.tensorflow.lite.Interpreter
import java.io.File
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.concurrent.atomic.AtomicReference

class SensorService : Service(), SensorEventListener {

    private lateinit var sensorManager: SensorManager
    private lateinit var db: SensorDatabase
    private val serviceJob = Job()
    private val serviceScope = CoroutineScope(Dispatchers.IO + serviceJob)
    private lateinit var prefs: SharedPreferences

    // ML resources are now nullable and loaded dynamically
    private var interpreter: Interpreter? = null
    private var labels: List<String> = emptyList()
    private var scaler: Pair<FloatArray, FloatArray>? = null // This will be loaded dynamically

    private val SENSOR_DELAY = SensorManager.SENSOR_DELAY_GAME
    private val latestAccel = AtomicReference<FloatArray>(null)
    private val latestGyro = AtomicReference<FloatArray>(null)
    private var dataProcessingJob: Job? = null

    private val windowBuffer = ArrayDeque<FloatArray>(WINDOW_SIZE)
    private var readingCountSinceLastInference = 0

    companion object {
        const val WINDOW_SIZE = 60
        const val INFERENCE_STRIDE = 5
        const val NUM_FEATURES = 6
        const val NOTIFICATION_CHANNEL_ID = "SensorServiceChannel"
        const val NOTIFICATION_ID = 1
        const val ACTION_BROADCAST_PREDICTION = "com.example.momentum.PREDICTION"
        const val EXTRA_PREDICTION = "extra_prediction"
        private const val PROCESSING_INTERVAL_MS = 50L
    }

    override fun onCreate() {
        super.onCreate()
        sensorManager = getSystemService(SENSOR_SERVICE) as SensorManager
        db = SensorDatabase.getInstance(applicationContext)
        prefs = applicationContext.getSharedPreferences("MomentumPrefs", Context.MODE_PRIVATE)
        createNotificationChannel()
        Log.i("SensorService", "Service onCreate")
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        Log.i("SensorService", "Service onStartCommand")

        val notification = createNotification("Initializing...")
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            startForeground(NOTIFICATION_ID, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC)
        } else {
            startForeground(NOTIFICATION_ID, notification)
        }

        serviceScope.launch {
            try {
                loadResources() // Now combines model, scaler, and labels loading
                Log.d("SensorService", "ML resources loaded. Interpreter: ${interpreter != null}, Scaler: ${scaler != null}, Labels size: ${labels.size}")
                if (interpreter == null || scaler == null || labels.isEmpty()) {
                    throw IllegalStateException("Model, scaler, or labels failed to load correctly.")
                }
                registerSensors()
                startDataProcessing()
            } catch (e: Exception) {
                Log.e("SensorService", "Failed to initialize service resources", e)
                updateNotification("Error: Service failed to start")
                stopSelf()
            }
        }
        return START_STICKY
    }

    // Combined function to load model, scaler, and labels
    private fun loadResources() {
        val context = applicationContext
        val customModelFile = File(context.filesDir, "custom_inference_model.tflite")
        val customScalerFile = File(context.filesDir, "custom_scaler.json")

        var modelLoaded = false
        var scalerLoaded = false

        // --- UPDATED LOGIC ---
        // 1. Try to load custom model/scaler from internal storage first.
        if (customModelFile.exists() && customScalerFile.exists()) {
            Log.d("SensorService", "Found custom model and scaler in internal storage. Attempting to load.")
            try {
                // Load custom model
                FileInputStream(customModelFile).channel.use { channel ->
                    interpreter = Interpreter(channel.map(FileChannel.MapMode.READ_ONLY, 0, channel.size()))
                }
                Log.d("SensorService", "Custom inference model loaded successfully from ${customModelFile.path}.")
                modelLoaded = true

                // Load custom scaler
                scaler = parseScaler(customScalerFile.readText())
                Log.d("SensorService", "Custom scaler parsed successfully from ${customScalerFile.path}.")
                scalerLoaded = true
            } catch (e: Exception) {
                Log.e("SensorService", "Error loading custom ML resources, falling back to assets.", e)
                interpreter = null // Reset on failure
                scaler = null
            }
        }

        // 2. If custom loading failed or files didn't exist, load default from assets.
        if (!modelLoaded || !scalerLoaded) {
            Log.d("SensorService", "Loading default inference model from assets.")
            try {
                assets.openFd("inference_model_f16.tflite").use { fileDescriptor ->
                    FileInputStream(fileDescriptor.fileDescriptor).channel.use { channel ->
                        interpreter = Interpreter(channel.map(FileChannel.MapMode.READ_ONLY, fileDescriptor.startOffset, fileDescriptor.declaredLength))
                    }
                }
                Log.d("SensorService", "Default inference model loaded from assets.")

                assets.open("scaler.json").bufferedReader().use { reader ->
                    scaler = parseScaler(reader.readText())
                }
                Log.d("SensorService", "Default scaler loaded from assets.")
            } catch (e: Exception) {
                Log.e("SensorService", "CRITICAL: Error loading default ML resources from assets.", e)
            }
        }

        // 3. Labels are always loaded from assets.
        try {
            assets.open("labels.json").bufferedReader().use { reader ->
                val jsonLabels = JSONObject(reader.readText())
                labels = (0 until jsonLabels.length()).map { i -> jsonLabels.getString(i.toString()) }
            }
            Log.d("SensorService", "Labels loaded: ${labels.joinToString()}")
        } catch (e: Exception) {
            Log.e("SensorService", "CRITICAL: Error loading labels.json.", e)
            labels = emptyList()
        }
    }

    private fun parseScaler(jsonString: String): Pair<FloatArray, FloatArray> {
        return try {
            Log.d("SensorService", "Parsing scaler JSON: ${jsonString.take(100)}...") // Log first 100 chars
            val json = JSONObject(jsonString)
            val meanArray = json.getJSONArray("mean")
            val scaleArray = json.getJSONArray("scale")
            val mean = FloatArray(meanArray.length()) { meanArray.getDouble(it).toFloat() }
            val scale = FloatArray(scaleArray.length()) { scaleArray.getDouble(it).toFloat() }
            Log.d("SensorService", "Scaler parsed: Mean size ${mean.size}, Scale size ${scale.size}")
            Pair(mean, scale)
        } catch (e: Exception) {
            Log.e("SensorService", "Error parsing scaler JSON: ${e.message}", e)
            // Return a default/empty scaler on failure to prevent crash, but indicate error
            // Assumes NUM_FEATURES from Companion object is the correct size.
            Pair(FloatArray(NUM_FEATURES) { 0.0f }, FloatArray(NUM_FEATURES) { 1.0f })
        }
    }

    private fun registerSensors() {
        val accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        val gyroscope = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
        sensorManager.registerListener(this, accelerometer, SENSOR_DELAY)
        sensorManager.registerListener(this, gyroscope, SENSOR_DELAY)
        Log.d("SensorService", "Sensors registered.")
    }

    override fun onSensorChanged(event: SensorEvent) {
        when (event.sensor.type) {
            Sensor.TYPE_ACCELEROMETER -> latestAccel.set(event.values.clone())
            Sensor.TYPE_GYROSCOPE -> latestGyro.set(event.values.clone())
        }
    }

    private fun startDataProcessing() {
        updateNotification("Sampling data at 20 Hz...")
        dataProcessingJob?.cancel()
        dataProcessingJob = serviceScope.launch {
            while (isActive) {
                val currentAccel = latestAccel.get()
                val currentGyro = latestGyro.get()

                if (currentAccel != null && currentGyro != null) {
                    val combinedData = floatArrayOf(
                        currentAccel[0], currentAccel[1], currentAccel[2],
                        currentGyro[0], currentGyro[1], currentGyro[2]
                    )
                    handleLiveInference(combinedData)
                    db.sensorDao().insert(SensorData(timestamp = System.currentTimeMillis(), xAccel = combinedData[0], yAccel = combinedData[1], zAccel = combinedData[2], xGyro = combinedData[3], yGyro = combinedData[4], zGyro = combinedData[5]))
                }
                delay(PROCESSING_INTERVAL_MS)
            }
        }
        Log.d("SensorService", "Data processing job started.")
    }

    private fun handleLiveInference(dataPoint: FloatArray) {
        if (interpreter == null || scaler == null || labels.isEmpty()) {
            Log.w("SensorService", "Inference skipped: ML resources not ready (Interpreter: ${interpreter != null}, Scaler: ${scaler != null}, Labels empty: ${labels.isEmpty()})")
            return
        }
        if (windowBuffer.size == WINDOW_SIZE) windowBuffer.removeFirst()
        windowBuffer.addLast(dataPoint)
        readingCountSinceLastInference++
        if (windowBuffer.size == WINDOW_SIZE && readingCountSinceLastInference >= INFERENCE_STRIDE) {
            readingCountSinceLastInference = 0
            runInference()
        }
    }

    private fun runInference() {
        val currentInterpreter = interpreter ?: run { Log.e("SensorService", "Interpreter is null in runInference, cannot run."); return }
        val (mean, scale) = scaler ?: run { Log.e("SensorService", "Scaler is null in runInference, cannot run."); return }
        if (labels.isEmpty()) { Log.e("SensorService", "Labels are empty in runInference, cannot determine prediction."); return }

        // Ensure windowBuffer has enough elements before attempting to access them
        if (windowBuffer.size < WINDOW_SIZE) {
            Log.w("SensorService", "Window buffer not full (${windowBuffer.size}/${WINDOW_SIZE}), cannot run inference.")
            return
        }

        val inputArray = Array(1) { Array(WINDOW_SIZE) { i -> FloatArray(NUM_FEATURES) { j ->
            val value = try {
                windowBuffer[i][j]
            } catch (e: IndexOutOfBoundsException) {
                Log.e("SensorService", "Array index out of bounds during input preparation: i=$i, j=$j, Buffer size: ${windowBuffer.size}", e)
                0.0f
            }
            // Ensure mean and scale arrays are of sufficient size before accessing j
            if (j < mean.size && j < scale.size && scale[j] != 0f) {
                (value - mean[j]) / scale[j]
            } else {
                Log.e("SensorService", "Mean or Scale array size mismatch or scale factor is zero at index $j.")
                value // Use unscaled value or handle as error
            }
        }}}
        val outputArray = Array(1) { FloatArray(labels.size) }

        try {
            Log.d("SensorService", "Running interpreter with input shape: ${inputArray[0].size}x${inputArray[0][0].size}")
            currentInterpreter.run(inputArray, outputArray)
            Log.d("SensorService", "Inference output (first row): ${outputArray[0].joinToString()}")

            val predictedIndex = outputArray[0].indices.maxByOrNull { outputArray[0][it] } ?: -1
            Log.d("SensorService", "Predicted index raw: $predictedIndex")

            if (predictedIndex != -1 && predictedIndex < labels.size) {
                val prediction = labels[predictedIndex]
                Log.d("SensorService", "Final Prediction: $prediction (Index: $predictedIndex)")
                serviceScope.launch {
                    withContext(Dispatchers.Main) {
                        broadcastPrediction(prediction)
                        updateNotification("Live Activity: $prediction")
                    }
                    if (prefs.getBoolean("share_predictions", false)) {
                        NetworkService.postLivePrediction(prediction)
                    }
                }
            } else {
                Log.w("SensorService", "Prediction condition not met. Predicted Index: $predictedIndex, Labels Size: ${labels.size}. Output might be invalid or labels are missing.")
            }
        } catch (e: Exception) {
            Log.e("SensorService", "Inference failed: ${e.message}", e)
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        dataProcessingJob?.cancel()
        sensorManager.unregisterListener(this)
        serviceJob.cancel()
        interpreter?.close()
        Log.i("SensorService", "Service fully destroyed.")
    }

    private fun broadcastPrediction(prediction: String) {
        val intent = Intent(ACTION_BROADCAST_PREDICTION).apply {
            putExtra(EXTRA_PREDICTION, prediction)
        }
        LocalBroadcastManager.getInstance(this).sendBroadcast(intent)
    }

    private fun createNotification(text: String): Notification {
        return NotificationCompat.Builder(this, NOTIFICATION_CHANNEL_ID)
            .setContentTitle("Momentum Active")
            .setContentText(text)
            .setSmallIcon(R.mipmap.ic_launcher)
            .setOngoing(true)
            .build()
    }

    private fun updateNotification(text: String) {
        val notificationManager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        notificationManager.notify(NOTIFICATION_ID, createNotification(text))
    }

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val serviceChannel = NotificationChannel(
                NOTIFICATION_CHANNEL_ID,
                "Sensor Service Channel",
                NotificationManager.IMPORTANCE_LOW
            )
            getSystemService(NotificationManager::class.java).createNotificationChannel(serviceChannel)
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) { /* Not used */ }
    override fun onBind(intent: Intent?): IBinder? = null
}