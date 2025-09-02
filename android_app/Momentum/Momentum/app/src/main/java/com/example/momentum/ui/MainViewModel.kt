// --- File Path: app/src/main/java/com/example/momentum/ui/MainViewModel.kt ---
package com.example.momentum.ui

import android.app.Application
import android.content.ContentValues
import android.content.Context
import android.os.Environment
import android.provider.MediaStore
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.example.momentum.data.SensorRepository
import com.example.momentum.model.FLState
import com.example.momentum.model.FederatedLearner
import com.example.momentum.net.ModelInfo
import com.example.momentum.net.NetworkService
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.text.SimpleDateFormat
import java.util.*

class MainViewModel(
    application: Application,
    private val repository: SensorRepository,
    private val federatedLearner: FederatedLearner
) : AndroidViewModel(application) {

    private val activityLabels = mapOf(
        "A" to "Walking", "B" to "Jogging", "C" to "Using Stairs",
        "D" to "Sitting", "E" to "Standing"
    )

    private val prefs = application.getSharedPreferences("MomentumPrefs", Context.MODE_PRIVATE)

    // --- State Management ---
    private val _livePrediction = MutableStateFlow("Waiting for data...")
    val livePrediction = _livePrediction.asStateFlow()

    private val _liveActivityName = MutableStateFlow("Waiting for data...")
    val liveActivityName = _liveActivityName.asStateFlow()

    private val _sharePredictions = MutableStateFlow(prefs.getBoolean("share_predictions", false))
    val sharePredictions = _sharePredictions.asStateFlow()

    private val _federationServerIp = MutableStateFlow(prefs.getString("federation_server_ip", "10.0.2.2") ?: "10.0.2.2")
    val federationServerIp = _federationServerIp.asStateFlow()

    private val _privateNodeIp = MutableStateFlow(prefs.getString("private_node_ip", "192.168.1.100") ?: "192.168.1.100")
    val privateNodeIp = _privateNodeIp.asStateFlow()

    private val _latestModelInfo = MutableStateFlow<ModelInfo?>(null)
    val latestModelInfo = _latestModelInfo.asStateFlow()

    // --- Federated Learning State & Actions ---
    val flState: StateFlow<FLState> = federatedLearner.flState

    fun startFederatedLearning() {
        // CORRECTED: Launch the entire process on a background thread pool (IO)
        viewModelScope.launch(Dispatchers.IO) {
            federatedLearner.runFullFLProcess()
        }
    }

    fun cancelFederatedLearning() {
        federatedLearner.cancel()
    }

    init {
        NetworkService.setFederationServerIp(_federationServerIp.value)
        NetworkService.setPrivateNodeIp(_privateNodeIp.value)
        checkRemoteModelVersion() // Check for new model on startup
    }

    override fun onCleared() {
        super.onCleared()
        federatedLearner.close()
    }

    // --- Live Data and Predictions ---

    fun updateLivePrediction(prediction: String) {
        _livePrediction.value = prediction
        _liveActivityName.value = activityLabels[prediction] ?: "Unknown Activity"
    }

    fun setSharePredictions(isEnabled: Boolean) {
        _sharePredictions.value = isEnabled
        viewModelScope.launch { prefs.edit().putBoolean("share_predictions", isEnabled).apply() }
    }

    // --- IP Address Configuration ---

    fun setFederationServerIp(ipAddress: String) {
        _federationServerIp.value = ipAddress
        viewModelScope.launch {
            prefs.edit().putString("federation_server_ip", ipAddress).apply()
            NetworkService.setFederationServerIp(ipAddress)
        }
    }

    fun setPrivateNodeIp(ipAddress: String) {
        _privateNodeIp.value = ipAddress
        viewModelScope.launch {
            prefs.edit().putString("private_node_ip", ipAddress).apply()
            NetworkService.setPrivateNodeIp(ipAddress)
        }
    }

    // --- Model Management ---

    fun checkRemoteModelVersion() {
        viewModelScope.launch(Dispatchers.IO) {
            _latestModelInfo.value = NetworkService.getLatestModelInfo()
        }
    }

    fun downloadLatestModel(callback: (Boolean) -> Unit) {
        viewModelScope.launch(Dispatchers.IO) {
            val context = getApplication<Application>().applicationContext
            val success = NetworkService.downloadLatestModel(context)
            if (success) {
                checkRemoteModelVersion() // Re-check version to update UI
            }
            withContext(Dispatchers.Main) { callback(success) }
        }
    }

    // --- Data Management (Labeling, Deleting, Exporting) ---

    fun applyWindowedLabels(fromTimeStr: String, toTimeStr: String, label: String, callback: (Int) -> Unit) {
        viewModelScope.launch(Dispatchers.IO) {
            var newWindowsAdded = 0
            try {
                val fromMillis = convertHmsToTodayTimestamp(fromTimeStr)
                val toMillis = convertHmsToTodayTimestamp(toTimeStr)
                if (fromMillis >= toMillis) {
                    withContext(Dispatchers.Main) { callback(-1) }
                    return@launch
                }

                val windowSizeSamples = 60
                val strideSamples = 3
                val rawData = repository.getDataInRange(fromMillis, toMillis)

                if (rawData.size < windowSizeSamples) {
                    withContext(Dispatchers.Main) { callback(0) }
                    return@launch
                }

                var currentIndex = 0
                while (currentIndex + windowSizeSamples <= rawData.size) {
                    val windowStart = rawData[currentIndex]
                    if (!repository.windowExists(windowStart.timestamp)) {
                        val windowData = rawData.subList(currentIndex, currentIndex + windowSizeSamples)
                        repository.saveLabeledWindow(windowData, label)
                        newWindowsAdded++
                    }
                    currentIndex += strideSamples
                }

                if (newWindowsAdded > 0) {
                    repository.enforceStorageLimit(maxSizeMb = 20)
                }
                withContext(Dispatchers.Main) { callback(newWindowsAdded) }
            } catch (e: Exception) {
                Log.e("MainViewModel", "Failed to apply windowed labels", e)
                withContext(Dispatchers.Main) { callback(-1) }
            }
        }
    }

    fun deleteLabeledWindowsInRange(fromTimeStr: String, toTimeStr: String, callback: (Int) -> Unit) {
        viewModelScope.launch(Dispatchers.IO) {
            try {
                val fromMillis = convertHmsToTodayTimestamp(fromTimeStr)
                val toMillis = convertHmsToTodayTimestamp(toTimeStr)
                if (fromMillis >= toMillis) {
                    withContext(Dispatchers.Main) { callback(-1) }
                    return@launch
                }
                val deletedCount = repository.deleteLabeledWindowsInRange(fromMillis, toMillis)
                withContext(Dispatchers.Main) { callback(deletedCount) }
            } catch (e: Exception) {
                Log.e("MainViewModel", "Failed to delete labeled windows", e)
                withContext(Dispatchers.Main) { callback(-1) }
            }
        }
    }

    fun exportRawDataToCsv(callback: (String) -> Unit) {
        viewModelScope.launch(Dispatchers.IO) {
            try {
                val allData = repository.getAllRawData()
                if (allData.isEmpty()) {
                    withContext(Dispatchers.Main) { callback("No data to export.") }
                    return@launch
                }

                val context = getApplication<Application>().applicationContext
                val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
                val fileName = "momentum_raw_data_$timestamp.csv"
                val contentValues = ContentValues().apply {
                    put(MediaStore.MediaColumns.DISPLAY_NAME, fileName)
                    put(MediaStore.MediaColumns.MIME_TYPE, "text/csv")
                    put(MediaStore.MediaColumns.RELATIVE_PATH, Environment.DIRECTORY_DOWNLOADS)
                }
                val resolver = context.contentResolver
                resolver.insert(MediaStore.Files.getContentUri("external"), contentValues)?.let { uri ->
                    resolver.openOutputStream(uri)?.bufferedWriter().use { writer ->
                        writer?.append("timestamp,x_accel,y_accel,z_accel,x_gyro,y_gyro,z_gyro\n")
                        allData.forEach { data ->
                            writer?.append("${data.timestamp},${data.xAccel},${data.yAccel},${data.zAccel},${data.xGyro},${data.yGyro},${data.zGyro}\n")
                        }
                    }
                    withContext(Dispatchers.Main) { callback("Exported successfully to Downloads/$fileName") }
                } ?: throw IllegalStateException("Failed to create new MediaStore record.")
            } catch (e: Exception) {
                Log.e("MainViewModel", "CSV Export failed", e)
                withContext(Dispatchers.Main) { callback("Export failed: ${e.message}") }
            }
        }
    }

    // --- Add this function inside your MainViewModel.kt class ---

    fun exportLabeledWindowsToJson(callback: (String) -> Unit) {
        viewModelScope.launch(Dispatchers.IO) {
            try {
                val allWindows = repository.getLabeledWindows()
                if (allWindows.isEmpty()) {
                    withContext(Dispatchers.Main) { callback("No labeled data to export.") }
                    return@launch
                }

                // Use Gson to convert the entire list to a JSON string
                val gson = com.google.gson.GsonBuilder().setPrettyPrinting().create()
                val jsonString = gson.toJson(allWindows)

                val context = getApplication<Application>().applicationContext
                val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
                val fileName = "momentum_labeled_data_$timestamp.json"

                val contentValues = ContentValues().apply {
                    put(MediaStore.MediaColumns.DISPLAY_NAME, fileName)
                    put(MediaStore.MediaColumns.MIME_TYPE, "application/json")
                    put(MediaStore.MediaColumns.RELATIVE_PATH, Environment.DIRECTORY_DOWNLOADS)
                }

                val resolver = context.contentResolver
                resolver.insert(MediaStore.Files.getContentUri("external"), contentValues)?.let { uri ->
                    resolver.openOutputStream(uri)?.bufferedWriter().use { writer ->
                        writer?.write(jsonString)
                    }
                    withContext(Dispatchers.Main) { callback("Exported successfully to Downloads/$fileName") }
                } ?: throw IllegalStateException("Failed to create new MediaStore record.")

            } catch (e: Exception) {
                Log.e("MainViewModel", "Labeled Data JSON Export failed", e)
                withContext(Dispatchers.Main) { callback("Export failed: ${e.message}") }
            }
        }
    }

    private fun convertHmsToTodayTimestamp(hms: String): Long {
        val parts = hms.split(":").map { it.toInt() }
        val cal = Calendar.getInstance().apply {
            set(Calendar.HOUR_OF_DAY, parts[0])
            set(Calendar.MINUTE, parts[1])
            set(Calendar.SECOND, parts[2])
            set(Calendar.MILLISECOND, 0)
        }
        return cal.timeInMillis
    }
}