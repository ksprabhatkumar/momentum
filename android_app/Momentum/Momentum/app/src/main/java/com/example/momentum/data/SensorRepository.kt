// --- File Path: app/src/main/java/com/example/momentum/data/SensorRepository.kt ---
package com.example.momentum.data

import com.google.gson.Gson
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileWriter
import java.text.SimpleDateFormat
import java.util.*
import kotlin.math.ceil

class SensorRepository(
    private val sensorDao: SensorDao,
    private val labeledWindowDao: LabeledWindowDao
) {
    private val gson = Gson()

    // --- Raw Data (Database A) Functions ---
    suspend fun insert(data: SensorData) = sensorDao.insert(data)
    suspend fun getDataInRange(from: Long, to: Long): List<SensorData> = sensorDao.getDataInRange(from, to)
    suspend fun purgeOldRawData(retentionPeriodMs: Long) {
        val purgeTimestamp = System.currentTimeMillis() - retentionPeriodMs
        sensorDao.purgeOlderThan(purgeTimestamp)
    }
    suspend fun getAllRawData(): List<SensorData> = sensorDao.getAll()

    // --- Labeled Window (Database B) Functions ---
    suspend fun getLabeledWindows(): List<LabeledWindow> = labeledWindowDao.getAll()

    suspend fun windowExists(timestamp: Long): Boolean = labeledWindowDao.windowExists(timestamp)

    suspend fun saveLabeledWindow(windowData: List<SensorData>, label: String) {
        if (windowData.isEmpty()) return
        val startTime = windowData.first().timestamp
        val dataAsFloatArray = windowData.map {
            floatArrayOf(it.xAccel, it.yAccel, it.zAccel, it.xGyro, it.yGyro, it.zGyro)
        }
        val jsonString = gson.toJson(dataAsFloatArray)
        val newWindow = LabeledWindow(
            start_timestamp = startTime,
            label = label,
            window_data_json = jsonString
        )
        labeledWindowDao.insert(newWindow)
    }

    suspend fun enforceStorageLimit(maxSizeMb: Int) {
        val targetSizeBytes = maxSizeMb * 1024 * 1024
        var currentSize = labeledWindowDao.getDatabaseSizeInBytes() ?: 0L

        if (currentSize <= targetSizeBytes) return

        // Get all windows and calculate average size to estimate how many to delete
        val allWindows = labeledWindowDao.getAll()
        if (allWindows.isEmpty()) return
        val avgSizePerWindow = currentSize / allWindows.size
        val overflow = currentSize - targetSizeBytes
        val windowsToDeleteCount = ceil(overflow.toDouble() / avgSizePerWindow).toInt()

        if (windowsToDeleteCount > 0) {
            val idsToDelete = labeledWindowDao.getOldestWindowIds(windowsToDeleteCount)
            labeledWindowDao.deleteWindowsByIds(idsToDelete)
        }
    }


    suspend fun deleteLabeledWindowsInRange(from: Long, to: Long): Int {
        return labeledWindowDao.deleteWindowsInRange(from, to)
    }


    suspend fun clearAllLabeledWindows() {
        labeledWindowDao.clearAll()
    }

}