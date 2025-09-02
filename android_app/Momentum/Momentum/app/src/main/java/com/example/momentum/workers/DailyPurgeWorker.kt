// --- File Path: app/src/main/java/com/example/momentum/workers/DailyPurgeWorker.kt ---
package com.example.momentum.workers

import android.content.Context
import android.util.Log
import androidx.work.CoroutineWorker
import androidx.work.WorkerParameters
import com.example.momentum.data.SensorDatabase
import com.example.momentum.data.SensorRepository
import java.util.concurrent.TimeUnit

class DailyPurgeWorker(
    appContext: Context,
    workerParams: WorkerParameters
): CoroutineWorker(appContext, workerParams) {

    override suspend fun doWork(): Result {
        return try {
            Log.i("DailyPurgeWorker", "Worker starting: Purging old raw sensor data.")
            val dao = SensorDatabase.getInstance(applicationContext).sensorDao()
            val windowDao = SensorDatabase.getInstance(applicationContext).labeledWindowDao()
            val repository = SensorRepository(dao, windowDao)

            // Keep raw data for 24 hours
            val retentionPeriodMs = TimeUnit.HOURS.toMillis(24)
            repository.purgeOldRawData(retentionPeriodMs)

            Log.i("DailyPurgeWorker", "Purge completed successfully.")
            Result.success()
        } catch (e: Exception) {
            Log.e("DailyPurgeWorker", "Error during purge", e)
            Result.failure()
        }
    }
}