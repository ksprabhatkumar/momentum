// --- File Path: app/src/main/java/com/example/momentum/App.kt ---
package com.example.momentum

import android.app.Application
import android.content.Context
import androidx.work.*
import com.example.momentum.data.SensorDatabase
import com.example.momentum.data.SensorRepository
import com.example.momentum.model.FederatedLearner
import com.example.momentum.workers.DailyPurgeWorker
import java.util.Calendar
import java.util.UUID
import java.util.concurrent.TimeUnit

class App : Application() {

    // --- Centralized dependencies ---
    val database: SensorDatabase by lazy { SensorDatabase.getInstance(this) }
    val repository: SensorRepository by lazy {
        SensorRepository(database.sensorDao(), database.labeledWindowDao())
    }


    // NEW: Centralized FL instance with a persistent client ID
    val federatedLearner: FederatedLearner by lazy {
        FederatedLearner(this, repository, getOrSetClientId())
    }

    override fun onCreate() {
        super.onCreate()
        scheduleDailyPurge()
    }

    private fun getOrSetClientId(): String {
        val prefs = getSharedPreferences("MomentumPrefs", Context.MODE_PRIVATE)
        var clientId = prefs.getString("fl_client_id", null)
        if (clientId == null) {
            clientId = "momentum-android-${UUID.randomUUID().toString().take(8)}"
            prefs.edit().putString("fl_client_id", clientId).apply()
        }
        return clientId
    }

    private fun scheduleDailyPurge() {
        val constraints = Constraints.Builder()
            .setRequiredNetworkType(NetworkType.NOT_REQUIRED)
            .build()

        val purgeRequest = PeriodicWorkRequestBuilder<DailyPurgeWorker>(1, TimeUnit.DAYS)
            .setConstraints(constraints)
            .setInitialDelay(calculateInitialDelay(), TimeUnit.MILLISECONDS)
            .build()

        WorkManager.getInstance(this).enqueueUniquePeriodicWork(
            "DailyDataPurge",
            ExistingPeriodicWorkPolicy.KEEP,
            purgeRequest
        )
    }

    private fun calculateInitialDelay(): Long {
        val calendar = Calendar.getInstance()
        val nowMillis = calendar.timeInMillis
        calendar.set(Calendar.HOUR_OF_DAY, 2)
        // ... rest of the function is unchanged
        calendar.set(Calendar.MINUTE, 0)
        calendar.set(Calendar.SECOND, 0)
        calendar.set(Calendar.MILLISECOND, 0)
        var targetMillis = calendar.timeInMillis
        if (targetMillis < nowMillis) {
            targetMillis += TimeUnit.DAYS.toMillis(1)
        }
        return targetMillis - nowMillis
    }
}