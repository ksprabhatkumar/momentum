// --- File Path: app/src/main/java/com/example/momentum/data/SensorData.kt ---
package com.example.momentum.data

import androidx.room.Entity
import androidx.room.PrimaryKey

/**
 * REFACTORED: This is now purely for raw, unlabeled sensor data ("Database A").
 * The `label` column has been removed.
 */
@Entity(tableName = "sensor_data")
data class SensorData(
    @PrimaryKey(autoGenerate = true) val id: Long = 0L,
    val timestamp: Long,
    val xAccel: Float,
    val yAccel: Float,
    val zAccel: Float,
    val xGyro: Float,
    val yGyro: Float,
    val zGyro: Float
)