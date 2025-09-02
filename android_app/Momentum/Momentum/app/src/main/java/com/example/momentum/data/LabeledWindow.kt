// --- File Path: app/src/main/java/com/example/momentum/data/LabeledWindow.kt ---
package com.example.momentum.data

import androidx.room.Entity
import androidx.room.PrimaryKey

/**
 * Represents a single, labeled 60-sample window. This is "Database B".
 * The sensor readings for the window are stored as a JSON string.
 * This class is NOT sent over the network, so it does not need to be serializable.
 */
@Entity(tableName = "labeled_windows")
data class LabeledWindow(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val start_timestamp: Long, // Timestamp of the first reading in the window
    val label: String,
    val window_data_json: String, // A JSON representation of the 60x6 sensor readings
    val model_version: String = "v1.2" // To track which model this might be used for
)