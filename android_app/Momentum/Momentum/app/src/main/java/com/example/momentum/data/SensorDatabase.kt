// --- File Path: app/src/main/java/com/example/momentum/data/SensorDatabase.kt ---
package com.example.momentum.data

import android.content.Context
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase

// REFACTORED: Added LabeledWindow entity, incremented version.
@Database(entities = [SensorData::class, LabeledWindow::class], version = 2)
abstract class SensorDatabase : RoomDatabase() {
    abstract fun sensorDao(): SensorDao
    abstract fun labeledWindowDao(): LabeledWindowDao // NEW DAO accessor

    companion object {
        @Volatile private var INSTANCE: SensorDatabase? = null

        fun getInstance(context: Context): SensorDatabase {
            return INSTANCE ?: synchronized(this) {
                val instance = Room.databaseBuilder(
                    context.applicationContext,
                    SensorDatabase::class.java,
                    "sensor_db"
                )
                    // In a real production app, you would add a proper migration path.
                    // For this refactor, falling back is the simplest way to apply the schema change.
                    .fallbackToDestructiveMigration()
                    .build()
                INSTANCE = instance
                instance
            }
        }
    }
}