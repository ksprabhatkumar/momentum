// --- File Path: app/src/main/java/com/example/momentum/data/SensorDao.kt ---
package com.example.momentum.data

import androidx.room.*

@Dao
interface SensorDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insert(data: SensorData)

    @Query("SELECT * FROM sensor_data ORDER BY timestamp ASC")
    suspend fun getAll(): List<SensorData>

    @Query("SELECT * FROM sensor_data WHERE timestamp BETWEEN :fromMillis AND :toMillis ORDER BY timestamp ASC")
    suspend fun getDataInRange(fromMillis: Long, toMillis: Long): List<SensorData>

    // NEW: Deletes all data older than the given timestamp.
    @Query("DELETE FROM sensor_data WHERE timestamp < :timestamp")
    suspend fun purgeOlderThan(timestamp: Long)

    // Unused, but can be kept for utility
    @Query("SELECT COUNT(*) FROM sensor_data")
    suspend fun count(): Int
}