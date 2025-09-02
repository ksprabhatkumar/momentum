// --- File Path: app/src/main/java/com/example/momentum/data/LabeledWindowDao.kt ---
package com.example.momentum.data

import androidx.room.*

@Dao
interface LabeledWindowDao {
    // ... (existing functions are unchanged)
    @Insert
    suspend fun insert(window: LabeledWindow)
    @Query("SELECT EXISTS(SELECT 1 FROM labeled_windows WHERE start_timestamp = :timestamp LIMIT 1)")
    suspend fun windowExists(timestamp: Long): Boolean
    @Query("SELECT * FROM labeled_windows ORDER BY start_timestamp ASC")
    suspend fun getAll(): List<LabeledWindow>
    @Query("SELECT SUM(LENGTH(window_data_json)) FROM labeled_windows")
    suspend fun getDatabaseSizeInBytes(): Long?
    @Query("SELECT id FROM labeled_windows ORDER BY start_timestamp ASC LIMIT :count")
    suspend fun getOldestWindowIds(count: Int): List<Long>
    @Query("DELETE FROM labeled_windows WHERE id IN (:ids)")
    suspend fun deleteWindowsByIds(ids: List<Long>)

    // NEW: Deletes windows where the start timestamp falls within the given range.
    @Query("DELETE FROM labeled_windows WHERE start_timestamp BETWEEN :fromTimestamp AND :toTimestamp")
    suspend fun deleteWindowsInRange(fromTimestamp: Long, toTimestamp: Long): Int // Returns number of rows deleted

    // NEW: Function to delete all labeled windows
    @Query("DELETE FROM labeled_windows")
    suspend fun clearAll()
}