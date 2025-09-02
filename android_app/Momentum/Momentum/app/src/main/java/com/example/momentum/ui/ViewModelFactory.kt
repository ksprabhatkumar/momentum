// --- File Path: Momentum/app/src/main/java/com/example/momentum/ui/ViewModelFactory.kt ---
package com.example.momentum.ui

import android.app.Application
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import com.example.momentum.data.SensorRepository
import com.example.momentum.model.FederatedLearner

class ViewModelFactory(
    private val application: Application,
    private val repository: SensorRepository,
    private val federatedLearner: FederatedLearner // NEW
) : ViewModelProvider.Factory {

    @Suppress("UNCHECKED_CAST")
    override fun <T : ViewModel> create(modelClass: Class<T>): T {
        if (modelClass.isAssignableFrom(MainViewModel::class.java)) {
            // UPDATED: Pass the federated learner to the ViewModel constructor
            return MainViewModel(application, repository, federatedLearner) as T
        }
        throw IllegalArgumentException("Unknown ViewModel class")
    }
}