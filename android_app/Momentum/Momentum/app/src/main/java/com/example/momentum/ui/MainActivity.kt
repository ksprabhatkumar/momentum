// --- File Path: app/src/main/java/com/example/momentum/ui/MainActivity.kt ---
package com.example.momentum.ui

import android.Manifest
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.core.content.ContextCompat
import androidx.localbroadcastmanager.content.LocalBroadcastManager
import com.example.momentum.App // Import the App class
import com.example.momentum.SensorService
import com.example.momentum.ui.theme.MomentumTheme // <-- THE MISSING IMPORT

class MainActivity : ComponentActivity() {

    private lateinit var viewModelFactory: ViewModelFactory
    private val viewModel: MainViewModel by viewModels { viewModelFactory }

    // Modern way to handle permission requests
    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        val permissionsGranted = permissions.entries.all { it.value }
        if (permissionsGranted) {
            // Permissions granted, start the service
            Log.d("MainActivity", "All necessary permissions granted.")
            startSensorCollectionService()
        } else {
            // Permissions denied
            Log.w("MainActivity", "Permissions were denied. The app's core functionality will not work.")
            Toast.makeText(this, "All permissions are required for the app to function.", Toast.LENGTH_LONG).show()
        }
    }

    private val predictionReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {
            val prediction = intent?.getStringExtra(SensorService.EXTRA_PREDICTION)
            if (prediction != null) {
                viewModel.updateLivePrediction(prediction)
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Get dependencies from the Application class
        val app = application as App
        viewModelFactory = ViewModelFactory(application, app.repository, app.federatedLearner)

        setContent {
            MomentumTheme { // This will now resolve correctly
                MainScreen(viewModel = viewModel)
            }
        }

        // Check for permissions after setting content, which is a common pattern.
        checkAndRequestPermissions()
    }

    override fun onResume() {
        super.onResume()
        val intentFilter = IntentFilter(SensorService.ACTION_BROADCAST_PREDICTION)
        LocalBroadcastManager.getInstance(this).registerReceiver(predictionReceiver, intentFilter)
    }

    override fun onPause() {
        super.onPause()
        LocalBroadcastManager.getInstance(this).unregisterReceiver(predictionReceiver)
    }

    // Function to check for and request necessary permissions
    private fun checkAndRequestPermissions() {
        val requiredPermissions = mutableListOf(
            Manifest.permission.ACTIVITY_RECOGNITION,
            Manifest.permission.BODY_SENSORS
        )

        // For Android 13 (API 33) and above, also need POST_NOTIFICATIONS for the foreground service
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            requiredPermissions.add(Manifest.permission.POST_NOTIFICATIONS)
        }

        // For Android 9 (API 28) and below, also need WRITE_EXTERNAL_STORAGE for CSV export
        // Note: The manifest already has maxSdkVersion="28" for this permission
        if (Build.VERSION.SDK_INT <= Build.VERSION_CODES.P) {
            requiredPermissions.add(Manifest.permission.WRITE_EXTERNAL_STORAGE)
        }


        val permissionsToRequest = requiredPermissions.filter {
            ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED
        }

        if (permissionsToRequest.isEmpty()) {
            // All permissions are already granted
            Log.d("MainActivity", "All necessary permissions are already granted.")
            startSensorCollectionService()
        } else {
            // Request the missing permissions
            Log.d("MainActivity", "Requesting missing permissions: $permissionsToRequest")
            permissionLauncher.launch(permissionsToRequest.toTypedArray())
        }
    }

    private fun startSensorCollectionService() {
        // Ensure service is not started if permissions were just denied
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.ACTIVITY_RECOGNITION) != PackageManager.PERMISSION_GRANTED) {
            Log.w("MainActivity", "Aborting service start, ACTIVITY_RECOGNITION permission is missing.")
            return
        }

        val serviceIntent = Intent(this, SensorService::class.java)
        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                startForegroundService(serviceIntent)
            } else {
                startService(serviceIntent)
            }
            Log.d("MainActivity", "Sensor service start command issued.")
        } catch (e: Exception) {
            Log.e("MainActivity", "Failed to start service", e)
            Toast.makeText(this, "Failed to start sensor service.", Toast.LENGTH_SHORT).show()
        }
    }
}