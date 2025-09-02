// --- File Path: app/src/main/java/com/example/momentum/ui/FederatedScreen.kt ---
package com.example.momentum.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp

@Composable
fun FederatedScreen(viewModel: MainViewModel) {
    val flState by viewModel.flState.collectAsState()

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Card(
            elevation = CardDefaults.cardElevation(defaultElevation = 4.dp),
            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
        ) {
            Column(modifier = Modifier.padding(16.dp).fillMaxWidth(), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                Text("Federated Training", style = MaterialTheme.typography.titleLarge)
                Text(
                    "Contribute to the global model by training on your locally labeled data. Your raw data never leaves your device.",
                    style = MaterialTheme.typography.bodyMedium
                )
                Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Button(
                        onClick = { viewModel.startFederatedLearning() },
                        enabled = !flState.isRunning,
                        modifier = Modifier.weight(1f)
                    ) {
                        Text(if (flState.isRunning) "Running..." else "Start Training")
                    }
                    OutlinedButton(
                        onClick = { viewModel.cancelFederatedLearning() },
                        enabled = flState.isRunning,
                        modifier = Modifier.weight(1f)
                    ) {
                        Text("Cancel")
                    }
                }
            }
        }

        Spacer(Modifier.height(16.dp))

        if (flState.isRunning) {
            LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
        }

        Text(
            "Live Log",
            style = MaterialTheme.typography.titleMedium,
            modifier = Modifier.padding(vertical = 8.dp)
        )
        Box(
            modifier = Modifier
                .fillMaxSize()
                .background(MaterialTheme.colorScheme.onSurface.copy(alpha = 0.05f), shape = MaterialTheme.shapes.medium)
                .padding(8.dp)
        ) {
            val scrollState = rememberScrollState()
            // Auto-scroll to the bottom
            LaunchedEffect(flState.log) {
                scrollState.animateScrollTo(scrollState.maxValue)
            }
            Text(
                text = flState.log,
                fontFamily = FontFamily.Monospace,
                modifier = Modifier
                    .fillMaxSize()
                    .verticalScroll(scrollState)
            )
        }
    }
}