// --- File Path: app/src/main/java/com/example/momentum/ui/SettingsScreen.kt ---
package com.example.momentum.ui

import android.widget.Toast
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import java.text.SimpleDateFormat
import java.util.*

@Composable
fun SettingsScreen(viewModel: MainViewModel) {
    val context = LocalContext.current

    var centralServerIpInput by remember { mutableStateOf("") }
    var privateNodeIpInput by remember { mutableStateOf("") }

    val savedCentralServerIp by viewModel.federationServerIp.collectAsState()
    val savedPrivateNodeIp by viewModel.privateNodeIp.collectAsState()
    val sharePredictions by viewModel.sharePredictions.collectAsState()
    val modelInfo by viewModel.latestModelInfo.collectAsState()

    LaunchedEffect(savedCentralServerIp, savedPrivateNodeIp) {
        centralServerIpInput = savedCentralServerIp
        privateNodeIpInput = savedPrivateNodeIp
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp)
            .verticalScroll(rememberScrollState()),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        // Card 1: Network Configuration
        Card(
            elevation = CardDefaults.cardElevation(defaultElevation = 4.dp),
            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
        ) {
            Column(modifier = Modifier.padding(16.dp).fillMaxWidth(), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text("Network Configuration", style = MaterialTheme.typography.titleLarge)
                OutlinedTextField(
                    value = centralServerIpInput,
                    onValueChange = { centralServerIpInput = it },
                    label = { Text("Federation Server IP / URL") },
                    placeholder = { Text("e.g., my-tunnel.trycloudflare.com") },
                    modifier = Modifier.fillMaxWidth()
                )
                OutlinedTextField(
                    value = privateNodeIpInput,
                    onValueChange = { privateNodeIpInput = it },
                    label = { Text("Private Debug Node IP") },
                    modifier = Modifier.fillMaxWidth()
                )
                Button(
                    onClick = {
                        viewModel.setFederationServerIp(centralServerIpInput)
                        viewModel.setPrivateNodeIp(privateNodeIpInput)
                        Toast.makeText(context, "IP addresses updated", Toast.LENGTH_SHORT).show()
                    },
                    modifier = Modifier.align(Alignment.End)
                ) {
                    Text("Save IPs")
                }
            }
        }

        // Card 2: Central Server Management
        Card(
            elevation = CardDefaults.cardElevation(defaultElevation = 4.dp),
            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
        ) {
            Column(modifier = Modifier.padding(16.dp).fillMaxWidth(), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                Text("Model Management", style = MaterialTheme.typography.titleLarge)
                Divider(color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.2f))

                Text("On-Device Inference Model", style = MaterialTheme.typography.titleMedium)
                if (modelInfo != null) {
                    val formattedDate = SimpleDateFormat("MMM dd, yyyy HH:mm", Locale.getDefault())
                        .format(Date(modelInfo!!.timestamp))
                    Text("Latest remote version: ${modelInfo?.version}", fontWeight = FontWeight.Bold)
                    Text("Published on: $formattedDate", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                } else {
                    Text("Could not reach server to check for model updates.", color = MaterialTheme.colorScheme.onSurfaceVariant)
                }

                Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.End, verticalAlignment = Alignment.CenterVertically) {
                    OutlinedButton(
                        onClick = { viewModel.checkRemoteModelVersion() },
                        modifier = Modifier.padding(end = 8.dp)
                    ) {
                        Text("Check Again")
                    }
                    Button(onClick = {
                        Toast.makeText(context, "Downloading model...", Toast.LENGTH_SHORT).show()
                        viewModel.downloadLatestModel { success ->
                            val message = if (success) "Model updated successfully! Please restart the app for changes to take effect." else "Model download failed."
                            Toast.makeText(context, message, Toast.LENGTH_LONG).show()
                        }
                    }) {
                        Text("Download Latest")
                    }
                }
            }
        }

        // Card 3: General Data Management
        Card(
            elevation = CardDefaults.cardElevation(defaultElevation = 4.dp),
            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
        ) {
            Column(modifier = Modifier.fillMaxWidth()) {
                Row(
                    modifier = Modifier.fillMaxWidth().padding(start = 16.dp, end = 8.dp, top = 8.dp, bottom = 8.dp),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    Text("Share Live Predictions", style = MaterialTheme.typography.bodyLarge, modifier = Modifier.weight(1f))
                    Switch(checked = sharePredictions, onCheckedChange = { viewModel.setSharePredictions(it) })
                }
                Divider(modifier = Modifier.padding(horizontal = 16.dp), color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.2f))
                Row(
                    modifier = Modifier.fillMaxWidth().padding(start = 16.dp, end = 16.dp, top = 8.dp, bottom = 8.dp),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    Text("Export Raw Sensor Data", style = MaterialTheme.typography.bodyLarge)
                    OutlinedButton(onClick = {
                        Toast.makeText(context, "Exporting...", Toast.LENGTH_SHORT).show()
                        viewModel.exportRawDataToCsv { message -> Toast.makeText(context, message, Toast.LENGTH_LONG).show() }
                    }) { Text("Export CSV") }
                }
                Divider(modifier = Modifier.padding(horizontal = 16.dp), color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.2f))
                Row(
                    modifier = Modifier.fillMaxWidth().padding(start = 16.dp, end = 16.dp, top = 8.dp, bottom = 8.dp),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    Text("Export Labeled Data", style = MaterialTheme.typography.bodyLarge)
                    OutlinedButton(onClick = {
                        Toast.makeText(context, "Exporting labeled data...", Toast.LENGTH_SHORT).show()
                        viewModel.exportLabeledWindowsToJson { message -> Toast.makeText(context, message, Toast.LENGTH_LONG).show() }
                    }) { Text("Export JSON") }
                }
            }
        }
    }
}