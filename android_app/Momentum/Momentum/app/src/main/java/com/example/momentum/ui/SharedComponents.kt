// --- File Path: app/src/main/java/com/example/momentum/ui/SharedComponents.kt ---
package com.example.momentum.ui

import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp

@Composable
fun ConfigCard(serverIp: String, onIpChange: (String) -> Unit, onSetIp: () -> Unit) {
    Card(
        elevation = CardDefaults.cardElevation(defaultElevation = 4.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
    ) {
        Column(
            modifier = Modifier
                .padding(16.dp)
                .fillMaxWidth(),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Text(text = "Server Configuration", style = MaterialTheme.typography.titleLarge)
            OutlinedTextField(
                value = serverIp,
                onValueChange = onIpChange,
                label = { Text("Server IP Address") },
                modifier = Modifier.fillMaxWidth()
            )
            Button(onClick = onSetIp, modifier = Modifier.align(Alignment.End)) {
                Text("Set IP")
            }
        }
    }
}

@Composable
fun LabelingCard(
    fromTime: String, toTime: String, label: String,
    onFromTimeChange: (String) -> Unit,
    onToTimeChange: (String) -> Unit,
    onLabelChange: (String) -> Unit,
    onApplyLabel: () -> Unit
) {
    Card(
        elevation = CardDefaults.cardElevation(defaultElevation = 4.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
    ) {
        Column(
            modifier = Modifier
                .padding(16.dp)
                .fillMaxWidth(),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Text(text = "Label Data Segment", style = MaterialTheme.typography.titleLarge)
            OutlinedTextField(
                value = fromTime,
                onValueChange = onFromTimeChange,
                label = { Text("From Time (HH:MM:SS)") },
                placeholder = { Text("e.g., 14:30:00") },
                modifier = Modifier.fillMaxWidth()
            )
            OutlinedTextField(
                value = toTime,
                onValueChange = onToTimeChange,
                label = { Text("To Time (HH:MM:SS)") },
                placeholder = { Text("e.g., 14:35:00") },
                modifier = Modifier.fillMaxWidth()
            )
            OutlinedTextField(
                value = label,
                onValueChange = onLabelChange,
                label = { Text("Activity Label (A, B, C, D, or E)") },
                modifier = Modifier.fillMaxWidth()
            )
            Button(onClick = onApplyLabel, modifier = Modifier.align(Alignment.End)) {
                Text("Apply Label")
            }
        }
    }
}


@Composable
fun DeleteCard(
    fromTime: String,
    toTime: String,
    onFromTimeChange: (String) -> Unit,
    onToTimeChange: (String) -> Unit,
    onDelete: () -> Unit
) {
    Card(
        elevation = CardDefaults.cardElevation(defaultElevation = 2.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.3f))
    ) {
        Column(
            modifier = Modifier
                .padding(16.dp)
                .fillMaxWidth(),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Text(text = "Delete Labeled Data", style = MaterialTheme.typography.titleLarge)
            Text(
                text = "Permanently remove labeled windows from a specific time range to correct mistakes.",
                style = MaterialTheme.typography.bodySmall
            )
            OutlinedTextField(
                value = fromTime,
                onValueChange = onFromTimeChange,
                label = { Text("From Time (HH:MM:SS)") },
                modifier = Modifier.fillMaxWidth()
            )
            OutlinedTextField(
                value = toTime,
                onValueChange = onToTimeChange,
                label = { Text("To Time (HH:MM:SS)") },
                modifier = Modifier.fillMaxWidth()
            )
            Button(
                onClick = onDelete,
                modifier = Modifier.align(Alignment.End),
                colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.error)
            ) {
                Text("Delete Range")
            }
        }
    }
}