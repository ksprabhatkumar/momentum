// --- File Path: app/src/main/java/com/example/momentum/ui/WorkbenchScreen.kt ---
package com.example.momentum.ui

import android.widget.Toast
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Divider
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp

@Composable
fun WorkbenchScreen(viewModel: MainViewModel) {
    val context = LocalContext.current
    var fromTimeLabel by remember { mutableStateOf("") }
    var toTimeLabel by remember { mutableStateOf("") }
    var label by remember { mutableStateOf("") }

    var fromTimeDelete by remember { mutableStateOf("") }
    var toTimeDelete by remember { mutableStateOf("") }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp)
            .verticalScroll(rememberScrollState()),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        // Card for Labeling Data
        LabelingCard(
            fromTime = fromTimeLabel, toTime = toTimeLabel, label = label,
            onFromTimeChange = { fromTimeLabel = it },
            onToTimeChange = { toTimeLabel = it },
            onLabelChange = { label = it },
            onApplyLabel = {
                if (label.isBlank() || fromTimeLabel.isBlank() || toTimeLabel.isBlank()) {
                    Toast.makeText(context, "All labeling fields are required", Toast.LENGTH_SHORT).show()
                } else {
                    viewModel.applyWindowedLabels(fromTimeLabel, toTimeLabel, label) { count ->
                        val message = when {
                            count > 0 -> "$count new windows labeled!"
                            count == 0 -> "No new raw data in that time range."
                            else -> "Error: Invalid time format or range."
                        }
                        Toast.makeText(context, message, Toast.LENGTH_LONG).show()
                    }
                    fromTimeLabel = ""; toTimeLabel = ""; label = ""
                }
            }
        )

        Divider(
            modifier = Modifier.padding(vertical = 8.dp),
            color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.2f)
        )

        // Card for Deleting Data
        DeleteCard(
            fromTime = fromTimeDelete,
            toTime = toTimeDelete,
            onFromTimeChange = { fromTimeDelete = it },
            onToTimeChange = { toTimeDelete = it },
            onDelete = {
                if (fromTimeDelete.isBlank() || toTimeDelete.isBlank()) {
                    Toast.makeText(context, "All deletion fields are required", Toast.LENGTH_SHORT).show()
                } else {
                    viewModel.deleteLabeledWindowsInRange(fromTimeDelete, toTimeDelete) { count ->
                        val message = if (count >= 0) {
                            "$count labeled windows deleted."
                        } else {
                            "Error: Invalid time format or range."
                        }
                        Toast.makeText(context, message, Toast.LENGTH_LONG).show()
                    }
                    fromTimeDelete = ""; toTimeDelete = ""
                }
            }
        )
    }
}