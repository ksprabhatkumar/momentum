// --- File Path: app/src/main/java/com/example/momentum/ui/DashboardScreen.kt ---
package com.example.momentum.ui

import androidx.compose.animation.AnimatedContent
import androidx.compose.animation.SizeTransform
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.slideInVertically
import androidx.compose.animation.slideOutVertically
import androidx.compose.animation.togetherWith
import androidx.compose.foundation.layout.*
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.* // Default icons
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.momentum.ui.theme.MomentumAccentIcon

@Composable
fun DashboardScreen(viewModel: MainViewModel) {
    val livePrediction by viewModel.livePrediction.collectAsState()
    val activityName by viewModel.liveActivityName.collectAsState()

    val activityIcon = when (livePrediction) {
        "A" -> Icons.Default.DirectionsWalk
        "B" -> Icons.Default.DirectionsRun
        "C" -> Icons.Default.Stairs
        "D" -> Icons.Default.Chair
        "E" -> Icons.Default.Person
        else -> Icons.Default.QuestionMark
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Card(
            shape = MaterialTheme.shapes.extraLarge,
            elevation = CardDefaults.cardElevation(defaultElevation = 8.dp),
            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
            modifier = Modifier
                .fillMaxWidth()
                .aspectRatio(1f) // Makes the card a square
        ) {
            Column(
                modifier = Modifier
                    .padding(24.dp)
                    .fillMaxSize(),
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.SpaceAround
            ) {
                Text(
                    text = "CURRENT ACTIVITY",
                    style = MaterialTheme.typography.titleMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    letterSpacing = 2.sp
                )

                AnimatedContent(
                    targetState = activityIcon,
                    transitionSpec = {
                        (slideInVertically { height -> height } + fadeIn())
                            .togetherWith(slideOutVertically { height -> -height } + fadeOut())
                            .using(SizeTransform(clip = false))
                    }, label = "ActivityIconAnimation"
                ) { targetIcon ->
                    Icon(
                        imageVector = targetIcon,
                        contentDescription = "Activity Icon",
                        modifier = Modifier.size(100.dp),
                        tint = MomentumAccentIcon
                    )
                }

                AnimatedContent(
                    targetState = activityName,
                    transitionSpec = {
                        (slideInVertically { height -> height } + fadeIn())
                            .togetherWith(slideOutVertically { height -> -height } + fadeOut())
                    }, label = "ActivityTextAnimation"
                ) { targetText ->
                    Text(
                        text = targetText,
                        style = MaterialTheme.typography.displaySmall,
                        fontWeight = FontWeight.Bold,
                        textAlign = TextAlign.Center,
                        color = MaterialTheme.colorScheme.onSurface
                    )
                }
            }
        }
    }
}