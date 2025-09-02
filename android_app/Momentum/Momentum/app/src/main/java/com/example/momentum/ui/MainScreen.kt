// --- File Path: app/src/main/java/com/example/momentum/ui/MainScreen.kt ---
package com.example.momentum.ui

import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Assessment
import androidx.compose.material.icons.filled.Dashboard
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material.icons.filled.Sync // NEW ICON
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.navigation.NavDestination.Companion.hierarchy
import androidx.navigation.NavGraph.Companion.findStartDestination
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.currentBackStackEntryAsState
import androidx.navigation.compose.rememberNavController

// Define our navigation routes
sealed class Screen(val route: String, val label: String, val icon: ImageVector) {
    object Dashboard : Screen("dashboard", "Dashboard", Icons.Filled.Dashboard)
    object Workbench : Screen("workbench", "Workbench", Icons.Filled.Assessment)
    object Federated : Screen("federated", "Federate", Icons.Filled.Sync) // NEW SCREEN
    object Settings : Screen("settings", "Settings", Icons.Filled.Settings)
}

val items = listOf(
    Screen.Dashboard,
    Screen.Workbench,
    Screen.Federated, // NEW
    Screen.Settings,
)

@Composable
fun MainScreen(viewModel: MainViewModel) {
    val navController = rememberNavController()
    Scaffold(
        bottomBar = {
            NavigationBar(
                containerColor = MaterialTheme.colorScheme.surface
            ) {
                val navBackStackEntry by navController.currentBackStackEntryAsState()
                val currentDestination = navBackStackEntry?.destination
                items.forEach { screen ->
                    val selected = currentDestination?.hierarchy?.any { it.route == screen.route } == true
                    NavigationBarItem(
                        colors = NavigationBarItemDefaults.colors(
                            selectedIconColor = MaterialTheme.colorScheme.onPrimary,
                            selectedTextColor = MaterialTheme.colorScheme.primary,
                            indicatorColor = MaterialTheme.colorScheme.primary,
                            unselectedIconColor = MaterialTheme.colorScheme.onSurfaceVariant,
                            unselectedTextColor = MaterialTheme.colorScheme.onSurfaceVariant,
                        ),
                        icon = { Icon(screen.icon, contentDescription = screen.label) },
                        label = { Text(screen.label) },
                        selected = selected,
                        onClick = {
                            navController.navigate(screen.route) {
                                popUpTo(navController.graph.findStartDestination().id) { saveState = true }
                                launchSingleTop = true
                                restoreState = true
                            }
                        }
                    )
                }
            }
        }
    ) { innerPadding ->
        NavHost(navController, startDestination = Screen.Dashboard.route, Modifier.padding(innerPadding)) {
            composable(Screen.Dashboard.route) { DashboardScreen(viewModel) }
            composable(Screen.Workbench.route) { WorkbenchScreen(viewModel) }
            composable(Screen.Federated.route) { FederatedScreen(viewModel) } // NEW ROUTE
            composable(Screen.Settings.route) { SettingsScreen(viewModel) }
        }
    }
}