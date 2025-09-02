package com.example.momentum.ui.theme

import androidx.compose.ui.graphics.Color

// --- Core Palette for Momentum App (Dark Theme Focused) ---

// Primary colors are used for main actions, buttons, and highlights.
val MomentumPrimary = Color(0xFF5C40FF) // A vibrant, modern purple/blue.
val MomentumSecondary = Color(0xFFFFD740) // A warm, energetic amber for secondary actions or highlights.

// Background and Surface colors define the app's base look.
val MomentumBackground = Color(0xFF1A1B24) // A deep, dark blue-charcoal for the main background.
val MomentumSurface = Color(0xFF23243D) // A slightly lighter slate blue for cards and elevated surfaces.

// "On" colors are used for text and icons placed on top of the primary, surface, etc.
val MomentumOnPrimary = Color(0xFFFFFFFF) // White text on primary buttons.
val MomentumOnSurface = Color(0xFFE1E2EF) // A soft, off-white for body text for better readability on dark surfaces.
val MomentumOnSurfaceVariant = Color(0xFF7B8FA1) // A muted gray for subtitles and less important text.

// Accent colors for specific UI states or elements.
val MomentumSuccess = Color(0xFF16B877) // A clear green for success states (e.g., sync complete).
val MomentumError = Color(0xFFFF4D4F) // A standard red for error messages or warnings.
val MomentumAccentIcon = Color(0xFF44C9E7) // A bright teal for the live activity icon on the dashboard.
val MomentumInactive = Color(0xFF7B8FA1) // The same as the variant text, used for inactive nav icons or disabled states.