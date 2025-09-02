pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
    }

    // versionCatalogs {  // Temporarily comment this out or remove it
    //     create("libs") {
    //         from(files("gradle/libs.versions.toml"))
    //     }
    // }
}

rootProject.name = "Momentum"
include(":app")