import org.jetbrains.kotlin.gradle.tasks.KotlinCompile

plugins {
    kotlin("jvm") version "1.5.0"
}

group = "com.github.rnett.weaver"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
}

dependencies {
    testImplementation(kotlin("test-junit"))
    implementation("org.tensorflow:tensorflow-core-api:0.3.1")
}

tasks.test {
    useJUnit()
}

tasks.withType<KotlinCompile>() {
    kotlinOptions.jvmTarget = "1.8"
}

kotlin{
    sourceSets.all {
        languageSettings{
            useExperimentalAnnotation("kotlin.contracts.ExperimentalContracts")
        }
    }
}