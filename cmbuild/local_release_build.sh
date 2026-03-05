#!/bin/bash

# Stop on error
set -e

# ================= configuration =================
# Set current directory to the script's directory (build folder)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
WORKSPACE="$(dirname "$SCRIPT_DIR")" # Parent of build dir (root of repo)

# Paths
ANDROID_PROJECT_DIR="${WORKSPACE}/examples/face_landmarker/android"
IOS_PROJECT_DIR="${WORKSPACE}/examples/face_landmarker/ios"
OUTPUT_DIR="${WORKSPACE}/release_output"

# Arguments
# Usage: ./local_release_build.sh [mode] [version]
# mode: all (default), ios, android
# version: 1.0.0 (default)

MODE=${1:-"all"}
VERSION_NAME=${2:-"1.0.0"}

# Handle case where user might swap args or only provide version
if [[ "$MODE" =~ ^[0-9]+(\.[0-9]+)+$ ]]; then
    TEMP=$VERSION_NAME
    VERSION_NAME=$MODE
    if [[ "$TEMP" == "ios" || "$TEMP" == "android" || "$TEMP" == "all" ]]; then
        MODE=$TEMP
    else
        MODE="all"
    fi
fi

# Update iOS Version in project.pbxproj
echo "🔧 Updating iOS Version to $VERSION_NAME..."
if [ -f "${IOS_PROJECT_DIR}/FaceLandmarker.xcodeproj/project.pbxproj" ]; then
    # Assuming standard version format x.y.z
    # We can split it if needed, or just set MARKETING_VERSION to full string
    # and CURRENT_PROJECT_VERSION to something unique (like a timestamp or just last digit)
    # matching jenkins logic:
    # majorNumber=$(echo $version | cut -d '.' -f1)
    # minorNumber=$(echo $version | cut -d '.' -f2)
    # patchNumber=$(echo $version | cut -d '.' -f3)
    # MARKETING_VERSION=${majorNumber}.${minorNumber}.${patchNumber}
    # CURRENT_PROJECT_VERSION=${patchNumber}

    # For local build simpliciy, let's just use the provided version string
    # But to be safe and standard, we should try to match the project structure.
    
    # Use perl or sed to update.
    # Pattern: MARKETING_VERSION = 1.0.0;
    # Pattern: CURRENT_PROJECT_VERSION = 1;
    
    # Extract components
    VERSION_MAJOR=$(echo $VERSION_NAME | cut -d. -f1)
    VERSION_MINOR=$(echo $VERSION_NAME | cut -d. -f2)
    VERSION_PATCH=$(echo $VERSION_NAME | cut -d. -f3)
    VERSION_BUILD=$(echo $VERSION_NAME | cut -d. -f4)
    
    # Fallback if empty
    VERSION_MAJOR=${VERSION_MAJOR:-1}
    VERSION_MINOR=${VERSION_MINOR:-0}
    VERSION_PATCH=${VERSION_PATCH:-0}
    
    # 如果有第 4 段，使用第 4 段作為 build number；否則使用第 3 段
    # 例如：1.0.0.33 -> MARKETING_VERSION=1.0.0, BUILD_NUMBER=33
    #      1.0.33.1 -> MARKETING_VERSION=1.0.33, BUILD_NUMBER=1
    if [ -n "$VERSION_BUILD" ] && [ "$VERSION_BUILD" != "$VERSION_NAME" ]; then
         BUILD_NUMBER="${VERSION_BUILD}"
    else
         BUILD_NUMBER="${VERSION_PATCH}"
    fi

    CLEAN_VERSION="${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}"
    # BUILD_NUMBER is already set above
    
    sed -i '' -E "s/(MARKETING_VERSION[[:space:]]*=).*/\1 $CLEAN_VERSION;/g" "${IOS_PROJECT_DIR}/FaceLandmarker.xcodeproj/project.pbxproj"
    sed -i '' -E "s/(CURRENT_PROJECT_VERSION[[:space:]]*=).*/\1 $BUILD_NUMBER;/g" "${IOS_PROJECT_DIR}/FaceLandmarker.xcodeproj/project.pbxproj"
    echo "✅ iOS Version updated to $CLEAN_VERSION ($BUILD_NUMBER)"
fi

# Clean Output Directory
echo "🧹 Cleaning output directory: $OUTPUT_DIR"
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"




# Helper function for iOS build
build_ios_scheme() {
    local SCHEME_NAME=$1
    local EXPORT_PLIST=$2
    local OUTPUT_NAME=$3
    local DEVELOPMENT_TEAM_ID=$4
    
    echo "-------------------------------------------"
    echo "🍎 Building iOS: $SCHEME_NAME ($OUTPUT_NAME)..."
    echo "-------------------------------------------"

    if [ -d "$IOS_PROJECT_DIR" ]; then
        cd "$IOS_PROJECT_DIR"
        
        ARCHIVE_PATH="${IOS_PROJECT_DIR}/build/${SCHEME_NAME}.xcarchive"
        EXPORT_PATH="${OUTPUT_DIR}"
        
        echo "Cleaning and Archiving $SCHEME_NAME..."
        xcodebuild -scheme "$SCHEME_NAME" \
            -configuration Release \
            -archivePath "$ARCHIVE_PATH" \
            -sdk iphoneos \
            -allowProvisioningUpdates \
            DEVELOPMENT_TEAM="${DEVELOPMENT_TEAM_ID}" \
            clean archive
            
        if [ $? -eq 0 ]; then
            echo "✅ iOS Archive Successful"
        else
            echo "❌ iOS Archive Failed"
            exit 1
        fi
        
        echo "Exporting IPA..."
        xcodebuild -exportArchive \
            -archivePath "$ARCHIVE_PATH" \
            -exportOptionsPlist "$EXPORT_PLIST" \
            -exportPath "$EXPORT_PATH" \
            -allowProvisioningUpdates
            
        if [ $? -eq 0 ]; then
            echo "✅ iOS Export Successful"
            
            local DATE_SUFFIX=$(date +%m%d%H%M%S)
            
            # Identify the generated IPA (xcodebuild uses Scheme Name usually)
            local GENERATED_IPA=""
            if [ -f "$OUTPUT_DIR/${SCHEME_NAME}.ipa" ]; then
                GENERATED_IPA="$OUTPUT_DIR/${SCHEME_NAME}.ipa"
            elif [ -f "$OUTPUT_DIR/${OUTPUT_NAME}.ipa" ]; then
                GENERATED_IPA="$OUTPUT_DIR/${OUTPUT_NAME}.ipa"
            elif [ -f "$OUTPUT_DIR/RoadSafetyAppEnt.ipa" ]; then
                GENERATED_IPA="$OUTPUT_DIR/RoadSafetyAppEnt.ipa"
            fi

            if [ -f "$GENERATED_IPA" ]; then
                echo "📦 IPA generated and available at: $GENERATED_IPA"

            else
                echo "❌ Could not find generated IPA."
            fi
            
        else
            echo "❌ iOS Export Failed"
            exit 1
        fi
    else
         echo "⚠️ iOS project directory not found at $IOS_PROJECT_DIR"
         exit 1
    fi
}

# =================================================
#                 iOS BUILD
# =================================================
if [[ "$MODE" == "ios" ]]; then
    if [ -f "${IOS_PROJECT_DIR}/FaceLandmarker.xcodeproj/project.pbxproj" ]; then
        echo "🔧 Switching Team ID to C8XRK2QZ5P (App Store) for local iOS build..."
        sed -i '' 's/DEVELOPMENT_TEAM = BW7EQP9YQ5;/DEVELOPMENT_TEAM = C8XRK2QZ5P;/g' "${IOS_PROJECT_DIR}/FaceLandmarker.xcodeproj/project.pbxproj"
    fi
    build_ios_scheme "FaceLandmarker" "${SCRIPT_DIR}/exportOptions_AdHoc.plist" "RoadSafetyApp" "C8XRK2QZ5P"
fi

if [[ "$MODE" == "ios-ent" || "$MODE" == "all" ]]; then
    # 1. Force Project Team ID to BW7EQP9YQ5 (Enterprise Team)
    if [ -f "${IOS_PROJECT_DIR}/FaceLandmarker.xcodeproj/project.pbxproj" ]; then
        echo "🔧 Switching Team ID to BW7EQP9YQ5 (Enterprise)..."
        sed -i '' 's/DEVELOPMENT_TEAM = C8XRK2QZ5P;/DEVELOPMENT_TEAM = BW7EQP9YQ5;/g' "${IOS_PROJECT_DIR}/FaceLandmarker.xcodeproj/project.pbxproj"
    fi
    
    build_ios_scheme "FaceLandmarker" "${SCRIPT_DIR}/exportOptions_Enterprise.plist" "RoadSafetyAppEnt" "BW7EQP9YQ5"
fi

if [[ "$MODE" == "ios-testflight" || "$MODE" == "all" ]]; then
    # 1. Force Project Team ID to C8XRK2QZ5P (App Store Team)
    if [ -f "${IOS_PROJECT_DIR}/FaceLandmarker.xcodeproj/project.pbxproj" ]; then
        echo "🔧 Switching Team ID to C8XRK2QZ5P (App Store) for TestFlight build..."
        sed -i '' 's/DEVELOPMENT_TEAM = BW7EQP9YQ5;/DEVELOPMENT_TEAM = C8XRK2QZ5P;/g' "${IOS_PROJECT_DIR}/FaceLandmarker.xcodeproj/project.pbxproj"
    fi

    # build_ios_scheme "FaceLandmarker" "${SCRIPT_DIR}/exportOptions_AppStore.plist" "RoadSafetyApp_TestFlight"
    # Note: TestFlight build logic is kept but commented out or minimal if not main focus.
    # To enable, uncomment line above. user requested 'all' currently includes it in previous versions.
    # Restoring it:
    build_ios_scheme "FaceLandmarker" "${SCRIPT_DIR}/exportOptions_AppStore.plist" "RoadSafetyApp" "C8XRK2QZ5P"
fi


# =================================================
#               ANDROID BUILD
# =================================================
if [[ "$MODE" == "android" || "$MODE" == "all" ]]; then
    echo "-------------------------------------------"
    echo "🤖 Building Android APK..."
    echo "-------------------------------------------"
    
    # Set Java 17 for Gradle 8.9 compatibility
    export JAVA_HOME=/Library/Java/JavaVirtualMachines/temurin-17.jdk/Contents/Home
    
    # Generate Timestamp
    ANDROID_TIMESTAMP=$(date +%m%d%H%M%S)
    
    if [ -d "$ANDROID_PROJECT_DIR" ]; then
        cd "$ANDROID_PROJECT_DIR"
        chmod +x gradlew
        
        ./gradlew clean assembleRelease
            
        if [ $? -eq 0 ]; then
            echo "✅ Android Build Successful"
            
            find app/build/outputs/apk/release -name "*.apk" -exec cp {} "$OUTPUT_DIR/FaceLandmarker.apk" \;
            echo "📦 APK copied to $OUTPUT_DIR/FaceLandmarker.apk"
        else
            echo "❌ Android Build Failed"
            exit 1
        fi
    else
        echo "⚠️ Android project directory not found at $ANDROID_PROJECT_DIR"
    fi
fi





# =================================================
#               FINALIZE
# =================================================


# =================================================
#               VERIFY BUILD
# =================================================
echo "-------------------------------------------"
echo "🔍 Verifying Build Artifacts..."
echo "-------------------------------------------"

VERIFY_SUCCESS=true

# 1. Check Android APK (if android build was attempted)
if [[ "$MODE" == "android" || "$MODE" == "all" ]]; then
    if find "$OUTPUT_DIR" -name "*.apk" | grep -q .; then
        echo "✅ Found Android APK"
    else
        echo "❌ Missing Android APK in $OUTPUT_DIR"
        VERIFY_SUCCESS=false
    fi
fi

# 2. Check iOS IPA (if ios build was attempted)
if [[ "$MODE" == *ios* || "$MODE" == "all" ]]; then
    if find "$OUTPUT_DIR" -name "*.ipa" | grep -q .; then
        echo "✅ Found iOS IPA"
    else
        echo "❌ Missing iOS IPA in $OUTPUT_DIR"
        VERIFY_SUCCESS=false
    fi
fi

echo "========================================="
if [ "$VERIFY_SUCCESS" = true ]; then
    echo "🎉 Build Verification PASSED!"
    echo "Files are located in: $OUTPUT_DIR"
    ls -R "$OUTPUT_DIR"
    exit 0
else
    echo "💥 Build Verification FAILED!"
    exit 1
fi

