package com.google.mediapipe.examples.facelandmarker

import android.graphics.PointF
import android.graphics.RectF
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import kotlin.math.*
import android.util.Log

// Simple 1D Kalman Filter class
private class KalmanFilter( var processNoise: Float,  var measurementNoise: Float,  var estimatedError: Float,  var lastEstimate: Float = 0.0f) {
    fun update(measurement: Float): Float {
        val kalmanGain = estimatedError / (estimatedError + measurementNoise)
        val currentEstimate = lastEstimate + kalmanGain * (measurement - lastEstimate)
        estimatedError = (1 - kalmanGain) * estimatedError + abs(lastEstimate - currentEstimate) * processNoise
        lastEstimate = currentEstimate
        return currentEstimate
    }

    fun reset(initialEstimate: Float) {
        lastEstimate = initialEstimate
        estimatedError = 1.0f // Reset error to a starting value
    }
}

// Basic 3D vector operations (using FloatArray for simplicity)
private fun subtractVectors(v1: FloatArray, v2: FloatArray): FloatArray {
    return floatArrayOf(v1[0] - v2[0], v1[1] - v2[1], v1[2] - v2[2])
}

private fun normalizeVector(v: FloatArray): FloatArray {
    val norm = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    return if (norm > 1e-6) floatArrayOf(v[0] / norm, v[1] / norm, v[2] / norm) else floatArrayOf(0f, 0f, 0f)
}

private fun crossProduct(v1: FloatArray, v2: FloatArray): FloatArray {
    return floatArrayOf(
        v1[1] * v2[2] - v1[2] * v2[1],
        v1[2] * v2[0] - v1[0] * v2[2],
        v1[0] * v2[1] - v1[1] * v2[0]
    )
}

// Simple matrix multiplication (3x3 * 3x3)
private fun multiplyMatrices(m1: Array<FloatArray>, m2: Array<FloatArray>): Array<FloatArray> {
    val result = Array(3) { FloatArray(3) }
    for (i in 0 until 3) {
        for (j in 0 until 3) {
            for (k in 0 until 3) {
                result[i][j] += m1[i][k] * m2[k][j]
            }
        }
    }
    return result
}

// Convert rotation matrix (3x3) to Euler angles (pitch, yaw, roll - YXZ convention often used in head pose)
// Source: https://www.learnopencv.com/rotation-matrix-to-euler-angles/ (Approximation, YXZ)
private fun rotationMatrixToEulerAngles(R: Array<FloatArray>): Triple<Float, Float, Float> {
    val sy = sqrt(R[0][0] * R[0][0] + R[1][0] * R[1][0])
    val singular = sy < 1e-6

    val x: Float // Pitch
    val y: Float // Yaw
    val z: Float // Roll

    if (!singular) {
        x = atan2(R[2][1], R[2][2])
        y = atan2(-R[2][0], sy)
        z = atan2(R[1][0], R[0][0])
    } else {
        x = atan2(-R[1][2], R[1][1])
        y = atan2(-R[2][0], sy)
        z = 0f
    }

    // Convert radians to degrees
    return Triple(Math.toDegrees(x.toDouble()).toFloat(), Math.toDegrees(y.toDouble()).toFloat(), Math.toDegrees(z.toDouble()).toFloat())
}

class FacePoseAnalyzer {

    // Kalman filters for yaw, pitch, and roll (for the original calculation)
    private var yawFilter: KalmanFilter = KalmanFilter(processNoise = 0.02f, measurementNoise = 0.1f, estimatedError = 1.0f)
    private var pitchFilter: KalmanFilter = KalmanFilter(processNoise = 0.02f, measurementNoise = 0.1f, estimatedError = 1.0f)
    private var rollFilter: KalmanFilter = KalmanFilter(processNoise = 0.02f, measurementNoise = 0.1f, estimatedError = 1.0f)

    // Kalman filters for the improved calculation
    private var improvedYawFilter: KalmanFilter = KalmanFilter(processNoise = 0.02f, measurementNoise = 0.1f, estimatedError = 1.0f)
    private var improvedPitchFilter: KalmanFilter = KalmanFilter(processNoise = 0.02f, measurementNoise = 0.1f, estimatedError = 1.0f)
    private var improvedRollFilter: KalmanFilter = KalmanFilter(processNoise = 0.02f, measurementNoise = 0.1f, estimatedError = 1.0f)

    // Flag to indicate if filters have been initialized
    private var filtersInitialized = false
    private var improvedFiltersInitialized = false

    // Face landmark indices for MediaPipe Face Landmarker
    companion object {
        // Face contour points
         const val NOSE_TIP = 1
         const val LEFT_EYE = 33
         const val RIGHT_EYE = 263
         const val LEFT_MOUTH = 61
         const val RIGHT_MOUTH = 291
         const val LEFT_EAR = 234
         const val RIGHT_EAR = 454
         const val CHIN = 152

        // Eye landmarks
        private val LEFT_EYE_LANDMARKS = listOf(33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246)
        private val RIGHT_EYE_LANDMARKS = listOf(263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466)

        // Iris landmarks
        private val LEFT_IRIS_LANDMARKS = listOf(468, 469, 470, 471, 472)
        private val RIGHT_IRIS_LANDMARKS = listOf(473, 474, 475, 476, 477)

        // Eye landmarks for openness calculation
        private const val LEFT_EYE_TOP = 159
        private const val LEFT_EYE_BOTTOM = 145
        private const val RIGHT_EYE_TOP = 386
        private const val RIGHT_EYE_BOTTOM = 374

        // Fatigue detection constants
        private const val EYE_OPEN_THRESHOLD = 0.25f  // Threshold for eye openness
        private const val BLINK_DURATION_THRESHOLD = 0.3f  // Seconds
        private const val FATIGUE_THRESHOLD = 0.5f  // Threshold for fatigue detection

        // Landmark indices used in the improved calculation (from Python code)
        private const val LM_280 = 280
        private const val LM_50 = 50
        private const val LM_352 = 352
        private const val LM_123 = 123
        private const val LM_376 = 376
        private const val LM_147 = 147
        private const val LM_416 = 416
        private const val LM_192 = 192
        private const val LM_298 = 298
        private const val LM_68 = 68
        private const val LM_301 = 301
        private const val LM_71 = 71
        private const val LM_10 = 10
        private const val LM_151 = 151
        private const val LM_8 = 8
        private const val LM_17 = 17
        private const val LM_5 = 5
        private const val LM_200 = 200
        private const val LM_6 = 6
        private const val LM_199 = 199
        private const val LM_18 = 18
        private const val LM_9 = 9
        private const val LM_175 = 175
    }

    // Fatigue detection state
    private var lastBlinkTime = 0L
    private var blinkCount = 0
    private var fatigueScore = 0f
    private var eyeOpennessHistory = mutableListOf<Float>()

    // Data class to hold pose information, including the improved calculation
    data class FacePose(
        val yaw: Float,      // Original calculation: Left-right rotation
        val pitch: Float,    // Original calculation: Up-down rotation
        val roll: Float,     // Original calculation: Tilt rotation
        val improvedYaw: Float, // Improved calculation: Left-right rotation
        val improvedPitch: Float, // Improved calculation: Up-down rotation
        val improvedRoll: Float, // Improved calculation: Tilt rotation
        val eyePositions: EyePositions,
        val irisPositions: IrisPositions,
        val screenDistance: Float? = null,
        val fatigueMetrics: FatigueMetrics? = null,
        val v2Yaw: Float = 0f, // New v2 yaw
        val v2Pitch: Float = 0f, // New v2 pitch
        val v2Roll: Float = 0f // New v2 roll
    )

    data class EyePositions(
        val leftEye: PointF,
        val rightEye: PointF
    )

    data class IrisPositions(
        val leftIris: PointF,
        val rightIris: PointF
    )

    data class FatigueMetrics(
        val leftEyeOpenness: Float,
        val rightEyeOpenness: Float,
        val averageEyeOpenness: Float,
        val isFatigued: Boolean,
        val blinkCount: Int,
        val fatigueScore: Float
    )

    fun analyzeFacePose(landmarks: List<NormalizedLandmark>, imageWidth: Int, imageHeight: Int): FacePose {
        // Calculate head pose angles using the original method
        val (rawYaw, rawPitch, rawRoll) = calculateHeadPose(landmarks)

        // Initialize original filters
        if (!filtersInitialized) {
            yawFilter.reset(rawYaw)
            pitchFilter.reset(rawPitch)
            rollFilter.reset(rawRoll)
            filtersInitialized = true
        } else {
             // Update original filters
            yawFilter.update(rawYaw)
            pitchFilter.update(rawPitch)
            rollFilter.update(rawRoll)
        }

        // Calculate head pose angles using the improved method
        val (improvedRawPitch, improvedRawYaw, improvedRawRoll) = calculateFacePoseImproved(landmarks, imageWidth, imageHeight)

        // Initialize improved filters
        if (!improvedFiltersInitialized) {
            improvedYawFilter.reset(improvedRawYaw)
            improvedPitchFilter.reset(improvedRawPitch)
            improvedRollFilter.reset(improvedRawRoll)
            improvedFiltersInitialized = true
        } else {
            // Update improved filters
            improvedYawFilter.update(improvedRawYaw)
            improvedPitchFilter.update(improvedRawPitch)
            improvedRollFilter.update(improvedRawRoll)
        }

        // Get filtered pose from both methods
        val filteredYaw = yawFilter.lastEstimate
        val filteredPitch = pitchFilter.lastEstimate
        val filteredRoll = rollFilter.lastEstimate

        val improvedFilteredYaw = improvedYawFilter.lastEstimate
        val improvedFilteredPitch = improvedPitchFilter.lastEstimate
        val improvedFilteredRoll = improvedRollFilter.lastEstimate

        // Calculate eye positions
        val eyePositions = calculateEyePositions(landmarks)

        // Calculate iris positions
        val irisPositions = calculateIrisPositions(landmarks)

        // Calculate screen distance
        val screenDistance = calculateScreenDistance(landmarks)

        // Calculate fatigue metrics
        val fatigueMetrics = calculateFatigueMetrics(landmarks)

        // Calculate v2 head pose estimation (based on user-provided Python logic)
        val (v2Yaw, v2Pitch, v2Roll) = calculateFacePoseV2(landmarks, imageWidth, imageHeight)

        return FacePose(
            yaw = filteredYaw,
            pitch = filteredPitch,
            roll = filteredRoll,
            improvedYaw = improvedFilteredYaw,
            improvedPitch = improvedFilteredPitch,
            improvedRoll = improvedFilteredRoll,
            eyePositions = eyePositions,
            irisPositions = irisPositions,
            screenDistance = screenDistance,
            fatigueMetrics = fatigueMetrics,
            v2Yaw = v2Yaw,
            v2Pitch = v2Pitch,
            v2Roll = v2Roll
        )
    }

    // Helper function to get landmark coordinates as FloatArray [x, y, z]
    private fun getLandmarkCoords(landmark: NormalizedLandmark, imageWidth: Int, imageHeight: Int): FloatArray {
         // Scale and adjust Y and Z as in Python code
        val x = landmark.x() * imageWidth
        val y = -landmark.y() * imageHeight
        val z = -landmark.z() * imageWidth // Assuming Z scaling similar to X
        return floatArrayOf(x, y, z)
    }

    // Implementation of the improved face pose calculation from Python
    private fun calculateFacePoseImproved(landmarks: List<NormalizedLandmark>, imageWidth: Int, imageHeight: Int): Triple<Float, Float, Float> {
        // Check if there are enough landmarks for this calculation (at least up to LM_416)
        if (landmarks.size <= LM_416) { 
             Log.e("FacePoseAnalyzer", "Not enough landmarks for improved pose calculation. Required at least ${LM_416 + 1}, but got ${landmarks.size}.")
             // Return a default or indicate an error
             return Triple(0f, 0f, 0f) // Default to 0 angles
         }

        // Get 3D coordinates of necessary landmarks
        val landmark3d = landmarks.map { getLandmarkCoords(it, imageWidth, imageHeight) }

        // Calculate X-axis vector based on the Python logic
        var xAxis = subtractVectors(landmark3d[LM_280], landmark3d[LM_50])
        xAxis = subtractVectors(xAxis, subtractVectors(landmark3d[LM_352], landmark3d[LM_123])) // Add by subtracting
        xAxis = subtractVectors(xAxis, subtractVectors(landmark3d[LM_280], landmark3d[LM_50])) // Add again? Seems redundant in Python, translating directly.
        xAxis = subtractVectors(xAxis, subtractVectors(landmark3d[LM_376], landmark3d[LM_147]))
        xAxis = subtractVectors(xAxis, subtractVectors(landmark3d[LM_416], landmark3d[LM_192]))
        xAxis = subtractVectors(xAxis, subtractVectors(landmark3d[LM_298], landmark3d[LM_68]))
        xAxis = subtractVectors(xAxis, subtractVectors(landmark3d[LM_301], landmark3d[LM_71]))

         // Calculate Y-axis vector based on the Python logic
        var yAxis = subtractVectors(landmark3d[LM_10], landmark3d[CHIN]) // Using CHIN=152
        yAxis = subtractVectors(yAxis, subtractVectors(landmark3d[LM_151], landmark3d[CHIN])) // Add by subtracting
        yAxis = subtractVectors(yAxis, subtractVectors(landmark3d[LM_8], landmark3d[LM_17]))
        yAxis = subtractVectors(yAxis, subtractVectors(landmark3d[LM_5], landmark3d[LM_200]))
        yAxis = subtractVectors(yAxis, subtractVectors(landmark3d[LM_6], landmark3d[LM_199]))
        yAxis = subtractVectors(yAxis, subtractVectors(landmark3d[LM_8], landmark3d[LM_18]))
        yAxis = subtractVectors(yAxis, subtractVectors(landmark3d[LM_9], landmark3d[LM_175]))


        // Normalize the calculated axes
        xAxis = normalizeVector(xAxis)
        yAxis = normalizeVector(yAxis)

        // Recalculate Z-axis as cross product of X and Y, then re-normalize Y
        var zAxis = crossProduct(xAxis, yAxis)
        zAxis = normalizeVector(zAxis)
        yAxis = crossProduct(zAxis, xAxis) // Re-orthogonalize Y to X and Z
        yAxis = normalizeVector(yAxis) // Normalize Y again

        // Construct the rotation matrix from the orthogonal basis vectors
        // The matrix columns are the X, Y, and Z basis vectors
        val rotationMatrix = arrayOf(
            floatArrayOf(xAxis[0], yAxis[0], zAxis[0]),
            floatArrayOf(xAxis[1], yAxis[1], zAxis[1]),
            floatArrayOf(xAxis[2], yAxis[2], zAxis[2])
        )

        // Python code applies an additional rotation Rtool.from_rotvec([-0.25, 0, 0])
        // This is a rotation around the X-axis by -0.25 radians.
        // We need to create a rotation matrix for this offset and multiply.
        val angleX = -0.25f // radians
        val cosX = cos(angleX)
        val sinX = sin(angleX)
        val offsetMatrix = arrayOf(
            floatArrayOf(1f, 0f, 0f),
            floatArrayOf(0f, cosX, -sinX),
            floatArrayOf(0f, sinX, cosX)
        )

        // Apply the offset rotation: Result = OriginalRotation * OffsetRotation
        val finalRotationMatrix = multiplyMatrices(rotationMatrix, offsetMatrix)

        // Convert the final rotation matrix to Euler angles (Pitch, Yaw, Roll)
        // Using YXZ convention as implied by the order of axes in Python logic
        // roll (Z), pitch (X), yaw (Y) -> order of application YXZ
         val (pitch, yaw, roll) = rotationMatrixToEulerAngles(finalRotationMatrix)


        // Note: The order of angles (yaw, pitch, roll) and their mapping to axes (X, Y, Z)
        // can vary by convention (e.g., aerospace ZYX, robotics ZYZ).
        // The Python code returns a rotation vector. Converting a rotation vector to Euler
        // angles also depends on the chosen convention.
        // The `rotationMatrixToEulerAngles` function uses a YXZ convention.
        // We return them as (pitch, yaw, roll) matching the existing FacePose fields,
        // assuming YXZ maps roughly to Pitch (X), Yaw (Y), Roll (Z).
        // Need to confirm the exact mapping if precise correspondence is critical.
        // Based on typical face pose:
        // Yaw: Left/Right (rotation around Y)
        // Pitch: Up/Down (rotation around X)
        // Roll: Tilt (rotation around Z)
        // The YXZ conversion function returns pitch (X), yaw (Y), roll (Z).
        // So the order (pitch, yaw, roll) from the function matches (pitch, yaw, roll) convention.

        return Triple(pitch, yaw, roll) // Return calculated Pitch, Yaw, Roll
    }

    // Function to get the current filtered pose values from the ORIGINAL calculation
    fun getFilteredPose(): Triple<Float, Float, Float> {
        return Triple(yawFilter.lastEstimate, pitchFilter.lastEstimate, rollFilter.lastEstimate)
    }

     // Function to get the current filtered pose values from the IMPROVED calculation
     fun getImprovedFilteredPose(): Triple<Float, Float, Float> {
         return Triple(improvedYawFilter.lastEstimate, improvedPitchFilter.lastEstimate, improvedRollFilter.lastEstimate)
     }

    // Function to calculate a simple bounding box around the nose
    fun getFaceBoundingBox(landmarks: List<NormalizedLandmark>, imageWidth: Int, imageHeight: Int, scaleFactor: Float, offsetX: Float, offsetY: Float): RectF? {
        if (landmarks.isEmpty() || landmarks.size <= NOSE_TIP) return null // Ensure landmark exists
        val noseTip = landmarks[NOSE_TIP]
        val noseX = noseTip.x() * imageWidth * scaleFactor + offsetX
        val noseY = noseTip.y() * imageHeight * scaleFactor + offsetY

        // Estimate a bounding box around the nose based on scale factor
        val boxHalfSize = 80f * scaleFactor // Adjust size as needed

        val left = noseX - boxHalfSize
        val top = noseY - boxHalfSize
        val right = noseX + boxHalfSize
        val bottom = noseY + boxHalfSize

        return RectF(left, top, right, bottom)
    }

    private fun calculateHeadPose(landmarks: List<NormalizedLandmark>): Triple<Float, Float, Float> {
         if (landmarks.size <= max(NOSE_TIP, max(LEFT_MOUTH, RIGHT_MOUTH))) { // Basic check for sufficient landmarks
              Log.e("FacePoseAnalyzer", "Not enough landmarks for basic pose calculation.")
              // Return a default or indicate an error
              return Triple(0f, 0f, 0f) // Default to 0 angles
          }
        // Get key points for head pose calculation
        val noseTip = landmarks[NOSE_TIP]
        val leftEye = landmarks[LEFT_EYE]
        val rightEye = landmarks[RIGHT_EYE]
        val leftMouth = landmarks[LEFT_MOUTH]
        val rightMouth = landmarks[RIGHT_MOUTH]

        // Calculate roll (tilt)
        val roll = calculateRoll(leftEye, rightEye)

        // Calculate pitch (up-down)
        val pitch = calculatePitch(noseTip, leftEye, rightEye)

        // Calculate yaw (left-right)
        val yaw = calculateYaw(leftEye, rightEye, leftMouth, rightMouth)

        return Triple(yaw, pitch, roll)
    }

    private fun calculateRoll(leftEye: NormalizedLandmark, rightEye: NormalizedLandmark): Float {
        val dx = rightEye.x() - leftEye.x()
        val dy = rightEye.y() - leftEye.y()
        return atan2(dy, dx) * (180f / PI.toFloat())
    }

    private fun calculatePitch(noseTip: NormalizedLandmark, leftEye: NormalizedLandmark, rightEye: NormalizedLandmark): Float {
        val eyeCenterY = (leftEye.y() + rightEye.y()) / 2
        val noseToEyeDistance = noseTip.y() - eyeCenterY
        return atan2(noseToEyeDistance, 0.1f) * (180f / PI.toFloat()) // 0.1f is an arbitrary depth reference
    }

    private fun calculateYaw(leftEye: NormalizedLandmark, rightEye: NormalizedLandmark,
                           leftMouth: NormalizedLandmark, rightMouth: NormalizedLandmark): Float {
        val eyeWidth = rightEye.x() - leftEye.x()
        val mouthWidth = rightMouth.x() - leftMouth.x()
        val ratio = if (mouthWidth == 0f) 0f else eyeWidth / mouthWidth
        return (ratio - 1.0f) * 45f // Approximate yaw angle, adjust multiplier as needed
    }

    private fun calculateEyePositions(landmarks: List<NormalizedLandmark>): EyePositions {
         if (landmarks.size <= max(LEFT_EYE_LANDMARKS.maxOrNull() ?: 0, RIGHT_EYE_LANDMARKS.maxOrNull() ?: 0)) {
             Log.e("FacePoseAnalyzer", "Not enough landmarks for eye position calculation.")
              return EyePositions(PointF(0f,0f), PointF(0f,0f)) // Default to 0 positions
         }
        val leftEyeCenter = calculateCenter(LEFT_EYE_LANDMARKS.map { landmarks[it] })
        val rightEyeCenter = calculateCenter(RIGHT_EYE_LANDMARKS.map { landmarks[it] })

        return EyePositions(
            leftEye = PointF(leftEyeCenter.x(), leftEyeCenter.y()),
            rightEye = PointF(rightEyeCenter.x(), rightEyeCenter.y())
        )
    }

    private fun calculateIrisPositions(landmarks: List<NormalizedLandmark>): IrisPositions {
         if (landmarks.size <= max(LEFT_IRIS_LANDMARKS.maxOrNull() ?: 0, RIGHT_IRIS_LANDMARKS.maxOrNull() ?: 0)) {
             Log.e("FacePoseAnalyzer", "Not enough landmarks for iris position calculation.")
             return IrisPositions(PointF(0f,0f), PointF(0f,0f)) // Default to 0 positions
         }
        val leftIrisCenter = calculateCenter(LEFT_IRIS_LANDMARKS.map { landmarks[it] })
        val rightIrisCenter = calculateCenter(RIGHT_IRIS_LANDMARKS.map { landmarks[it] })

        return IrisPositions(
            leftIris = PointF(leftIrisCenter.x(), leftIrisCenter.y()),
            rightIris = PointF(rightIrisCenter.x(), rightIrisCenter.y())
        )
    }

    private fun calculateCenter(landmarks: List<NormalizedLandmark>): NormalizedLandmark {
        val x = landmarks.map { it.x() }.average().toFloat()
        val y = landmarks.map { it.y() }.average().toFloat()
        return NormalizedLandmark.create(x, y, 0f) // z is not used here
    }

    private fun calculateScreenDistance(landmarks: List<NormalizedLandmark>): Float? {
        // This is a placeholder. In a real implementation, you would:
        // 1. Use depth information if available from a depth camera
        // 2. Or use the size of known facial features (like inter-pupillary distance)
        //    to estimate distance based on perspective
        return null
    }

    private fun calculateFatigueMetrics(landmarks: List<NormalizedLandmark>): FatigueMetrics {
         if (landmarks.size <= max(LEFT_EYE_TOP, RIGHT_EYE_BOTTOM)) { // Basic check for sufficient landmarks
             Log.e("FacePoseAnalyzer", "Not enough landmarks for fatigue calculation.")
              return FatigueMetrics(0f, 0f, 0f, false, 0, 0f) // Default metrics
         }
        // Calculate eye openness
        val leftEyeOpenness = calculateEyeOpenness(
            landmarks[LEFT_EYE_TOP],
            landmarks[LEFT_EYE_BOTTOM]
        )
        val rightEyeOpenness = calculateEyeOpenness(
            landmarks[RIGHT_EYE_TOP],
            landmarks[RIGHT_EYE_BOTTOM]
        )
        val averageEyeOpenness = (leftEyeOpenness + rightEyeOpenness) / 2f

        // Update eye openness history
        eyeOpennessHistory.add(averageEyeOpenness)
        if (eyeOpennessHistory.size > 30) { // Keep last 30 frames
            eyeOpennessHistory.removeAt(0)
        }

        // Detect blinks
        val currentTime = System.currentTimeMillis()
        if (averageEyeOpenness < EYE_OPEN_THRESHOLD) {
            if (currentTime - lastBlinkTime > BLINK_DURATION_THRESHOLD * 1000) {
                blinkCount++
                lastBlinkTime = currentTime
            }
        }

        // Calculate fatigue score
        fatigueScore = calculateFatigueScore(
            averageEyeOpenness,
            eyeOpennessHistory,
            blinkCount
        )

        return FatigueMetrics(
            leftEyeOpenness = leftEyeOpenness,
            rightEyeOpenness = rightEyeOpenness,
            averageEyeOpenness = averageEyeOpenness,
            isFatigued = fatigueScore > FATIGUE_THRESHOLD, // Use correct threshold
            blinkCount = blinkCount,
            fatigueScore = fatigueScore
        )
    }

    private fun calculateEyeOpenness(top: NormalizedLandmark, bottom: NormalizedLandmark): Float {
        // Calculate distance between top and bottom eye landmarks
        val dy = bottom.y() - top.y()
        val dx = bottom.x() - top.x() // Consider horizontal distance too for a more robust measure
        val eyeHeight = sqrt(dx * dx + dy * dy)

        // Using the vertical distance as a simple proxy for openness for now.
        return eyeHeight
    }

    private fun calculateFatigueScore(
        currentOpenness: Float,
        opennessHistory: List<Float>,
        blinkCount: Int
    ): Float {
        // Calculate average eye openness over time
        val averageOpenness = if (opennessHistory.isNotEmpty()) opennessHistory.average().toFloat() else 0f

        // Calculate variance in eye openness
        val variance = if (opennessHistory.size > 1) opennessHistory.map { (it - averageOpenness) * (it - averageOpenness) }.average().toFloat() else 0f

        // A simple approach: fatigue increases as average openness decreases and variance increases.
        // Incorporate blink count as well.

        val opennessFactor = (1f - (currentOpenness / (if (averageOpenness > 0) averageOpenness * 1.2f else 1f))).coerceIn(0f, 1f) // Compare to recent average
        val varianceFactor = (variance * 1000f).coerceIn(0f, 1f) // Scale variance
        val blinkFactor = (blinkCount.toFloat() / 50f).coerceIn(0f, 1f) // Assume 50 blinks is high fatigue

        // Weighted combination (adjust weights as needed)
        return (opennessFactor * 0.4f + varianceFactor * 0.3f + blinkFactor * 0.3f).coerceIn(0f, 1f)
    }

    // New function to calculate head pose based on v2 logic
    private fun calculateFacePoseV2(
        landmarks: List<NormalizedLandmark>,
        imageWidth: Int,
        imageHeight: Int
    ): Triple<Float, Float, Float> {

        // Ensure required landmarks exist
        if (landmarks.size <= 263) {
            return Triple(0f, 0f, 0f) // Return zero angles if landmarks are insufficient
        }

        val noseTip = landmarks[1]
        val chin = landmarks[152]
        val leftEye = landmarks[33]
        val rightEye = landmarks[263]

        // Convert normalized landmarks to image coordinates for calculation
        val noseTipImg = Triple(noseTip.x() * imageWidth, noseTip.y() * imageHeight, noseTip.z())
        val chinImg = Triple(chin.x() * imageWidth, chin.y() * imageHeight, chin.z())
        val leftEyeImg = Triple(leftEye.x() * imageWidth, leftEye.y() * imageHeight, leftEye.z())
        val rightEyeImg = Triple(rightEye.x() * imageWidth, rightEye.y() * imageHeight, rightEye.z())

        // Calculate angles based on provided Python logic
        // Roll: atan2(delta_y_eyes, delta_x_eyes)
        val rollRad = atan2(
            (rightEyeImg.second - leftEyeImg.second),
            (rightEyeImg.first - leftEyeImg.first)
        )
        // Pitch: atan2(delta_y_chin_nose, delta_z_chin_nose) - Note: Z is typically inverted in screen space
        // Python code uses chin[2] - nose_tip[2] which implies Z increases away from camera. MediaPipe Z seems to increase towards camera.
        // Let's use noseTip.z() - chin.z() to align with a Z axis pointing away from the camera.
        val pitchRad = atan2(
            (chinImg.second - noseTipImg.second),
            (noseTipImg.third - chinImg.third) // Adjusted Z difference
        )
        // Yaw: atan2(delta_x_nose_chin, delta_z_nose_chin) - Note: Z is typically inverted in screen space
        // Python code uses nose_tip[2] - chin[2] for Z difference, same as Pitch.
        // Python code uses nose_tip[0] - chin[0] for X difference.
        // Let's use chinImg.first - noseTipImg.first for X difference (consistent direction check) and noseTipImg.third - chinImg.third for Z difference.
        val yawRad = atan2(
            (chinImg.first - noseTipImg.first), // Adjusted X difference
            (noseTipImg.third - chinImg.third) // Adjusted Z difference
        )

        // Convert radians to degrees
        val rollDeg = Math.toDegrees(rollRad.toDouble()).toFloat()
        // Adjust Pitch and Yaw based on the coordinate system used in the Python example.
        // The Python example's atan2 arguments for Pitch and Yaw suggest a different coordinate system or interpretation.
        // Let's start with the direct translation and adjust if the angles are inverted or offset.
        val pitchDeg = Math.toDegrees(pitchRad.toDouble()).toFloat()
        val yawDeg = Math.toDegrees(yawRad.toDouble()).toFloat()

        // The Python yaw calculation might be effectively `atan2(chin_x - nose_x, chin_z - nose_z)`
        // and pitch might be `atan2(nose_y - chin_y, nose_z - chin_z)`. Let's re-check the Python code.
        // Python: pitch = np.arctan2(chin[1] - nose_tip[1], chin[2] - nose_tip[2]) # delta Y, delta Z
        // Python: yaw = np.arctan2(nose_tip[0] - chin[0], nose_tip[2] - chin[2]) # delta X, delta Z
        // It seems the Z difference in Python is `chin[2] - nose_tip[2]` for pitch and `nose_tip[2] - chin[2]` for yaw.
        // This is inconsistent. Let's assume one is correct and apply to both.
        // Standard computer graphics: +X right, +Y down, +Z into screen.
        // Pitch (rotation around X): Looking down is positive pitch. If Y increases downwards, chin_y > nose_y means negative pitch. Z increases towards camera means chin_z < nose_z. atan2(dy, dz). If +Y down, +Z into screen, atan2(chin_y-nose_y, chin_z-nose_z) would give angle in YZ plane.
        // Yaw (rotation around Y): Looking right is positive yaw. If +X right, +Z into screen, atan2(dx, dz). atan2(nose_x-chin_x, nose_z-chin_z) would give angle in XZ plane.
        // Roll (rotation around Z): Tilting head right is positive roll. If +X right, +Y down, atan2(dy, dx). atan2(right_eye_y - left_eye_y, right_eye_x - left_eye_x) seems correct for roll around Z.

        // Let's trust the user's provided Python code's atan2 arguments directly.
        // Pitch: atan2(delta_y_chin_nose, delta_z_chin_nose) => atan2(chin_y - nose_y, chin_z - nose_z)
        // Yaw: atan2(delta_x_nose_chin, delta_z_nose_chin) => atan2(nose_x - chin_x, nose_z - chin_z)

        val pitchRadCorrected = atan2(
            (chinImg.second - noseTipImg.second), // delta Y
            (chinImg.third - noseTipImg.third) // delta Z (using chin - nose as in Python for Pitch)
        )

        val yawRadCorrected = atan2(
            (noseTipImg.first - chinImg.first), // delta X
            (noseTipImg.third - chinImg.third) // delta Z (using nose - chin as in Python for Yaw)
        )

        val pitchDegCorrected = Math.toDegrees(pitchRadCorrected.toDouble()).toFloat()
        val yawDegCorrected = Math.toDegrees(yawRadCorrected.toDouble()).toFloat()

        // The Python example seems to have inconsistent Z differences for pitch and yaw.
        // Let's assume a standard coordinate system (+X right, +Y up, +Z away from camera)
        // MediaPipe landmarks: +X right, +Y down, +Z towards camera
        // To convert MediaPipe to a standard (+Y up, +Z away) system:
        // standard_y = -mediapipe_y
        // standard_z = -mediapipe_z
        // standard_x = mediapipe_x

        // Roll (around Z): atan2(std_dy, std_dx) = atan2(-mp_dy, mp_dx) = atan2(-(right_y-left_y), right_x-left_x) = atan2(left_y-right_y, right_x-left_x)
        // This is atan2(delta_y, delta_x) from left to right eye, which is consistent with Python roll logic.
        // Pitch (around X): Looking up is positive pitch. atan2(std_dy, std_dz) = atan2(-mp_dy, -mp_dz) = atan2(mp_dy, mp_dz)
        // atan2(nose_y - chin_y, nose_z - chin_z) using MediaPipe coords.
        // Yaw (around Y): Looking right is positive yaw. atan2(std_dx, std_dz) = atan2(mp_dx, -mp_dz)
        // atan2(nose_x - chin_x, chin_z - nose_z) using MediaPipe coords.

        // Let's re-implement using MediaPipe coordinates directly based on expected rotation directions.
        // Pitch (rotation around X): +ve pitch is looking up. Use nose-to-chin vector. atan2(delta_y, delta_z)
        // Y increases downwards in MediaPipe. Z increases towards camera.
        // Vector from chin to nose: (nose_x-chin_x, nose_y-chin_y, nose_z-chin_z)
        // Pitch: rotation in YZ plane. atan2(delta_y, delta_z). If +Y down, +Z towards camera, looking up means nose_y < chin_y (neg dy), nose_z > chin_z (pos dz). atan2(neg, pos) is Q2 angle.
        // Let's try atan2(chin_y - nose_y, nose_z - chin_z)
        val pitchRadMP = atan2(
            (chinImg.second - noseTipImg.second),
            (noseTipImg.third - chinImg.third)
        )

        // Yaw (rotation around Y): +ve yaw is looking right. Use chin-to-nose vector. atan2(delta_x, delta_z)
        // X increases rightwards in MediaPipe. Z increases towards camera.
        // Yaw: rotation in XZ plane. atan2(delta_x, delta_z). If +X right, +Z towards camera, looking right means nose_x > chin_x (pos dx), nose_z > chin_z (pos dz). atan2(pos, pos) is Q1 angle.
        // Let's try atan2(nose_x - chin_x, nose_z - chin_z)
         val yawRadMP = atan2(
            (noseTipImg.first - chinImg.first),
            (noseTipImg.third - chinImg.third)
         )

        // Roll (rotation around Z): +ve roll is tilting right. Use left-to-right eye vector. atan2(delta_y, delta_x)
        // Y increases downwards, X increases rightwards.
        // Vector from left eye to right eye: (right_x-left_x, right_y-left_y, right_z-left_z)
        // Roll: rotation in XY plane. atan2(delta_y, delta_x). Tilting right means right_y > left_y (pos dy), right_x > left_x (pos dx). atan2(pos, pos) is Q1 angle.
        // This matches the user's Python roll calculation.
        val rollRadMP = atan2(
            (rightEyeImg.second - leftEyeImg.second),
            (rightEyeImg.first - leftEyeImg.first)
        )

        val pitchDegMP = Math.toDegrees(pitchRadMP.toDouble()).toFloat()
        val yawDegMP = Math.toDegrees(yawRadMP.toDouble()).toFloat()
        val rollDegMP = Math.toDegrees(rollRadMP.toDouble()).toFloat()

        // The Python code's Z difference for Pitch was chin-nose, for Yaw was nose-chin.
        // Let's try the Python Z differences exactly, but using MediaPipe coords
        val pitchRadPy = atan2(
             (chinImg.second - noseTipImg.second), // delta Y (chin - nose)
             (chinImg.third - noseTipImg.third) // delta Z (chin - nose)
        )
        val yawRadPy = atan2(
            (noseTipImg.first - chinImg.first), // delta X (nose - chin)
            (noseTipImg.third - chinImg.third) // delta Z (nose - chin)
        )

        val pitchDegPy = Math.toDegrees(pitchRadPy.toDouble()).toFloat()
        val yawDegPy = Math.toDegrees(yawRadPy.toDouble()).toFloat()
        val rollDegPy = rollDegMP // Roll calculation was consistent

        // Let's use the angles derived directly from the Python atan2 arguments, adjusting for MediaPipe's +Z towards camera.
        // Python Pitch: atan2(chin_y - nose_y, chin_z - nose_z)
        // Python Yaw: atan2(nose_x - chin_x, nose_z - chin_z)
        // MediaPipe Z: increases towards camera. Let's use (nose_z - chin_z) as the Z difference magnitude
        // And rely on atan2 to give the correct sign based on coordinate system.

        val pitchRadFinal = atan2(
            (chinImg.second - noseTipImg.second), // Y diff (chin - nose)
            (chinImg.third - noseTipImg.third) // Z diff (chin - nose)
        )

        val yawRadFinal = atan2(
            (noseTipImg.first - chinImg.first), // X diff (nose - chin)
            (noseTipImg.third - chinImg.third) // Z diff (nose - chin)
        )

        val pitchDegFinal = Math.toDegrees(pitchRadFinal.toDouble()).toFloat()
        val yawDegFinal = Math.toDegrees(yawRadFinal.toDouble()).toFloat()
        val rollDegFinal = rollDegMP // Roll calculation is consistent

        // Note: The exact mapping of atan2 results to Euler angles depends heavily on the chosen coordinate system and rotation order.
        // The Python code's calculation might assume a specific coordinate system that doesn't directly map to MediaPipe's.
        // We are implementing the atan2 calls as written in Python, using MediaPipe coordinates scaled by image dimensions.
        // The resulting angles might require sign inversion or offset depending on how they are interpreted visually.

        return Triple(yawDegFinal, pitchDegFinal, rollDegFinal)
    }
} 