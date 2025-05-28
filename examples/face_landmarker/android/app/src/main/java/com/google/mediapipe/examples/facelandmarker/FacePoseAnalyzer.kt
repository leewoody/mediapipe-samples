package com.google.mediapipe.examples.facelandmarker

import android.graphics.PointF
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import kotlin.math.*

class FacePoseAnalyzer {
    // Face landmark indices for MediaPipe Face Landmarker
    companion object {
        // Face contour points
        private const val NOSE_TIP = 1
        private const val LEFT_EYE = 33
        private const val RIGHT_EYE = 263
        private const val LEFT_MOUTH = 61
        private const val RIGHT_MOUTH = 291
        private const val LEFT_EAR = 234
        private const val RIGHT_EAR = 454
        private const val CHIN = 152

        // Eye landmarks
        private val LEFT_EYE_LANDMARKS = listOf(33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246)
        private val RIGHT_EYE_LANDMARKS = listOf(263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466)
        
        // Iris landmarks
        private val LEFT_IRIS_LANDMARKS = listOf(468, 469, 470, 471, 472)
        private val RIGHT_IRIS_LANDMARKS = listOf(473, 474, 475, 476, 477)

        // Eye landmarks for openness calculation
        private val LEFT_EYE_TOP = 159
        private val LEFT_EYE_BOTTOM = 145
        private val RIGHT_EYE_TOP = 386
        private val RIGHT_EYE_BOTTOM = 374

        // Fatigue detection constants
        private const val EYE_OPEN_THRESHOLD = 0.25f  // Threshold for eye openness
        private const val BLINK_DURATION_THRESHOLD = 0.3f  // Seconds
        private const val FATIGUE_THRESHOLD = 0.5f  // Threshold for fatigue detection
    }

    // Fatigue detection state
    private var lastBlinkTime = 0L
    private var blinkCount = 0
    private var fatigueScore = 0f
    private var eyeOpennessHistory = mutableListOf<Float>()

    data class FacePose(
        val yaw: Float,      // Left-right rotation
        val pitch: Float,    // Up-down rotation
        val roll: Float,     // Tilt rotation
        val eyePositions: EyePositions,
        val irisPositions: IrisPositions,
        val screenDistance: Float? = null,
        val fatigueMetrics: FatigueMetrics? = null
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

    fun analyzeFacePose(landmarks: List<NormalizedLandmark>): FacePose {
        // Calculate head pose angles
        val (yaw, pitch, roll) = calculateHeadPose(landmarks)
        
        // Calculate eye positions
        val eyePositions = calculateEyePositions(landmarks)
        
        // Calculate iris positions
        val irisPositions = calculateIrisPositions(landmarks)
        
        // Calculate screen distance
        val screenDistance = calculateScreenDistance(landmarks)

        // Calculate fatigue metrics
        val fatigueMetrics = calculateFatigueMetrics(landmarks)

        return FacePose(
            yaw = yaw,
            pitch = pitch,
            roll = roll,
            eyePositions = eyePositions,
            irisPositions = irisPositions,
            screenDistance = screenDistance,
            fatigueMetrics = fatigueMetrics
        )
    }

    private fun calculateHeadPose(landmarks: List<NormalizedLandmark>): Triple<Float, Float, Float> {
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
        return atan2(noseToEyeDistance, 0.1f) * (180f / PI.toFloat())
    }

    private fun calculateYaw(leftEye: NormalizedLandmark, rightEye: NormalizedLandmark, 
                           leftMouth: NormalizedLandmark, rightMouth: NormalizedLandmark): Float {
        val eyeWidth = rightEye.x() - leftEye.x()
        val mouthWidth = rightMouth.x() - leftMouth.x()
        val ratio = eyeWidth / mouthWidth
        return (ratio - 1.0f) * 45f // Approximate yaw angle
    }

    private fun calculateEyePositions(landmarks: List<NormalizedLandmark>): EyePositions {
        val leftEyeCenter = calculateCenter(LEFT_EYE_LANDMARKS.map { landmarks[it] })
        val rightEyeCenter = calculateCenter(RIGHT_EYE_LANDMARKS.map { landmarks[it] })

        return EyePositions(
            leftEye = PointF(leftEyeCenter.x(), leftEyeCenter.y()),
            rightEye = PointF(rightEyeCenter.x(), rightEyeCenter.y())
        )
    }

    private fun calculateIrisPositions(landmarks: List<NormalizedLandmark>): IrisPositions {
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
        return NormalizedLandmark.create(x, y, 0f)
    }

    private fun calculateScreenDistance(landmarks: List<NormalizedLandmark>): Float? {
        // This is a placeholder. In a real implementation, you would:
        // 1. Use depth information if available from a depth camera
        // 2. Or use the size of known facial features (like inter-pupillary distance)
        //    to estimate distance based on perspective
        return null
    }

    private fun calculateFatigueMetrics(landmarks: List<NormalizedLandmark>): FatigueMetrics {
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
            isFatigued = fatigueScore > FATIGUE_THRESHOLD,
            blinkCount = blinkCount,
            fatigueScore = fatigueScore
        )
    }

    private fun calculateEyeOpenness(top: NormalizedLandmark, bottom: NormalizedLandmark): Float {
        val eyeHeight = abs(top.y() - bottom.y())
        val eyeWidth = abs(top.x() - bottom.x())
        return eyeHeight / eyeWidth
    }

    private fun calculateFatigueScore(
        currentOpenness: Float,
        opennessHistory: List<Float>,
        blinkCount: Int
    ): Float {
        // Calculate average eye openness over time
        val averageOpenness = opennessHistory.average().toFloat()
        
        // Calculate variance in eye openness
        val variance = opennessHistory.map { (it - averageOpenness) * (it - averageOpenness) }.average().toFloat()
        
        // Calculate blink rate (blinks per minute)
        val timeWindow = 60f // 60 seconds
        val blinkRate = blinkCount / timeWindow

        // Combine factors to get fatigue score
        val opennessScore = 1f - (currentOpenness / EYE_OPEN_THRESHOLD).coerceIn(0f, 1f)
        val varianceScore = (variance / 0.1f).coerceIn(0f, 1f)
        val blinkRateScore = (blinkRate / 20f).coerceIn(0f, 1f) // Normalize to 20 blinks per minute

        // Weighted combination of factors
        return (opennessScore * 0.4f + varianceScore * 0.3f + blinkRateScore * 0.3f)
    }
} 