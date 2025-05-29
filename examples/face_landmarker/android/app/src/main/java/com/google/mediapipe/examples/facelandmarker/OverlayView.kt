package com.google.mediapipe.examples.facelandmarker

/*
 * Copyright 2023 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *             http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View
import androidx.core.content.ContextCompat
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarkerResult
import kotlin.math.max
import kotlin.math.min

class OverlayView(context: Context?, attrs: AttributeSet?) :
    View(context, attrs) {

    private var results: FaceLandmarkerResult? = null
    private var linePaint = Paint()
    private var pointPaint = Paint()
    private var textPaint = Paint()
    private val facePoseAnalyzer = FacePoseAnalyzer()

    private var scaleFactor: Float = 1f
    private var imageWidth: Int = 1
    private var imageHeight: Int = 1

    private val EYE_OPEN_THRESHOLD = 0.25f    // Threshold for eye openness
    private val BLINK_DURATION_THRESHOLD = 0.3f  // Seconds
    private val FATIGUE_THRESHOLD = 0.5f      // Threshold for fatigue detection

    private val faceColors = arrayOf(
        Color.rgb(255, 200, 200),  // Light Red
        Color.rgb(200, 255, 200),  // Light Green
        Color.rgb(200, 200, 255),  // Light Blue
        Color.rgb(255, 255, 200),  // Light Yellow
        Color.rgb(255, 200, 255)   // Light Purple
    )

    init {
        initPaints()
    }

    fun clear() {
        results = null
        linePaint.reset()
        pointPaint.reset()
        textPaint.reset()
        invalidate()
        initPaints()
    }

    private fun initPaints() {
        linePaint.color =
            ContextCompat.getColor(context!!, R.color.mp_color_primary)
        linePaint.strokeWidth = LANDMARK_STROKE_WIDTH
        linePaint.style = Paint.Style.STROKE

        pointPaint.color = Color.YELLOW
        pointPaint.strokeWidth = LANDMARK_STROKE_WIDTH
        pointPaint.style = Paint.Style.FILL

        textPaint.color = Color.BLACK
        textPaint.textSize = 20f
        textPaint.isAntiAlias = true
    }

    override fun draw(canvas: Canvas) {
        super.draw(canvas)

        if (results?.faceLandmarks().isNullOrEmpty()) {
            clear()
            return
        }

        results?.let { faceLandmarkerResult ->
            val scaledImageWidth = imageWidth * scaleFactor
            val scaledImageHeight = imageHeight * scaleFactor
            val offsetX = (width - scaledImageWidth) / 2f
            val offsetY = (height - scaledImageHeight) / 2f

            // Draw face number indicator
            textPaint.textSize = 40f
            canvas.drawText(
                "Detected Faces: ${faceLandmarkerResult.faceLandmarks().size}",
                width * 0.05f,  // Position at 5% of screen width
                50f,
                textPaint
            )
            textPaint.textSize = 30f  // Reset text size

            // Draw information for each face
            faceLandmarkerResult.faceLandmarks().forEachIndexed { index, faceLandmarks ->
                // Draw landmarks and connectors
                drawFaceLandmarks(canvas, faceLandmarks, offsetX, offsetY)
                drawFaceConnectors(canvas, faceLandmarks, offsetX, offsetY)

                // Draw face number near the face with matching color
                val noseTip = faceLandmarks[1]  // Nose tip landmark
                val faceNumberX = noseTip.x() * imageWidth * scaleFactor + offsetX
                val faceNumberY = noseTip.y() * imageHeight * scaleFactor + offsetY - 50f
                
                // Draw colored background for face number
                val numberBackgroundPaint = Paint().apply {
                    color = faceColors[index % faceColors.size]
                    alpha = 180
                    style = Paint.Style.FILL
                }
                canvas.drawRect(
                    faceNumberX - 10f,
                    faceNumberY - 30f,
                    faceNumberX + 100f,
                    faceNumberY + 10f,
                    numberBackgroundPaint
                )
                
                textPaint.color = Color.BLACK
                canvas.drawText(
                    "Face ${index + 1}",
                    faceNumberX,
                    faceNumberY,
                    textPaint
                )

                // Analyze and draw face pose information
                val facePose = facePoseAnalyzer.analyzeFacePose(faceLandmarks)
                drawFacePoseInfo(canvas, facePose, offsetX, offsetY, index)
            }
        }
    }

    /**
     * Draws all landmarks for a single face on the canvas.
     */
    private fun drawFaceLandmarks(
        canvas: Canvas,
        faceLandmarks: List<NormalizedLandmark>,
        offsetX: Float,
        offsetY: Float
    ) {
        // Draw all face landmarks
        faceLandmarks.forEach { landmark ->
            val x = landmark.x() * imageWidth * scaleFactor + offsetX
            val y = landmark.y() * imageHeight * scaleFactor + offsetY
            canvas.drawPoint(x, y, pointPaint)
        }

        // Draw head pose axes on the face
        drawHeadPoseOnFace(canvas, faceLandmarks, offsetX, offsetY)
        
        // Draw gaze direction on the face
        drawGazeOnFace(canvas, faceLandmarks, offsetX, offsetY)
    }

    private fun drawHeadPoseOnFace(
        canvas: Canvas,
        faceLandmarks: List<NormalizedLandmark>,
        offsetX: Float,
        offsetY: Float
    ) {
        // Get nose tip and forehead points for reference
        val noseTip = faceLandmarks[1]  // Nose tip landmark
        val forehead = faceLandmarks[10]  // Forehead landmark
        
        val noseX = noseTip.x() * imageWidth * scaleFactor + offsetX
        val noseY = noseTip.y() * imageHeight * scaleFactor + offsetY
        val foreheadX = forehead.x() * imageWidth * scaleFactor + offsetX
        val foreheadY = forehead.y() * imageHeight * scaleFactor + offsetY

        // Calculate face orientation
        val facePose = facePoseAnalyzer.analyzeFacePose(faceLandmarks)
        
        // Define colors for each rotation axis
        val yawColor = Color.rgb(255, 100, 100)    // Red for yaw (左右)
        val pitchColor = Color.rgb(100, 255, 100)  // Green for pitch (上下)
        val rollColor = Color.rgb(100, 100, 255)   // Blue for roll (旋轉)
        
        // Draw coordinate axes
        val axisLength = 100f * scaleFactor
        val axisPaint = Paint().apply {
            strokeWidth = 5f
            isAntiAlias = true
        }

        // Save canvas state
        canvas.save()
        canvas.translate(noseX, noseY)
        canvas.rotate(facePose.roll)

        // Draw Yaw axis (red) - horizontal rotation
        axisPaint.color = yawColor
        canvas.drawLine(0f, 0f, axisLength, 0f, axisPaint)
        
        // Draw Pitch axis (green) - vertical rotation
        axisPaint.color = pitchColor
        canvas.drawLine(0f, 0f, 0f, -axisLength, axisPaint)
        
        // Draw Roll axis (blue) - rotation around Z
        axisPaint.color = rollColor
        canvas.drawLine(0f, 0f, axisLength * 0.5f, axisLength * 0.5f, axisPaint)

        // Restore canvas state
        canvas.restore()

        // Draw color-coded orientation text with background
        val textBackgroundPaint = Paint().apply {
            color = Color.argb(180, 0, 0, 0)  // Semi-transparent black background
            style = Paint.Style.FILL
        }

        // Draw background for text
        val textX = noseX - 120f * scaleFactor
        val textY = noseY - 60f * scaleFactor
        val textWidth = 240f * scaleFactor
        val textHeight = 90f * scaleFactor
        canvas.drawRect(
            textX,
            textY - textHeight,
            textX + textWidth,
            textY,
            textBackgroundPaint
        )

        // Draw yaw text (red)
        textPaint.textSize = 25f * scaleFactor
        textPaint.color = yawColor
        canvas.drawText(
            "Yaw: ${String.format("%.1f", facePose.yaw)}°",
            textX + 10f * scaleFactor,
            textY - 40f * scaleFactor,
            textPaint
        )

        // Draw pitch text (green)
        textPaint.color = pitchColor
        canvas.drawText(
            "Pitch: ${String.format("%.1f", facePose.pitch)}°",
            textX + 10f * scaleFactor,
            textY - 20f * scaleFactor,
            textPaint
        )

        // Draw roll text (blue)
        textPaint.color = rollColor
        canvas.drawText(
            "Roll: ${String.format("%.1f", facePose.roll)}°",
            textX + 10f * scaleFactor,
            textY,
            textPaint
        )

        // Draw small color indicators next to the axes
        val indicatorRadius = 8f * scaleFactor
        val indicatorPaint = Paint().apply {
            style = Paint.Style.FILL
            isAntiAlias = true
        }

        // Yaw indicator (red)
        indicatorPaint.color = yawColor
        canvas.drawCircle(noseX + axisLength + 20f * scaleFactor, noseY, indicatorRadius, indicatorPaint)

        // Pitch indicator (green)
        indicatorPaint.color = pitchColor
        canvas.drawCircle(noseX, noseY - axisLength - 20f * scaleFactor, indicatorRadius, indicatorPaint)

        // Roll indicator (blue)
        indicatorPaint.color = rollColor
        canvas.drawCircle(noseX + axisLength * 0.5f + 20f * scaleFactor, noseY + axisLength * 0.5f + 20f * scaleFactor, indicatorRadius, indicatorPaint)
    }

    private fun drawGazeOnFace(
        canvas: Canvas,
        faceLandmarks: List<NormalizedLandmark>,
        offsetX: Float,
        offsetY: Float
    ) {
        val facePose = facePoseAnalyzer.analyzeFacePose(faceLandmarks)
        
        // Get eye landmarks
        val leftEye = facePose.eyePositions.leftEye
        val rightEye = facePose.eyePositions.rightEye
        val leftIris = facePose.irisPositions.leftIris
        val rightIris = facePose.irisPositions.rightIris

        // Calculate eye positions
        val leftEyeX = leftEye.x * imageWidth * scaleFactor + offsetX
        val leftEyeY = leftEye.y * imageHeight * scaleFactor + offsetY
        val rightEyeX = rightEye.x * imageWidth * scaleFactor + offsetX
        val rightEyeY = rightEye.y * imageHeight * scaleFactor + offsetY

        // Draw eye circles
        val eyePaint = Paint().apply {
            color = Color.WHITE
            style = Paint.Style.STROKE
            strokeWidth = 3f * scaleFactor
            isAntiAlias = true
        }

        val eyeRadius = 15f * scaleFactor
        canvas.drawCircle(leftEyeX, leftEyeY, eyeRadius, eyePaint)
        canvas.drawCircle(rightEyeX, rightEyeY, eyeRadius, eyePaint)

        // Draw gaze direction
        val gazePaint = Paint().apply {
            color = Color.YELLOW
            strokeWidth = 3f * scaleFactor
            isAntiAlias = true
        }

        val gazeLength = 50f * scaleFactor
        canvas.drawLine(
            leftEyeX,
            leftEyeY,
            leftEyeX + (leftIris.x - 0.5f) * gazeLength,
            leftEyeY + (leftIris.y - 0.5f) * gazeLength,
            gazePaint
        )
        canvas.drawLine(
            rightEyeX,
            rightEyeY,
            rightEyeX + (rightIris.x - 0.5f) * gazeLength,
            rightEyeY + (rightIris.y - 0.5f) * gazeLength,
            gazePaint
        )
    }

    /**
     * Draws all the connectors between landmarks for a single face on the canvas.
     */
    private fun drawFaceConnectors(
        canvas: Canvas,
        faceLandmarks: List<NormalizedLandmark>,
        offsetX: Float,
        offsetY: Float
    ) {
        FaceLandmarker.FACE_LANDMARKS_CONNECTORS.filterNotNull().forEach { connector ->
            val startLandmark = faceLandmarks.getOrNull(connector.start())
            val endLandmark = faceLandmarks.getOrNull(connector.end())

            if (startLandmark != null && endLandmark != null) {
                val startX = startLandmark.x() * imageWidth * scaleFactor + offsetX
                val startY = startLandmark.y() * imageHeight * scaleFactor + offsetY
                val endX = endLandmark.x() * imageWidth * scaleFactor + offsetX
                val endY = endLandmark.y() * imageHeight * scaleFactor + offsetY

                canvas.drawLine(startX, startY, endX, endY, linePaint)
            }
        }
    }

    private fun drawFacePoseInfo(canvas: Canvas, facePose: FacePoseAnalyzer.FacePose, offsetX: Float, offsetY: Float, faceIndex: Int) {
        // Calculate base text position based on screen width and face index
        val columnWidth = width * 0.20f  // Each face gets 25% of screen width
        val baseTextX = width * 0.68f + (columnWidth * faceIndex)  // Start at 75% and increment by column width
        val baseTextY = 50f
        val lineHeight = 40f
        val textSpacing = 20f  // Additional spacing between different faces

        // Draw background for face information
        val backgroundColor = faceColors[faceIndex % faceColors.size]
        val backgroundPaint = Paint().apply {
            color = backgroundColor
            alpha = 180  // Semi-transparent
            style = Paint.Style.FILL
        }

        // Draw background rectangle
        canvas.drawRect(
            baseTextX - 10f,
            baseTextY - 30f,
            baseTextX + columnWidth - 20f,
            baseTextY + lineHeight * 14f,
            backgroundPaint
        )

        // Draw face number header with border
        textPaint.textSize = 35f
        textPaint.color = Color.BLACK
        canvas.drawText(
            "Face ${faceIndex + 1}",
            baseTextX,
            baseTextY,
            textPaint
        )
        textPaint.textSize = 30f  // Reset text size

        // Draw head pose visualization
        val headPoseX = baseTextX
        val headPoseY = baseTextY + lineHeight * 2
        drawHeadPoseVisualization(canvas, facePose, headPoseX, headPoseY)
        
        // Draw gaze visualization
        val gazeX = baseTextX
        val gazeY = baseTextY + lineHeight * 8
        drawGazeVisualization(canvas, facePose, gazeX, gazeY)

        // Draw head pose angles
        canvas.drawText(
            "Yaw: ${String.format("%.1f", facePose.yaw)}°",
            baseTextX,
            baseTextY + lineHeight * 3,
            textPaint
        )
        canvas.drawText(
            "Pitch: ${String.format("%.1f", facePose.pitch)}°",
            baseTextX,
            baseTextY + lineHeight * 4,
            textPaint
        )
        canvas.drawText(
            "Roll: ${String.format("%.1f", facePose.roll)}°",
            baseTextX,
            baseTextY + lineHeight * 5,
            textPaint
        )

        // Draw eye positions
        val leftEye = facePose.eyePositions.leftEye
        val rightEye = facePose.eyePositions.rightEye
        canvas.drawText(
            "Left Eye: (${String.format("%.2f", leftEye.x)}, ${String.format("%.2f", leftEye.y)})",
            baseTextX,
            baseTextY + lineHeight * 6,
            textPaint
        )
        canvas.drawText(
            "Right Eye: (${String.format("%.2f", rightEye.x)}, ${String.format("%.2f", rightEye.y)})",
            baseTextX,
            baseTextY + lineHeight * 7,
            textPaint
        )

        // Draw iris positions
        val leftIris = facePose.irisPositions.leftIris
        val rightIris = facePose.irisPositions.rightIris
        canvas.drawText(
            "Left Iris: (${String.format("%.2f", leftIris.x)}, ${String.format("%.2f", leftIris.y)})",
            baseTextX,
            baseTextY + lineHeight * 8,
            textPaint
        )
        canvas.drawText(
            "Right Iris: (${String.format("%.2f", rightIris.x)}, ${String.format("%.2f", rightIris.y)})",
            baseTextX,
            baseTextY + lineHeight * 9,
            textPaint
        )

        // Draw screen distance if available
        facePose.screenDistance?.let { distance ->
            canvas.drawText(
                "Screen Distance: ${String.format("%.2f", distance)}",
                baseTextX,
                baseTextY + lineHeight * 10,
                textPaint
            )
        }

        // Draw fatigue metrics
        facePose.fatigueMetrics?.let { metrics ->
            // Set text color based on fatigue status
            textPaint.color = if (metrics.isFatigued) Color.RED else Color.GREEN

            canvas.drawText(
                "Eye Openness: ${String.format("%.2f", metrics.averageEyeOpenness)}",
                baseTextX,
                baseTextY + lineHeight * 11,
                textPaint
            )
            canvas.drawText(
                "Blink Count: ${metrics.blinkCount}",
                baseTextX,
                baseTextY + lineHeight * 12,
                textPaint
            )
            canvas.drawText(
                "Fatigue Score: ${String.format("%.2f", metrics.fatigueScore)}",
                baseTextX,
                baseTextY + lineHeight * 13,
                textPaint
            )
            canvas.drawText(
                "Status: ${if (metrics.isFatigued) "FATIGUED" else "ALERT"}",
                baseTextX,
                baseTextY + lineHeight * 14,
                textPaint
            )

            // Reset text color
            textPaint.color = Color.BLACK
        }
    }

    private fun drawHeadPoseVisualization(canvas: Canvas, facePose: FacePoseAnalyzer.FacePose, startX: Float, startY: Float) {
        val size = 100f
        val centerX = startX + size / 2
        val centerY = startY + size / 2

        // Draw coordinate system
        val axisPaint = Paint().apply {
            strokeWidth = 3f
            isAntiAlias = true
        }

        // X-axis (red)
        axisPaint.color = Color.RED
        canvas.drawLine(centerX, centerY, centerX + size * 0.8f, centerY, axisPaint)
        
        // Y-axis (green)
        axisPaint.color = Color.GREEN
        canvas.drawLine(centerX, centerY, centerX, centerY - size * 0.8f, axisPaint)
        
        // Z-axis (blue)
        axisPaint.color = Color.BLUE
        canvas.drawLine(centerX, centerY, centerX + size * 0.4f, centerY + size * 0.4f, axisPaint)

        // Draw rotated coordinate system based on head pose
        canvas.save()
        canvas.translate(centerX, centerY)
        canvas.rotate(facePose.roll)
        canvas.scale(
            1f - facePose.pitch / 90f,
            1f - facePose.yaw / 90f
        )
        
        // Draw rotated axes
        axisPaint.alpha = 128
        axisPaint.color = Color.RED
        canvas.drawLine(0f, 0f, size * 0.8f, 0f, axisPaint)
        axisPaint.color = Color.GREEN
        canvas.drawLine(0f, 0f, 0f, -size * 0.8f, axisPaint)
        axisPaint.color = Color.BLUE
        canvas.drawLine(0f, 0f, size * 0.4f, size * 0.4f, axisPaint)
        
        canvas.restore()
    }

    private fun drawGazeVisualization(canvas: Canvas, facePose: FacePoseAnalyzer.FacePose, startX: Float, startY: Float) {
        val size = 100f
        val centerX = startX + size / 2
        val centerY = startY + size / 2

        // Draw eye positions
        val eyePaint = Paint().apply {
            color = Color.BLUE
            style = Paint.Style.FILL
            isAntiAlias = true
        }

        // Draw left eye
        val leftEyeX = centerX - size * 0.3f
        val leftEyeY = centerY
        canvas.drawCircle(leftEyeX, leftEyeY, 10f, eyePaint)

        // Draw right eye
        val rightEyeX = centerX + size * 0.3f
        val rightEyeY = centerY
        canvas.drawCircle(rightEyeX, rightEyeY, 10f, eyePaint)

        // Draw gaze direction
        val gazePaint = Paint().apply {
            color = Color.RED
            strokeWidth = 3f
            isAntiAlias = true
        }

        // Calculate gaze direction based on iris positions
        val leftIris = facePose.irisPositions.leftIris
        val rightIris = facePose.irisPositions.rightIris
        
        // Draw gaze lines
        val gazeLength = size * 0.4f
        canvas.drawLine(
            leftEyeX,
            leftEyeY,
            leftEyeX + (leftIris.x - 0.5f) * gazeLength,
            leftEyeY + (leftIris.y - 0.5f) * gazeLength,
            gazePaint
        )
        canvas.drawLine(
            rightEyeX,
            rightEyeY,
            rightEyeX + (rightIris.x - 0.5f) * gazeLength,
            rightEyeY + (rightIris.y - 0.5f) * gazeLength,
            gazePaint
        )
    }

    fun setResults(
        faceLandmarkerResults: FaceLandmarkerResult,
        imageHeight: Int,
        imageWidth: Int,
        runningMode: RunningMode = RunningMode.IMAGE
    ) {
        results = faceLandmarkerResults

        this.imageHeight = imageHeight
        this.imageWidth = imageWidth

        scaleFactor = when (runningMode) {
            RunningMode.IMAGE,
            RunningMode.VIDEO -> {
                min(width * 1f / imageWidth, height * 1f / imageHeight)
            }
            RunningMode.LIVE_STREAM -> {
                // PreviewView is in FILL_START mode. So we need to scale up the
                // landmarks to match with the size that the captured images will be
                // displayed.
                max(width * 1f / imageWidth, height * 1f / imageHeight)
            }
        }
        invalidate()
    }

    companion object {
        private const val LANDMARK_STROKE_WIDTH = 4F
        private const val TAG = "Face Landmarker Overlay"
    }
}
