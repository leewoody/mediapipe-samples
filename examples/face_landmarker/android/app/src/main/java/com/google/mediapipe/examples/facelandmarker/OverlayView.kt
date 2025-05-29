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
        faceLandmarks.forEach { landmark ->
            val x = landmark.x() * imageWidth * scaleFactor + offsetX
            val y = landmark.y() * imageHeight * scaleFactor + offsetY
            canvas.drawPoint(x, y, pointPaint)
        }
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
