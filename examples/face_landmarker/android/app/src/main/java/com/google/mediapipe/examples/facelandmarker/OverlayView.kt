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
        textPaint.textSize = 30f
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

            faceLandmarkerResult.faceLandmarks().forEach { faceLandmarks ->
                // Draw landmarks and connectors
                drawFaceLandmarks(canvas, faceLandmarks, offsetX, offsetY)
                drawFaceConnectors(canvas, faceLandmarks, offsetX, offsetY)

                // Analyze and draw face pose information
                val facePose = facePoseAnalyzer.analyzeFacePose(faceLandmarks)
                drawFacePoseInfo(canvas, facePose, offsetX, offsetY)
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

    private fun drawFacePoseInfo(canvas: Canvas, facePose: FacePoseAnalyzer.FacePose, offsetX: Float, offsetY: Float) {
        val textX = 1500f
        val textY = 50f
        val lineHeight = 40f

        // Draw head pose angles
//        canvas.drawText(
//            "Yaw: ${String.format("%.1f", facePose.yaw)}°",
//            textX,
//            textY+ lineHeight * 2,
//            textPaint
//        )
        canvas.drawText(
            "Yaw: ${String.format("%.1f", facePose.yaw)}°,\n " + "Pitch: ${String.format("%.1f", facePose.pitch)}°, \n" ,
            textX,
            textY + lineHeight * 3,
            textPaint
        )
        canvas.drawText(
            "Roll: ${String.format("%.1f", facePose.roll)}°",
            textX,
            textY + lineHeight * 4,
            textPaint
        )

        // Draw eye positions
        val leftEye = facePose.eyePositions.leftEye
        val rightEye = facePose.eyePositions.rightEye
        canvas.drawText(
            "Left Eye: (${String.format("%.2f", leftEye.x)}, ${String.format("%.2f", leftEye.y)})",
            textX,
            textY + lineHeight * 5,
            textPaint
        )
        canvas.drawText(
            "Right Eye: (${String.format("%.2f", rightEye.x)}, ${String.format("%.2f", rightEye.y)})",
            textX,
            textY + lineHeight * 6,
            textPaint
        )

        // Draw iris positions
        val leftIris = facePose.irisPositions.leftIris
        val rightIris = facePose.irisPositions.rightIris
        canvas.drawText(
            "Left Iris: (${String.format("%.2f", leftIris.x)}, ${String.format("%.2f", leftIris.y)})",
            textX,
            textY + lineHeight * 7,
            textPaint
        )
        canvas.drawText(
            "Right Iris: (${String.format("%.2f", rightIris.x)}, ${String.format("%.2f", rightIris.y)})",
            textX,
            textY + lineHeight * 8,
            textPaint
        )

        // Draw screen distance if available
        facePose.screenDistance?.let { distance ->
            canvas.drawText(
                "Screen Distance: ${String.format("%.2f", distance)}",
                textX,
                textY + lineHeight * 9,
                textPaint
            )
        }

        // Draw fatigue metrics
        facePose.fatigueMetrics?.let { metrics ->
            // Set text color based on fatigue status
            textPaint.color = if (metrics.isFatigued) Color.RED else Color.GREEN

            canvas.drawText(
                "Eye Openness: ${String.format("%.2f", metrics.averageEyeOpenness)}",
                textX,
                textY + lineHeight * 10,
                textPaint
            )
            canvas.drawText(
                "Blink Count: ${metrics.blinkCount}",
                textX,
                textY + lineHeight * 11,
                textPaint
            )
            canvas.drawText(
                "Fatigue Score: ${String.format("%.2f", metrics.fatigueScore)}",
                textX,
                textY + lineHeight * 12,
                textPaint
            )
            canvas.drawText(
                "Status: ${if (metrics.isFatigued) "FATIGUED" else "ALERT"}",
                textX,
                textY + lineHeight * 13,
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
