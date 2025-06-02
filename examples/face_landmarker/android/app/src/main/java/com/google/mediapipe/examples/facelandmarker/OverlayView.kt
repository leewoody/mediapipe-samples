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
import android.graphics.DashPathEffect
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

    private var scaleFactor: Float = 3.5f
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

    // Drawing mode flags
    var simpleDrawMode: Boolean = false
        set(value) {
            field = value
            invalidate() // Redraw when mode changes
        }
    
    // Individual component visibility flags
    var showFaceLandmarks: Boolean = true
        set(value) {
            field = value
            invalidate()
        }
    
    var showConnectors: Boolean = true
        set(value) {
            field = value
            invalidate()
        }
    
    var showHeadPoseAxes: Boolean = true
        set(value) {
            field = value
            invalidate()
        }
    
    var showGaze: Boolean = true
        set(value) {
            field = value
            invalidate()
        }
    
    var showFaceInfo: Boolean = true
        set(value) {
            field = value
            invalidate()
        }
    
    var showFacePoseInfo: Boolean = true
        set(value) {
            field = value
            invalidate()
        }
    
    var showHeadPoseVisualization: Boolean = true
        set(value) {
            field = value
            invalidate()
        }

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

        pointPaint.color = Color.BLUE
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
                drawFaceConnectors(canvas, faceLandmarks, offsetX, offsetY, index)
                drawFaceLandmarks(canvas, faceLandmarks, offsetX, offsetY, index)

                // Draw face number near the face with matching color
                val noseTip = faceLandmarks[1]  // Nose tip landmark
                val faceNumberX = noseTip.x() * imageWidth * scaleFactor + offsetX
                val faceNumberY = noseTip.y() * imageHeight * scaleFactor + offsetY - 150f
                
                // Draw colored background for face number
                // val numberBackgroundPaint = Paint().apply {
                //     color = faceColors[index % faceColors.size]
                //     alpha = 180
                //     style = Paint.Style.FILL
                // }
                // canvas.drawRect(
                //     faceNumberX - 10f,
                //     faceNumberY - 30f,
                //     faceNumberX + 100f,
                //     faceNumberY + 10f,
                //     numberBackgroundPaint
                // )

                // textPaint.color = Color.BLACK
                // canvas.drawText(
                //     "Face ${index + 1}",
                //     faceNumberX,
                //     faceNumberY,
                //     textPaint
                // )

                // Analyze and draw face pose information
                //val facePose = facePoseAnalyzer.analyzeFacePose(faceLandmarks)
                //drawFacePoseInfo(canvas, facePose, offsetX, offsetY, index)
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
        offsetY: Float,
        faceIndex: Int
    ) {
        if (!showFaceLandmarks) return

        if (simpleDrawMode) {
            // Use different color for each face
            val colorConn = faceColors[faceIndex % faceColors.size]
            // Draw connectors first
            val connectorPaint = Paint().apply {
                color = colorConn
                strokeWidth = 2f
                style = Paint.Style.STROKE
                isAntiAlias = true
            }
            FaceLandmarker.FACE_LANDMARKS_CONNECTORS.filterNotNull().forEach { connector ->
                val startLandmark = faceLandmarks.getOrNull(connector.start())
                val endLandmark = faceLandmarks.getOrNull(connector.end())
                if (startLandmark != null && endLandmark != null) {
                    val startX = startLandmark.x() * imageWidth * scaleFactor + offsetX
                    val startY = startLandmark.y() * imageHeight * scaleFactor + offsetY
                    val endX = endLandmark.x() * imageWidth * scaleFactor + offsetX
                    val endY = endLandmark.y() * imageHeight * scaleFactor + offsetY
                    canvas.drawLine(startX, startY, endX, endY, connectorPaint)
                }
            }
            // Draw landmarks after connectors
            val landmarkPaint = Paint().apply {
                color = colorConn
                style = Paint.Style.FILL
                isAntiAlias = true
            }
        faceLandmarks.forEach { landmark ->
            val x = landmark.x() * imageWidth * scaleFactor + offsetX
            val y = landmark.y() * imageHeight * scaleFactor + offsetY
                canvas.drawCircle(x, y, 2f, landmarkPaint)
            }
        } else {
            // Full/original mode: connectors first, then landmarks (move landmarks after connectors)
            val connectorPaint = Paint(linePaint)
            connectorPaint.strokeWidth = LANDMARK_STROKE_WIDTH
            FaceLandmarker.FACE_LANDMARKS_CONNECTORS.filterNotNull().forEach { connector ->
                val startLandmark = faceLandmarks.getOrNull(connector.start())
                val endLandmark = faceLandmarks.getOrNull(connector.end())
                if (startLandmark != null && endLandmark != null) {
                    val startX = startLandmark.x() * imageWidth * scaleFactor + offsetX
                    val startY = startLandmark.y() * imageHeight * scaleFactor + offsetY
                    val endX = endLandmark.x() * imageWidth * scaleFactor + offsetX
                    val endY = endLandmark.y() * imageHeight * scaleFactor + offsetY
                    canvas.drawLine(startX, startY, endX, endY, connectorPaint)
                }
            }
            val landmarkPaint = Paint().apply {
                color = Color.argb(150, 200, 10, 200)
                strokeWidth = 1f * scaleFactor
                style = Paint.Style.FILL
                isAntiAlias = true
            }
            faceLandmarks.forEach { landmark ->
                val x = landmark.x() * imageWidth * scaleFactor + offsetX
                val y = landmark.y() * imageHeight * scaleFactor + offsetY
                canvas.drawCircle(x, y, 1f * scaleFactor, landmarkPaint)
            }

            // Draw head pose and gaze visualizations at the bottom of the face mesh (landmark 152)
            if (showHeadPoseVisualization && faceLandmarks.size > 152) {
                val facePose = facePoseAnalyzer.analyzeFacePose(faceLandmarks)
                val chin = faceLandmarks[152]
                val chinX = chin.x() * imageWidth * scaleFactor + offsetX
                val chinY = chin.y() * imageHeight * scaleFactor + offsetY
                drawHeadPoseVisualization(canvas, facePose, chinX, chinY)
                drawGazeVisualization(canvas, facePose, chinX, chinY)
            }

            // Draw head pose axes on the face
            if (showHeadPoseAxes) {
                drawHeadPoseOnFace(canvas, faceLandmarks, offsetX, offsetY)
            }

            // Draw gaze direction on the face
            if (showGaze) {
                drawGazeOnFace(canvas, faceLandmarks, offsetX, offsetY)
            }

            // Draw face pose info
            if (showFacePoseInfo) {
                val facePose = facePoseAnalyzer.analyzeFacePose(faceLandmarks)
                drawFacePoseInfo(canvas, facePose, offsetX, offsetY, faceIndex)
            }
        }
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

        // Calculate face orientation
        val facePose = facePoseAnalyzer.analyzeFacePose(faceLandmarks)

        // Define colors for each rotation axis
        val yawColor = Color.rgb(255, 100, 100)    // Red for yaw (左右)
        val pitchColor = Color.rgb(100, 255, 100)  // Green for pitch (上下)
        val rollColor = Color.rgb(0, 0, 160)       // Deep blue for roll (旋轉)

        // Draw rotation indicators with larger size
        drawRotationIndicators(canvas, noseX, noseY, facePose, yawColor, pitchColor, rollColor)

        // Draw 3D coordinate system with increased axis length
        val axisLength = 200f * scaleFactor  // Increased from 150f to 200f
        val points = calculate3DPoints(facePose.yaw, facePose.pitch, facePose.roll, axisLength)
        val projectedPoints = project3DPoints(points, noseX, noseY)
        draw3DAxes(canvas, projectedPoints, yawColor, pitchColor, rollColor)

        // Draw orientation text
        drawOrientationText(canvas, noseX, noseY, facePose, yawColor, pitchColor, rollColor)
    }

    private fun calculate3DPoints(
        yaw: Float,
        pitch: Float,
        roll: Float,
        axisLength: Float
    ): Array<FloatArray> {
        // Convert angles to radians
        val yawRad = Math.toRadians(yaw.toDouble())
        val pitchRad = Math.toRadians(pitch.toDouble())
        val rollRad = Math.toRadians(roll.toDouble())

        // Initial 3D points
        val points = arrayOf(
            floatArrayOf(0f, 0f, 0f),                    // Origin
            floatArrayOf(axisLength, 0f, 0f),            // X-axis
            floatArrayOf(0f, -axisLength, 0f),           // Y-axis
            floatArrayOf(0f, 0f, axisLength)             // Z-axis
        )

        // Apply rotations
        return points.map { point ->
            var x = point[0]
            var y = point[1]
            var z = point[2]

            // Roll rotation around Z
            val rollX = x * Math.cos(rollRad) - y * Math.sin(rollRad)
            val rollY = x * Math.sin(rollRad) + y * Math.cos(rollRad)
            x = rollX.toFloat()
            y = rollY.toFloat()

            // Pitch rotation around X
            val pitchY = y * Math.cos(pitchRad) - z * Math.sin(pitchRad)
            val pitchZ = y * Math.sin(pitchRad) + z * Math.cos(pitchRad)
            y = pitchY.toFloat()
            z = pitchZ.toFloat()

            // Yaw rotation around Y
            val yawX = x * Math.cos(yawRad) + z * Math.sin(yawRad)
            val yawZ = -x * Math.sin(yawRad) + z * Math.cos(yawRad)
            x = yawX.toFloat()
            z = yawZ.toFloat()

            floatArrayOf(x, y, z)
        }.toTypedArray()
    }

    private fun project3DPoints(
        points: Array<FloatArray>,
        centerX: Float,
        centerY: Float
    ): Array<FloatArray> {
        return points.map { point ->
            // Simple perspective projection
            val scale = 1f / (point[2] + 2f)  // Add offset to prevent division by zero
            floatArrayOf(
                point[0] * scale + centerX,
                point[1] * scale + centerY
            )
        }.toTypedArray()
    }

    private fun drawRotationIndicators(
        canvas: Canvas,
        centerX: Float,
        centerY: Float,
        facePose: FacePoseAnalyzer.FacePose,
        yawColor: Int,
        pitchColor: Int,
        rollColor: Int
    ) {
        // Draw rotation indicators with thinner lines and high opacity
        val indicatorPaint = Paint().apply {
            style = Paint.Style.STROKE
            strokeWidth = 1f * scaleFactor  // Thinner lines
            isAntiAlias = true
        }

        // Draw yaw rotation indicator (horizontal arc)
        indicatorPaint.color =
            Color.argb(240, Color.red(yawColor), Color.green(yawColor), Color.blue(yawColor))
        val yawRadius = 60f * scaleFactor
        canvas.drawArc(
            centerX - yawRadius,
            centerY - yawRadius,
            centerX + yawRadius,
            centerY + yawRadius,
            -90f,
            180f,
            false,
            indicatorPaint
        )

        // Draw pitch rotation indicator (vertical arc)
        indicatorPaint.color =
            Color.argb(240, Color.red(pitchColor), Color.green(pitchColor), Color.blue(pitchColor))
        val pitchRadius = 45f * scaleFactor
        canvas.drawArc(
            centerX - pitchRadius,
            centerY - pitchRadius,
            centerX + pitchRadius,
            centerY + pitchRadius,
            0f,
            180f,
            false,
            indicatorPaint
        )

        // Draw roll rotation indicator (circle with line)
        indicatorPaint.color =
            Color.argb(240, Color.red(rollColor), Color.green(rollColor), Color.blue(rollColor))
        val rollRadius = 30f * scaleFactor
        canvas.drawCircle(centerX, centerY, rollRadius, indicatorPaint)
        canvas.drawLine(
            centerX - rollRadius,
            centerY,
            centerX + rollRadius,
            centerY,
            indicatorPaint
        )

        // Draw rotation arrows with thinner lines
        val arrowPaint = Paint().apply {
            strokeWidth = 1f * scaleFactor  // Thinner lines
            isAntiAlias = true
        }

        // Draw yaw arrow
        drawRotationArrow(
            canvas,
            centerX,
            centerY,
            yawRadius,
            facePose.yaw,
            Color.argb(240, Color.red(yawColor), Color.green(yawColor), Color.blue(yawColor)),
            arrowPaint
        )

        // Draw pitch arrow
        drawRotationArrow(
            canvas,
            centerX,
            centerY,
            pitchRadius,
            facePose.pitch,
            Color.argb(240, Color.red(pitchColor), Color.green(pitchColor), Color.blue(pitchColor)),
            arrowPaint
        )

        // Draw roll arrow
        drawRotationArrow(
            canvas,
            centerX,
            centerY,
            rollRadius,
            facePose.roll,
            Color.argb(240, Color.red(rollColor), Color.green(rollColor), Color.blue(rollColor)),
            arrowPaint
        )
    }

    private fun drawRotationArrow(
        canvas: Canvas,
        centerX: Float,
        centerY: Float,
        radius: Float,
        angle: Float,
        color: Int,
        paint: Paint
    ) {
        paint.color = color
        paint.strokeWidth = 2f * scaleFactor  // Changed to 2f

        // Convert angle to radians
        val angleRad = Math.toRadians(angle.toDouble())

        // Calculate arrow end point
        val endX = centerX + radius * Math.cos(angleRad).toFloat()
        val endY = centerY + radius * Math.sin(angleRad).toFloat()

        // Calculate arrow head points
        val arrowLength = 15f * scaleFactor
        val arrowAngle = Math.PI / 6  // 30 degrees

        val arrowAngle1 = angleRad + arrowAngle
        val arrowAngle2 = angleRad - arrowAngle

        val arrowX1 = endX - arrowLength * Math.cos(arrowAngle1).toFloat()
        val arrowY1 = endY - arrowLength * Math.sin(arrowAngle1).toFloat()
        val arrowX2 = endX - arrowLength * Math.cos(arrowAngle2).toFloat()
        val arrowY2 = endY - arrowLength * Math.sin(arrowAngle2).toFloat()

        // Draw arrow line
        canvas.drawLine(centerX, centerY, endX, endY, paint)

        // Draw arrow head
        canvas.drawLine(endX, endY, arrowX1, arrowY1, paint)
        canvas.drawLine(endX, endY, arrowX2, arrowY2, paint)
    }

    private fun draw3DAxes(
        canvas: Canvas,
        points: Array<FloatArray>,
        yawColor: Int,
        pitchColor: Int,
        rollColor: Int
    ) {
        val axisPaint = Paint().apply {
            strokeWidth = 2f * scaleFactor
            isAntiAlias = true
        }

        // Draw axes with gradient effect
        val gradientPaint = Paint().apply {
            style = Paint.Style.FILL
            isAntiAlias = true
        }

        // Draw each axis with gradient and high opacity
        drawGradientAxis(
            canvas,
            points[0],
            points[1],
            Color.argb(240, Color.red(yawColor), Color.green(yawColor), Color.blue(yawColor)),
            axisPaint,
            gradientPaint
        )
        drawGradientAxis(
            canvas,
            points[0],
            points[2],
            Color.argb(240, Color.red(pitchColor), Color.green(pitchColor), Color.blue(pitchColor)),
            axisPaint,
            gradientPaint
        )
        drawGradientAxis(
            canvas,
            points[0],
            points[3],
            Color.argb(240, Color.red(rollColor), Color.green(rollColor), Color.blue(rollColor)),
            axisPaint,
            gradientPaint
        )

        // Draw axis endpoints with smaller radius (half size)
        val endpointRadius = 2f * scaleFactor  // Reduced from 16f
        val endpointPaint = Paint().apply {
            style = Paint.Style.STROKE
            isAntiAlias = true
        }

        // Draw endpoints for each axis with high opacity
        endpointPaint.color =
            Color.argb(240, Color.red(yawColor), Color.green(yawColor), Color.blue(yawColor))
        canvas.drawCircle(points[1][0], points[1][1], endpointRadius, endpointPaint)

        endpointPaint.color =
            Color.argb(240, Color.red(pitchColor), Color.green(pitchColor), Color.blue(pitchColor))
        canvas.drawCircle(points[2][0], points[2][1], endpointRadius, endpointPaint)

        endpointPaint.color =
            Color.argb(240, Color.red(rollColor), Color.green(rollColor), Color.blue(rollColor))
        canvas.drawCircle(points[3][0], points[3][1], endpointRadius, endpointPaint)

        // Draw axis labels with smaller text and transparency
        // val labelPaint = Paint().apply {
        //     textSize = 10f * scaleFactor  // Changed from 10f to 12f
        //     isAntiAlias = true
        //     alpha = 140  // Added transparency
        // }

        // // Draw labels for each axis
        // labelPaint.color = Color.argb(140, Color.red(yawColor), Color.green(yawColor), Color.blue(yawColor))
        // canvas.drawText("Yaw", points[1][0] + 10f * scaleFactor, points[1][1], labelPaint)

        // labelPaint.color = Color.argb(140, Color.red(pitchColor), Color.green(pitchColor), Color.blue(pitchColor))
        // canvas.drawText("Pitch", points[2][0], points[2][1] - 10f * scaleFactor, labelPaint)

        // labelPaint.color = Color.argb(140, Color.red(rollColor), Color.green(rollColor), Color.blue(rollColor))
        // canvas.drawText("Roll", points[3][0] + 10f * scaleFactor, points[3][1] + 10f * scaleFactor, labelPaint)
    }

    private fun drawGradientAxis(
        canvas: Canvas,
        start: FloatArray,
        end: FloatArray,
        color: Int,
        linePaint: Paint,
        gradientPaint: Paint
    ) {
        // Draw main axis line
        linePaint.color = color
        canvas.drawLine(start[0], start[1], end[0], end[1], linePaint)

        // Draw gradient effect
        gradientPaint.color =
            Color.argb(100, Color.red(color), Color.green(color), Color.blue(color))
        val radius = 5f * scaleFactor
        canvas.drawCircle(start[0], start[1], radius, gradientPaint)
        canvas.drawCircle(end[0], end[1], radius, gradientPaint)
    }


    private fun drawOrientationText(
        canvas: Canvas,
        centerX: Float,
        centerY: Float,
        facePose: FacePoseAnalyzer.FacePose,
        yawColor: Int,
        pitchColor: Int,
        rollColor: Int
    ) {
        // Draw text background with transparency
        val textBackgroundPaint = Paint().apply {
            color = Color.argb(100, 0, 0, 0)  // Changed alpha to 140
            style = Paint.Style.FILL
        }

        val textX = centerX + 60f * scaleFactor
        val textY = centerY - 60f * scaleFactor
        val textWidth = 80f * scaleFactor
        val textHeight = 50f * scaleFactor

        canvas.drawRect(
            textX,
            textY - textHeight,
            textX + textWidth,
            textY,
            textBackgroundPaint
        )

        // Draw text with smaller size and transparency
        textPaint.textSize = 15f * scaleFactor  //Changed from 10f to 12f
        textPaint.alpha = 100  // Added transparency

        // Yaw text
        textPaint.color =
            Color.argb(140, Color.red(yawColor), Color.green(yawColor), Color.blue(yawColor))
        canvas.drawText(
            "Yaw: ${String.format("%.1f", facePose.yaw)}°",
            textX + 10f * scaleFactor,
            textY - 40f * scaleFactor,
            textPaint
        )

        // Pitch text
        textPaint.color =
            Color.argb(140, Color.red(pitchColor), Color.green(pitchColor), Color.blue(pitchColor))
        canvas.drawText(
            "Pitch: ${String.format("%.1f", facePose.pitch)}°",
            textX + 10f * scaleFactor,
            textY - 20f * scaleFactor,
            textPaint
        )

        // Roll text
        textPaint.color =
            Color.argb(140, Color.red(rollColor), Color.green(rollColor), Color.blue(rollColor))
        canvas.drawText(
            "Roll: ${String.format("%.1f", facePose.roll)}°",
            textX + 10f * scaleFactor,
            textY,
            textPaint
        )
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
        // val eyePaint = Paint().apply {
        //     color = Color.BLUE
        //     style = Paint.Style.STROKE
        //     strokeWidth = 2f * scaleFactor
        //     isAntiAlias = true
        // }

        // val eyeRadius = 15f * scaleFactor
        // canvas.drawCircle(leftEyeX, leftEyeY, eyeRadius, eyePaint)
        // canvas.drawCircle(rightEyeX, rightEyeY, eyeRadius, eyePaint)

        // Draw gaze direction
        val gazePaint = Paint().apply {
            color = Color.RED
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
        offsetY: Float,
        faceIndex: Int
    ) {
        if (!showConnectors) return

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

        if (simpleDrawMode) {
            // Get nose tip position
            val noseTip = faceLandmarks[1]  // Nose tip landmark
            val noseX = noseTip.x() * imageWidth * scaleFactor + offsetX
            val noseY = noseTip.y() * imageHeight * scaleFactor + offsetY

            // Draw simple head pose axes at nose tip
            if (showHeadPoseAxes) {
                val facePose = facePoseAnalyzer.analyzeFacePose(faceLandmarks)
                drawSimpleHeadPoseAxes(canvas, facePose, noseX, noseY)
            }
            
            // Draw simple gaze lines from nose tip
            if (showGaze) {
                val facePose = facePoseAnalyzer.analyzeFacePose(faceLandmarks)
                drawSimpleGaze(canvas, facePose, offsetX, offsetY, gazeLength = 80f)
            }
            
            // Draw simple face info near nose tip
            if (showFaceInfo) {
                val facePose = facePoseAnalyzer.analyzeFacePose(faceLandmarks)
                drawSimpleFaceInfo(canvas, noseX, noseY, faceIndex, facePose)
            }
        }
    }

    private fun drawFacePoseInfo(
        canvas: Canvas,
        facePose: FacePoseAnalyzer.FacePose,
        offsetX: Float,
        offsetY: Float,
        faceIndex: Int
    ) {
        // Calculate base text position based on screen width and face index
        val columnWidth = width * 0.20f  // Each face gets 25% of screen width
        val baseTextX =
            width * 0.00f + (columnWidth * faceIndex)  // Start at 75% and increment by column width
        val baseTextY = 30f
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
        val gazeY = baseTextY + lineHeight * 9
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
            "Right Eye: (${String.format("%.2f", rightEye.x)}, ${
                String.format(
                    "%.2f",
                    rightEye.y
                )
            })",
            baseTextX,
            baseTextY + lineHeight * 7,
            textPaint
        )

        // Draw iris positions
        val leftIris = facePose.irisPositions.leftIris
        val rightIris = facePose.irisPositions.rightIris
        canvas.drawText(
            "Left Iris: (${String.format("%.2f", leftIris.x)}, ${
                String.format(
                    "%.2f",
                    leftIris.y
                )
            })",
            baseTextX,
            baseTextY + lineHeight * 8,
            textPaint
        )
        canvas.drawText(
            "Right Iris: (${String.format("%.2f", rightIris.x)}, ${
                String.format(
                    "%.2f",
                    rightIris.y
                )
            })",
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

    private fun drawHeadPoseVisualization(
        canvas: Canvas,
        facePose: FacePoseAnalyzer.FacePose,
        startX: Float,
        startY: Float
    ) {
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

    private fun drawGazeVisualization(
        canvas: Canvas,
        facePose: FacePoseAnalyzer.FacePose,
        startX: Float,
        startY: Float
    ) {
        val size = 200f
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
        canvas.drawCircle(leftEyeX, leftEyeY, 20f, eyePaint)

        // Draw right eye
        val rightEyeX = centerX + size * 0.3f
        val rightEyeY = centerY
        canvas.drawCircle(rightEyeX, rightEyeY, 20f, eyePaint)

        // Draw gaze direction
        val gazePaint = Paint().apply {
            color = Color.RED
            strokeWidth = 6f
            isAntiAlias = true
        }

        // Calculate gaze direction based on iris positions
        val leftIris = facePose.irisPositions.leftIris
        val rightIris = facePose.irisPositions.rightIris

        // Draw gaze lines
        val gazeLength = size * 0.8f
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

    private fun drawSimpleHeadPoseAxes(
        canvas: Canvas,
        facePose: FacePoseAnalyzer.FacePose,
        centerX: Float,
        centerY: Float
    ) {
        val axisLength = 60f * scaleFactor
        val yawColor = Color.RED
        val pitchColor = Color.GREEN
        val rollColor = Color.BLUE

        // Draw reference lines for zero degrees (base position)
        val referencePaint = Paint().apply {
            strokeWidth = 2f * scaleFactor
            style = Paint.Style.STROKE
            isAntiAlias = true
            pathEffect = DashPathEffect(floatArrayOf(10f * scaleFactor, 5f * scaleFactor), 0f)
        }

        // Draw yaw reference (horizontal line)
        referencePaint.color = Color.argb(150, Color.red(yawColor), Color.green(yawColor), Color.blue(yawColor))
        canvas.drawLine(
            centerX - axisLength,
            centerY,
            centerX + axisLength,
            centerY,
            referencePaint
        )

        // Draw pitch reference (vertical line)
        referencePaint.color = Color.argb(150, Color.red(pitchColor), Color.green(pitchColor), Color.blue(pitchColor))
        canvas.drawLine(
            centerX,
            centerY - axisLength,
            centerX,
            centerY + axisLength,
            referencePaint
        )

        // Draw roll reference (diagonal line)
        referencePaint.color = Color.argb(150, Color.red(rollColor), Color.green(rollColor), Color.blue(rollColor))
        canvas.drawLine(
            centerX - axisLength * 0.7f,
            centerY - axisLength * 0.7f,
            centerX + axisLength * 0.7f,
            centerY + axisLength * 0.7f,
            referencePaint
        )

        // Draw actual head pose axes
        val axisPaint = Paint().apply {
            strokeWidth = 3f * scaleFactor
            isAntiAlias = true
        }

        // 3D axes calculation
        val points = arrayOf(
            floatArrayOf(0f, 0f, 0f),
            floatArrayOf(axisLength, 0f, 0f),
            floatArrayOf(0f, -axisLength, 0f),
            floatArrayOf(0f, 0f, axisLength)
        )
        val yawRad = Math.toRadians(facePose.yaw.toDouble())
        val pitchRad = Math.toRadians(facePose.pitch.toDouble())
        val rollRad = Math.toRadians(facePose.roll.toDouble())
        val rotatedPoints = points.map { point ->
            var x = point[0]
            var y = point[1]
            var z = point[2]
            val rollX = x * Math.cos(rollRad) - y * Math.sin(rollRad)
            val rollY = x * Math.sin(rollRad) + y * Math.cos(rollRad)
            x = rollX.toFloat(); y = rollY.toFloat()
            val pitchY = y * Math.cos(pitchRad) - z * Math.sin(pitchRad)
            val pitchZ = y * Math.sin(pitchRad) + z * Math.cos(pitchRad)
            y = pitchY.toFloat(); z = pitchZ.toFloat()
            val yawX = x * Math.cos(yawRad) + z * Math.sin(yawRad)
            val yawZ = -x * Math.sin(yawRad) + z * Math.cos(yawRad)
            x = yawX.toFloat(); z = yawZ.toFloat()
            floatArrayOf(x, y, z)
        }
        val projected = rotatedPoints.map { p -> floatArrayOf(p[0] + centerX, p[1] + centerY) }

        // Draw axes with thicker lines
        axisPaint.color = yawColor
        canvas.drawLine(
            projected[0][0],
            projected[0][1],
            projected[1][0],
            projected[1][1],
            axisPaint
        )
        axisPaint.color = pitchColor
        canvas.drawLine(
            projected[0][0],
            projected[0][1],
            projected[2][0],
            projected[2][1],
            axisPaint
        )
        axisPaint.color = rollColor
        canvas.drawLine(
            projected[0][0],
            projected[0][1],
            projected[3][0],
            projected[3][1],
            axisPaint
        )

        // Draw endpoints with larger size
        val endpointPaint = Paint().apply {
            style = Paint.Style.FILL
            color = Color.BLACK
            isAntiAlias = true
        }
        canvas.drawCircle(
            projected[1][0],
            projected[1][1],
            5f * scaleFactor,  // Increased from 4f
            endpointPaint
        )
        canvas.drawCircle(
            projected[2][0],
            projected[2][1],
            5f * scaleFactor,  // Increased from 4f
            endpointPaint
        )
        canvas.drawCircle(
            projected[3][0],
            projected[3][1],
            5f * scaleFactor,  // Increased from 4f
            endpointPaint
        )
    }

    private fun drawSimpleGaze(
        canvas: Canvas,
        facePose: FacePoseAnalyzer.FacePose,
        offsetX: Float,
        offsetY: Float,
        gazeLength: Float = 40f
    ) {
        val leftEye = facePose.eyePositions.leftEye
        val rightEye = facePose.eyePositions.rightEye
        val leftIris = facePose.irisPositions.leftIris
        val rightIris = facePose.irisPositions.rightIris
        val gazePaint = Paint().apply {
            color = Color.RED
            strokeWidth = 2f * scaleFactor  // Add scaleFactor
            isAntiAlias = true
        }
        val leftEyeX = leftEye.x * imageWidth * scaleFactor + offsetX  // Add scaleFactor
        val leftEyeY = leftEye.y * imageHeight * scaleFactor + offsetY  // Add scaleFactor
        val rightEyeX = rightEye.x * imageWidth * scaleFactor + offsetX  // Add scaleFactor
        val rightEyeY = rightEye.y * imageHeight * scaleFactor + offsetY  // Add scaleFactor
        canvas.drawLine(
            leftEyeX,
            leftEyeY,
            leftEyeX + (leftIris.x - 0.5f) * gazeLength * scaleFactor,  // Add scaleFactor
            leftEyeY + (leftIris.y - 0.5f) * gazeLength * scaleFactor,  // Add scaleFactor
            gazePaint
        )
        canvas.drawLine(
            rightEyeX,
            rightEyeY,
            rightEyeX + (rightIris.x - 0.5f) * gazeLength * scaleFactor,  // Add scaleFactor
            rightEyeY + (rightIris.y - 0.5f) * gazeLength * scaleFactor,  // Add scaleFactor
            gazePaint
        )
    }

    private fun drawSimpleFaceInfo(
        canvas: Canvas,
        x: Float,
        y: Float,
        faceIndex: Int,
        facePose: FacePoseAnalyzer.FacePose
    ) {
        // Calculate box dimensions with scaleFactor
        val boxWidth = 100f * scaleFactor  // Reduced from 120f
        val boxHeight = 60f * scaleFactor  // Reduced from 60f
        val boxMargin = 10f * scaleFactor
        val textMargin = 15f * scaleFactor  // Reduced from 18f
        val textSpacing = 15f * scaleFactor  // Reduced from 18f

        // Draw background box
        val paint = Paint().apply {
            color = Color.WHITE
            style = Paint.Style.FILL
            alpha = 150
        }
        canvas.drawRect(
            x - boxWidth - boxMargin,
            y - boxHeight,
            x - boxMargin,
            y,
            paint
        )

        // Draw text with scaled size
        val textPaint = Paint().apply {
            textSize = 10f * scaleFactor  // Reduced from 15f
            isAntiAlias = true
        }

        // Draw face number
        textPaint.color = Color.BLACK
        canvas.drawText(
            "AFace ${faceIndex + 1}",
            x - boxWidth - boxMargin + textMargin,
            y - boxHeight + textSpacing,
            textPaint
        )

        // Draw yaw with red color
        textPaint.color = Color.RED
        canvas.drawText(
            "Yaw: ${String.format("%.1f", facePose.yaw)}",
            x - boxWidth - boxMargin + textMargin,
            y - boxHeight + textSpacing * 2,
            textPaint
        )

        // Draw pitch with green color
        textPaint.color = Color.GREEN
        canvas.drawText(
            "Pitch: ${String.format("%.1f", facePose.pitch)}",
            x - boxWidth - boxMargin + textMargin,
            y - boxHeight + textSpacing * 3,
            textPaint
        )

        // Draw roll with blue color
        textPaint.color = Color.BLUE
        canvas.drawText(
            "Roll: ${String.format("%.1f", facePose.roll)}",
            x - boxWidth - boxMargin + textMargin,
            y - boxHeight + textSpacing * 4,
            textPaint
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
                2.5f * min(width * 1f / imageWidth, height * 1f / imageHeight) // Double the scale
            }

            RunningMode.LIVE_STREAM -> {
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
