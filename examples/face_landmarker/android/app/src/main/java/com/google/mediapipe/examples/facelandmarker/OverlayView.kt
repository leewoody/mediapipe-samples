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
                val facePose = facePoseAnalyzer.analyzeFacePose(faceLandmarks, imageWidth, imageHeight) // Analyze pose once per face

                // Draw landmarks and connectors
                drawFaceConnectors(canvas, faceLandmarks, offsetX, offsetY, index)
                drawFaceLandmarks(canvas, faceLandmarks, offsetX, offsetY, index)

                // Draw head pose cube on the face (centered at nose)
                val noseTip = faceLandmarks[1] // Nose tip landmark
                val noseX = noseTip.x() * imageWidth * scaleFactor + offsetX
                val noseY = noseTip.y() * imageHeight * scaleFactor + offsetY
                if (showHeadPoseVisualization) { // Use HeadPoseVisualization flag for the cube on face
                     drawHeadPoseCube(canvas, facePose, noseX, noseY)
                }

                // Draw face number near the face with matching color
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

        // Draw XYZ grid background
        drawXYZGrid(canvas, faceLandmarks, offsetX, offsetY)

        if (showFaceLandmarks) {
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
        }

        if (!simpleDrawMode) {
            
            // Full/original mode: connectors first, then landmarks (move landmarks after connectors)
            // val connectorPaint = Paint(linePaint)
            // connectorPaint.strokeWidth = LANDMARK_STROKE_WIDTH
            // FaceLandmarker.FACE_LANDMARKS_CONNECTORS.filterNotNull().forEach { connector ->
            //     val startLandmark = faceLandmarks.getOrNull(connector.start())
            //     val endLandmark = faceLandmarks.getOrNull(connector.end())
            //     if (startLandmark != null && endLandmark != null) {
            //         val startX = startLandmark.x() * imageWidth * scaleFactor + offsetX
            //         val startY = startLandmark.y() * imageHeight * scaleFactor + offsetY
            //         val endX = endLandmark.x() * imageWidth * scaleFactor + offsetX
            //         val endY = endLandmark.y() * imageHeight * scaleFactor + offsetY
            //         canvas.drawLine(startX, startY, endX, endY, connectorPaint)
            //     }
            // }
            // val landmarkPaint = Paint().apply {
            //     color = Color.argb(150, 200, 10, 200)
            //     strokeWidth = 1f * scaleFactor
            //     style = Paint.Style.FILL
            //     isAntiAlias = true
            // }
            // faceLandmarks.forEach { landmark ->
            //     val x = landmark.x() * imageWidth * scaleFactor + offsetX
            //     val y = landmark.y() * imageHeight * scaleFactor + offsetY
            //     canvas.drawCircle(x, y, 1f * scaleFactor, landmarkPaint)
            // }

            // Draw head pose and gaze visualizations at the bottom of the face mesh (landmark 152)
            if (faceLandmarks.size > 152) {
                val facePose = facePoseAnalyzer.analyzeFacePose(faceLandmarks, imageWidth, imageHeight)
                val chin = faceLandmarks[152]
                val chinX = chin.x() * imageWidth * scaleFactor + offsetX
                val chinY = chin.y() * imageHeight * scaleFactor + offsetY
                if (showHeadPoseAxes) {
                    drawHeadPoseVisualization(canvas, facePose, chinX, chinY)
                }
                if (showGaze) {
                    drawGazeVisualization(canvas, facePose, chinX, chinY)
                }
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
                val facePose = facePoseAnalyzer.analyzeFacePose(faceLandmarks, imageWidth, imageHeight)
                drawFacePoseInfo(canvas, facePose, offsetX, offsetY, faceIndex)
            }
        }
    }

    private fun drawXYZGrid(
        canvas: Canvas,
        faceLandmarks: List<NormalizedLandmark>,
        offsetX: Float,
        offsetY: Float
    ) {
        // Find the approximate center of the face mesh for grid placement
        val centerX = faceLandmarks.map { it.x() }.average().toFloat() * imageWidth * scaleFactor + offsetX
        val centerY = faceLandmarks.map { it.y() }.average().toFloat() * imageHeight * scaleFactor + offsetY

        val gridColor = Color.GRAY
        val axisColor = Color.BLACK
        val gridLinePaint = Paint().apply {
            color = gridColor
            strokeWidth = 1f * scaleFactor
            style = Paint.Style.STROKE
            isAntiAlias = true
        }
        val axisPaint = Paint().apply {
            color = axisColor
            strokeWidth = 2f * scaleFactor
            isAntiAlias = true
        }
        val textPaint = Paint().apply {
            color = axisColor
            textSize = 15f * scaleFactor
            isAntiAlias = true
        }

        val gridSize = 50f * scaleFactor // Size of each grid cell
        val gridExtent = 3 // Number of grid cells in each direction from the center

        // Draw grid lines parallel to XZ plane (horizontal in 2D projection)
        for (i in -gridExtent..gridExtent) {
            val z = i * gridSize
            // Project points for this Z plane
            val p1 = projectPoint(floatArrayOf(-gridExtent * gridSize, 0f, z), centerX, centerY, scaleFactor)
            val p2 = projectPoint(floatArrayOf(gridExtent * gridSize, 0f, z), centerX, centerY, scaleFactor)
            canvas.drawLine(p1[0], p1[1], p2[0], p2[1], gridLinePaint)

            val p3 = projectPoint(floatArrayOf(i * gridSize, 0f, -gridExtent * gridSize), centerX, centerY, scaleFactor)
            val p4 = projectPoint(floatArrayOf(i * gridSize, 0f, gridExtent * gridSize), centerX, centerY, scaleFactor)
            canvas.drawLine(p3[0], p3[1], p4[0], p4[1], gridLinePaint)
        }

        // Draw grid lines parallel to YZ plane (vertical in 2D projection)
         for (i in -gridExtent..gridExtent) {
             val y = i * gridSize
             // Project points for this Y plane
             val p1 = projectPoint(floatArrayOf(0f, y, -gridExtent * gridSize), centerX, centerY, scaleFactor)
             val p2 = projectPoint(floatArrayOf(0f, y, gridExtent * gridSize), centerX, centerY, scaleFactor)
             canvas.drawLine(p1[0], p1[1], p2[0], p2[1], gridLinePaint)

             val p3 = projectPoint(floatArrayOf(0f, -gridExtent * gridSize, i * gridSize), centerX, centerY, scaleFactor)
             val p4 = projectPoint(floatArrayOf(0f, gridExtent * gridSize, i * gridSize), centerX, centerY, scaleFactor)
             // Need to adjust for depth sorting or draw order for vertical lines
             // For simplicity, drawing all lines for now
             canvas.drawLine(p3[0], p3[1], p4[0], p4[1], gridLinePaint)
         }

         // Draw axes
         val axisLength = (gridExtent + 1) * gridSize
         val origin = projectPoint(floatArrayOf(0f, 0f, 0f), centerX, centerY, scaleFactor)
         val xAxisEnd = projectPoint(floatArrayOf(axisLength, 0f, 0f), centerX, centerY, scaleFactor)
         val yAxisEnd = projectPoint(floatArrayOf(0f, -axisLength, 0f), centerX, centerY, scaleFactor) // Y is typically down in 2D graphics
         val zAxisEnd = projectPoint(floatArrayOf(0f, 0f, axisLength), centerX, centerY, scaleFactor)

         axisPaint.color = Color.RED // X-axis
         canvas.drawLine(origin[0], origin[1], xAxisEnd[0], xAxisEnd[1], axisPaint)
         axisPaint.color = Color.GREEN // Y-axis
         canvas.drawLine(origin[0], origin[1], yAxisEnd[0], yAxisEnd[1], axisPaint)
         axisPaint.color = Color.BLUE // Z-axis
         canvas.drawLine(origin[0], origin[1], zAxisEnd[0], zAxisEnd[1], axisPaint)

         // Draw axis labels
         canvas.drawText("X", xAxisEnd[0], xAxisEnd[1], textPaint)
         canvas.drawText("Y", yAxisEnd[0], yAxisEnd[1], textPaint)
         canvas.drawText("Z", zAxisEnd[0], zAxisEnd[1], textPaint)

         // Draw origin circle
         canvas.drawCircle(origin[0], origin[1], 3f * scaleFactor, axisPaint)
    }

    private fun projectPoint(
        point: FloatArray,
        centerX: Float,
        centerY: Float,
        scaleFactor: Float
    ): FloatArray {
        // Simple perspective projection helper
        // This is a simplified projection and may not perfectly match the example
        val scale = 1f / (point[2] / (1000f * scaleFactor) + 1f) // Adjust 1000f for perspective strength
        val projectedX = point[0] * scale + centerX
        val projectedY = point[1] * scale + centerY
        return floatArrayOf(projectedX, projectedY)
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
        val facePose = facePoseAnalyzer.analyzeFacePose(faceLandmarks, imageWidth, imageHeight)

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
        val facePose = facePoseAnalyzer.analyzeFacePose(faceLandmarks, imageWidth, imageHeight)

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
        if (showConnectors)
        {
         
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
        if (simpleDrawMode) {
            // Get nose tip position
            val noseTip = faceLandmarks[1]  // Nose tip landmark
            val noseX = noseTip.x() * imageWidth * scaleFactor + offsetX
            val noseY = noseTip.y() * imageHeight * scaleFactor + offsetY

            // Draw simple head pose axes at nose tip
            if (showHeadPoseAxes) {
                val facePose = facePoseAnalyzer.analyzeFacePose(faceLandmarks, imageWidth, imageHeight)
                drawSimpleHeadPoseAxes(canvas, facePose, noseX, noseY)
                // Draw improved head pose axes and 3D box
                drawImprovedHeadPoseAxes(canvas, facePose, noseX + 100f * scaleFactor, noseY)
            }
            
            // Draw simple gaze lines from nose tip
            if (showGaze) {
                val facePose = facePoseAnalyzer.analyzeFacePose(faceLandmarks, imageWidth, imageHeight)
                drawSimpleGaze(canvas, facePose, offsetX, offsetY, gazeLength = 80f)
            }
            
            // Draw simple face info near nose tip
            if (showFacePoseInfo) {
                val facePose = facePoseAnalyzer.analyzeFacePose(faceLandmarks, imageWidth, imageHeight)
                drawSimpleFaceInfo(canvas, noseX, noseY, faceIndex, facePose)
            }
        } else {
            // In non-simple mode, also draw the 3D box visualization
            if (showGaze) {
                val noseTip = faceLandmarks[1]  // Nose tip landmark
                val noseX = noseTip.x() * imageWidth * scaleFactor + offsetX
                val noseY = noseTip.y() * imageHeight * scaleFactor + offsetY
                val facePose = facePoseAnalyzer.analyzeFacePose(faceLandmarks, imageWidth, imageHeight)
                draw3DBoxVisualization(canvas, facePose, noseX, noseY)
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

        // Draw head pose cube visualization
        // val cubeX = baseTextX + 150f // Position the cube to the right of other info
        // val cubeY = baseTextY + lineHeight * 5 // Position it around the head pose angles
        // drawHeadPoseCube(canvas, facePose, cubeX, cubeY)

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

        // Draw improved head pose angles
        textPaint.textSize = 25f // Slightly smaller text for improved values
        textPaint.color = Color.RED // Red for improved yaw
        canvas.drawText(
            "Improved Yaw: ${String.format("%.1f", facePose.improvedYaw)}°",
            baseTextX,
            baseTextY + lineHeight * 6.5f, // Position below original yaw
            textPaint
        )
        textPaint.color = Color.GREEN // Green for improved pitch
        canvas.drawText(
            "Improved Pitch: ${String.format("%.1f", facePose.improvedPitch)}°",
            baseTextX,
            baseTextY + lineHeight * 7.5f, // Position below original pitch
            textPaint
        )
        textPaint.color = Color.BLUE // Blue for improved roll
        canvas.drawText(
            "Improved Roll: ${String.format("%.1f", facePose.improvedRoll)}°",
            baseTextX,
            baseTextY + lineHeight * 8.5f, // Position below original roll
            textPaint
        )
        textPaint.textSize = 30f // Reset text size
        textPaint.color = Color.BLACK // Reset text color

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

    private fun drawHeadPoseCube(
        canvas: Canvas,
        facePose: FacePoseAnalyzer.FacePose,
        centerX: Float,
        centerY: Float
    ) {
        val cubeSize = 50f * scaleFactor // Increased size of the cube
        val yawColor = Color.RED    // Red for yaw (X-axis)
        val pitchColor = Color.GREEN  // Green for pitch (Y-axis)
        val rollColor = Color.BLUE       // Blue for roll (Z-axis)

        // Convert angles to radians (using original pose for this visualization)
        val yawRad = Math.toRadians(facePose.yaw.toDouble())
        val pitchRad = Math.toRadians(facePose.pitch.toDouble())
        val rollRad = Math.toRadians(facePose.roll.toDouble())

        // Define cube vertices
        val vertices = arrayOf(
            floatArrayOf(-cubeSize, -cubeSize, -cubeSize),  // 0: left-bottom-back
            floatArrayOf(cubeSize, -cubeSize, -cubeSize),   // 1: right-bottom-back
            floatArrayOf(cubeSize, cubeSize, -cubeSize),    // 2: right-top-back
            floatArrayOf(-cubeSize, cubeSize, -cubeSize),   // 3: left-top-back
            floatArrayOf(-cubeSize, -cubeSize, cubeSize),   // 4: left-bottom-front
            floatArrayOf(cubeSize, -cubeSize, cubeSize),    // 5: right-bottom-front
            floatArrayOf(cubeSize, cubeSize, cubeSize),     // 6: right-top-front
            floatArrayOf(-cubeSize, cubeSize, cubeSize)     // 7: left-top-front
        )

        // Apply rotations
        val rotatedVertices = vertices.map { vertex ->
            var x = vertex[0]
            var y = vertex[1]
            var z = vertex[2]

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
        }

        // Project vertices to 2D with scaling
        val projectedVertices = rotatedVertices.map { vertex ->
             val scale = 1f / (vertex[2] / (500f * scaleFactor) + 1f) // Adjust perspective with scaleFactor
            floatArrayOf(
                vertex[0] * scale + centerX,
                vertex[1] * scale + centerY
            )
        }

        // Draw box edges
        val linePaint = Paint().apply {
            strokeWidth = 3f * scaleFactor // Increased line thickness with scaleFactor
            isAntiAlias = true
        }

        // Define edges for drawing
        val edges = arrayOf(
            0 to 1, 1 to 2, 2 to 3, 3 to 0, // Back face
            4 to 5, 5 to 6, 6 to 7, 7 to 4, // Front face
            0 to 4, 1 to 5, 2 to 6, 3 to 7  // Connecting edges
        )

        // Draw edges with colors representing axes orientation
        edges.forEach { (startIdx, endIdx) ->
            val p1 = projectedVertices[startIdx]
            val p2 = projectedVertices[endIdx]

            // Simple coloring based on edge direction (primarily for visual aid)
            linePaint.color = when {
                // Edges roughly aligned with X (connecting left and right vertices on the back/front faces)
                (startIdx == 0 && endIdx == 1) || (startIdx == 3 && endIdx == 2) ||
                (startIdx == 4 && endIdx == 5) || (startIdx == 7 && endIdx == 6) -> yawColor
                // Edges roughly aligned with Y (connecting top and bottom vertices on the back/front faces)
                (startIdx == 0 && endIdx == 3) || (startIdx == 1 && endIdx == 2) ||
                (startIdx == 4 && endIdx == 7) || (startIdx == 5 && endIdx == 6) -> pitchColor
                // Edges roughly aligned with Z (connecting back and front faces)
                (startIdx == 0 && endIdx == 4) || (startIdx == 1 && endIdx == 5) ||
                (startIdx == 2 && endIdx == 6) || (startIdx == 3 && endIdx == 7) -> rollColor
                else -> Color.BLACK // Default color
            }

            canvas.drawLine(p1[0], p1[1], p2[0], p2[1], linePaint)
        }
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
        val yawColor = Color.rgb(255, 100, 100)    // Light red for yaw
        val pitchColor = Color.rgb(100, 255, 100)  // Light green for pitch
        val rollColor = Color.rgb(100, 100, 255)   // Light blue for roll

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

        // Draw endpoints with smaller size
        val endpointPaint = Paint().apply {
            style = Paint.Style.FILL
            color = Color.BLACK
            isAntiAlias = true
        }
        canvas.drawCircle(
            projected[1][0],
            projected[1][1],
            1f * scaleFactor,  // Reduced from 5f to 3f
            endpointPaint
        )
        canvas.drawCircle(
            projected[2][0],
            projected[2][1],
            1f * scaleFactor,  // Reduced from 5f to 3f
            endpointPaint
        )
        canvas.drawCircle(
            projected[3][0],
            projected[3][1],
            1f * scaleFactor,  // Reduced from 5f to 3f
            endpointPaint
        )

        // Draw angle values
        val textPaint = Paint().apply {
            textSize = 12f * scaleFactor
            isAntiAlias = true
        }
        textPaint.color = yawColor
        canvas.drawText(
            "Y: ${String.format("%.1f", facePose.yaw)}°",
            projected[1][0] + 5f * scaleFactor,
            projected[1][1],
            textPaint
        )
        textPaint.color = pitchColor
        canvas.drawText(
            "P: ${String.format("%.1f", facePose.pitch)}°",
            projected[2][0],
            projected[2][1] - 5f * scaleFactor,
            textPaint
        )
        textPaint.color = rollColor
        canvas.drawText(
            "R: ${String.format("%.1f", facePose.roll)}°",
            projected[3][0] + 5f * scaleFactor,
            projected[3][1] + 5f * scaleFactor,
            textPaint
        )
    }

    private fun drawImprovedHeadPoseAxes(
        canvas: Canvas,
        facePose: FacePoseAnalyzer.FacePose,
        centerX: Float,
        centerY: Float
    ) {
        val axisLength = 60f * scaleFactor
        val yawColor = Color.rgb(247, 38, 38)    // Light red for yaw
        val pitchColor = Color.rgb(43, 244, 43)  // Light green for pitch
        val rollColor = Color.rgb(44, 44, 245)   // Light blue for roll

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

        // 3D axes calculation using improved angles
        val points = arrayOf(
            floatArrayOf(0f, 0f, 0f),
            floatArrayOf(axisLength, 0f, 0f),
            floatArrayOf(0f, -axisLength, 0f),
            floatArrayOf(0f, 0f, axisLength)
        )
        val yawRad = Math.toRadians(facePose.improvedYaw.toDouble())
        val pitchRad = Math.toRadians(facePose.improvedPitch.toDouble())
        val rollRad = Math.toRadians(facePose.improvedRoll.toDouble())
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
            1f * scaleFactor,
            endpointPaint
        )
        canvas.drawCircle(
            projected[2][0],
            projected[2][1],
            1f * scaleFactor,
            endpointPaint
        )
        canvas.drawCircle(
            projected[3][0],
            projected[3][1],
            1f * scaleFactor,
            endpointPaint
        )

        // Draw improved angle values
        val textPaint = Paint().apply {
            textSize = 12f * scaleFactor
            isAntiAlias = true
        }
        textPaint.color = yawColor
        canvas.drawText(
            "Y: ${String.format("%.1f", facePose.improvedYaw)}°",
            projected[1][0] + 5f * scaleFactor,
            projected[1][1],
            textPaint
        )
        textPaint.color = pitchColor
        canvas.drawText(
            "P: ${String.format("%.1f", facePose.improvedPitch)}°",
            projected[2][0],
            projected[2][1] - 5f * scaleFactor,
            textPaint
        )
        textPaint.color = rollColor
        canvas.drawText(
            "R: ${String.format("%.1f", facePose.improvedRoll)}°",
            projected[3][0] + 5f * scaleFactor,
            projected[3][1] + 5f * scaleFactor,
            textPaint
        )

        // Draw 3D box visualization at the same position as the face mesh
        draw3DBoxVisualization(canvas, facePose, centerX, centerY)
    }

    private fun draw3DBoxVisualization(
        canvas: Canvas,
        facePose: FacePoseAnalyzer.FacePose,
        centerX: Float,
        centerY: Float
    ) {
        // Calculate box size based on face mesh size
        val boxWidth = 80f * scaleFactor  // Reduced width
        val boxHeight = 110f * scaleFactor // Reduced height
        val boxDepth = 60f * scaleFactor   // Reduced depth

        val yawColor = Color.rgb(61, 1, 1)    // Dark red for yaw
        val pitchColor = Color.rgb(1, 64, 1)  // Dark green for pitch
        val rollColor = Color.rgb(2, 2, 53)   // Dark blue for roll

        // Convert angles to radians
        val yawRad = Math.toRadians(facePose.improvedYaw.toDouble())
        val pitchRad = Math.toRadians(facePose.improvedPitch.toDouble())
        val rollRad = Math.toRadians(facePose.improvedRoll.toDouble())

        // Define box vertices with adjusted dimensions
        val vertices = arrayOf(
            floatArrayOf(-boxWidth / 2, -boxHeight / 2, -boxDepth / 2),  // 0: left-bottom-back
            floatArrayOf(boxWidth / 2, -boxHeight / 2, -boxDepth / 2),   // 1: right-bottom-back
            floatArrayOf(boxWidth / 2, boxHeight / 2, -boxDepth / 2),    // 2: right-top-back
            floatArrayOf(-boxWidth / 2, boxHeight / 2, -boxDepth / 2),   // 3: left-top-back
            floatArrayOf(-boxWidth / 2, -boxHeight / 2, boxDepth / 2),   // 4: left-bottom-front
            floatArrayOf(boxWidth / 2, -boxHeight / 2, boxDepth / 2),    // 5: right-bottom-front
            floatArrayOf(boxWidth / 2, boxHeight / 2, boxDepth / 2),     // 6: right-top-front
            floatArrayOf(-boxWidth / 2, boxHeight / 2, boxDepth / 2)     // 7: left-top-front
        )

        // Apply rotations
        val rotatedVertices = vertices.map { vertex ->
            var x = vertex[0]
            var y = vertex[1]
            var z = vertex[2]

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
        }

        // Project vertices to 2D with adjusted perspective
        val projectedVertices = rotatedVertices.map { vertex ->
            val scale = 250f / (vertex[2] + 100f * scaleFactor)  // Adjusted scale and offset
            floatArrayOf(
                vertex[0] * scale + centerX,
                vertex[1] * scale + centerY
            )
        }

        // Draw box edges with increased thickness
        val linePaint = Paint().apply {
            strokeWidth = 2f * scaleFactor  // Slightly reduced thickness
            isAntiAlias = true
        }

        // Draw edges - need to handle visibility based on depth (simple approach)
        // Drawing all edges for now, a proper 3D rendering would require depth sorting

        // Back face (more likely to be occluded, draw first)
        linePaint.color = yawColor
        drawBoxEdge(canvas, projectedVertices[0], projectedVertices[1], linePaint)
        drawBoxEdge(canvas, projectedVertices[1], projectedVertices[2], linePaint)
        drawBoxEdge(canvas, projectedVertices[2], projectedVertices[3], linePaint)
        drawBoxEdge(canvas, projectedVertices[3], projectedVertices[0], linePaint)

        // Connecting edges (draw next)
        linePaint.color = rollColor
        drawBoxEdge(canvas, projectedVertices[0], projectedVertices[4], linePaint)
        drawBoxEdge(canvas, projectedVertices[1], projectedVertices[5], linePaint)
        drawBoxEdge(canvas, projectedVertices[2], projectedVertices[6], linePaint)
        drawBoxEdge(canvas, projectedVertices[3], projectedVertices[7], linePaint)

        // Front face (less likely to be occluded, draw last)
        linePaint.color = pitchColor
        drawBoxEdge(canvas, projectedVertices[4], projectedVertices[5], linePaint)
        drawBoxEdge(canvas, projectedVertices[5], projectedVertices[6], linePaint)
        drawBoxEdge(canvas, projectedVertices[6], projectedVertices[7], linePaint)
        drawBoxEdge(canvas, projectedVertices[7], projectedVertices[4], linePaint)

        // Draw angle values with larger text
        val textPaint = Paint().apply {
            textSize = 15f * scaleFactor  // Slightly reduced text size
            isAntiAlias = true
        }

        // Draw yaw value
        textPaint.color = yawColor
        canvas.drawText(
            "Yaw: ${String.format("%.1f", facePose.improvedYaw)}°",
            centerX - boxWidth / 2,
            centerY - boxHeight / 2 - 15f * scaleFactor, // Adjusted position
            textPaint
        )

        // Draw pitch value
        textPaint.color = pitchColor
        canvas.drawText(
            "Pitch: ${String.format("%.1f", facePose.improvedPitch)}°",
            centerX - boxWidth / 2,
            centerY - boxHeight / 2 - 30f * scaleFactor, // Adjusted position
            textPaint
        )

        // Draw roll value
        textPaint.color = rollColor
        canvas.drawText(
            "Roll: ${String.format("%.1f", facePose.improvedRoll)}°",
            centerX - boxWidth / 2,
            centerY - boxHeight / 2 - 45f * scaleFactor, // Adjusted position
            textPaint
        )
    }

    private fun drawBoxEdge(
        canvas: Canvas,
        start: FloatArray,
        end: FloatArray,
        paint: Paint
    ) {
        canvas.drawLine(start[0], start[1], end[0], end[1], paint)
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
                2.0f * min(width * 1f / imageWidth, height * 1f / imageHeight) // Double the scale
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
