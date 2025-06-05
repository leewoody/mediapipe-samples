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
import android.graphics.PointF
import android.util.AttributeSet
import android.view.View
import androidx.core.content.ContextCompat
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarkerResult
import kotlin.math.max
import kotlin.math.min
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.PI

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

                //Draw head pose cube on the face (centered at nose)
                val noseTip = faceLandmarks[1] // Nose tip landmark
                val noseX = noseTip.x() * imageWidth * scaleFactor + offsetX
                val noseY = noseTip.y() * imageHeight * scaleFactor + offsetY
                if (showFacePoseInfo) { // Use HeadPoseVisualization flag for the cube on face
                    drawHeadPoseCube(canvas, faceLandmarks, facePose, noseX, noseY)
                }

                // Draw face number near the face with matching color
                // val faceNumberX = noseTip.x() * imageWidth * scaleFactor + offsetX
                // val faceNumberY = noseTip.y() * imageHeight * scaleFactor + offsetY - 150f
                
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

                // Draw head pose and gaze visualizations at the bottom of the face mesh (landmark 152)
                if (faceLandmarks.size > 152) {
                    val facePose = facePoseAnalyzer.analyzeFacePose(faceLandmarks, imageWidth, imageHeight)
                    val chin = faceLandmarks[152]
                    val chinX = chin.x() * imageWidth * scaleFactor + offsetX
                    val chinY = chin.y() * imageHeight * scaleFactor + offsetY

                    // Get nose tip and forehead points for reference
                    val noseTip = faceLandmarks[1]  // Nose tip landmark
                    val forehead = faceLandmarks[10]  // Forehead landmark

                    val noseX = noseTip.x() * imageWidth * scaleFactor + offsetX
                    val noseY = noseTip.y() * imageHeight * scaleFactor + offsetY
                    
                    // if (showHeadPoseAxes) {
                    //     drawHeadPoseVisualization(canvas, facePose, chinX, chinY)
                    // }
                    if (showGaze) {
                         //drawHeadPoseVisualization(canvas, facePose, noseX, noseY)
                        drawSimpleHeadPoseAxes(canvas, facePose, noseX + 100f * scaleFactor, noseY)
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
                    drawFacePoseInfo(canvas, faceLandmarks, facePose, offsetX, offsetY, index)
                }

                // Draw individual pose indicators near the ears
                if (showHeadPoseAxes) {
                    val leftEar = faceLandmarks.getOrNull(FacePoseAnalyzer.LEFT_EAR) // Landmark for left ear (index 234)
                    val rightEar = faceLandmarks.getOrNull(FacePoseAnalyzer.RIGHT_EAR) // Landmark for right ear (index 454)

                    // Offset the indicators slightly from the ear landmark position
                    val indicatorHorizontalOffset = 30f * scaleFactor // Offset horizontally
                    val indicatorVerticalOffset = 0f
                    val indicatorSpacing = 50f * scaleFactor // Horizontal spacing between indicators

                    if (leftEar != null) {
                        val leftEarCoords = getLandmarkScreenCoords(leftEar, offsetX, offsetY)

                        // Position indicators horizontally to the left of the left ear
                        // Yaw indicator (leftmost position)
                        drawYawIndicatorNearEar(canvas, leftEarCoords.x - indicatorHorizontalOffset - indicatorSpacing * 2, leftEarCoords.y + indicatorVerticalOffset, facePose.yaw, scaleFactor)
                        // Pitch indicator (middle position)
                        drawPitchIndicatorNearEar(canvas, leftEarCoords.x - indicatorHorizontalOffset - indicatorSpacing, leftEarCoords.y + indicatorVerticalOffset, facePose.pitch, scaleFactor)
                        // Roll indicator (rightmost position)
                        drawRollIndicatorNearEar(canvas, leftEarCoords.x - indicatorHorizontalOffset, leftEarCoords.y + indicatorVerticalOffset, facePose.roll, scaleFactor)
                    }

                    if (rightEar != null) {
                        val rightEarCoords = getLandmarkScreenCoords(rightEar, offsetX, offsetY)
                        // Position indicators horizontally to the right of the right ear
                        // Yaw indicator (leftmost position)
                        drawYawIndicatorNearEar(canvas, rightEarCoords.x + indicatorHorizontalOffset, rightEarCoords.y + indicatorVerticalOffset, facePose.yaw, scaleFactor)
                        // Pitch indicator (middle position)
                        drawPitchIndicatorNearEar(canvas, rightEarCoords.x + indicatorHorizontalOffset + indicatorSpacing, rightEarCoords.y + indicatorVerticalOffset, facePose.pitch, scaleFactor)
                        // Roll indicator (rightmost position)
                        drawRollIndicatorNearEar(canvas, rightEarCoords.x + indicatorHorizontalOffset + indicatorSpacing * 2, rightEarCoords.y + indicatorVerticalOffset, facePose.roll, scaleFactor)
                    }
                }
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
        // drawXYZGrid(canvas, faceLandmarks, offsetX, offsetY)

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

                // Get nose tip and forehead points for reference
                val noseTip = faceLandmarks[1]  // Nose tip landmark
                val forehead = faceLandmarks[10]  // Forehead landmark

                val noseX = noseTip.x() * imageWidth * scaleFactor + offsetX
                val noseY = noseTip.y() * imageHeight * scaleFactor + offsetY
                
                // if (showHeadPoseAxes) {
                //     drawHeadPoseVisualization(canvas, facePose, chinX, chinY)
                // }
                if (showGaze) {
                     //drawHeadPoseVisualization(canvas, facePose, noseX, noseY)
                    drawSimpleHeadPoseAxes(canvas, facePose, noseX + 100f * scaleFactor, noseY)
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
                drawFacePoseInfo(canvas, faceLandmarks, facePose, offsetX, offsetY, faceIndex)
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
        val rollColor = Color.rgb(100, 100, 255)   // Light blue for roll (旋轉)

        // Draw rotation indicators with larger size
        drawRotationIndicators(canvas, noseX, noseY, facePose, yawColor, pitchColor, rollColor)

        // Draw 3D coordinate system with increased axis length
        // val axisLength = 200f * scaleFactor  // Increased from 150f to 200f
        // val points = calculate3DPoints(facePose.yaw, facePose.pitch, facePose.roll, axisLength)
        // val projectedPoints = project3DPoints(points, noseX, noseY)
        //draw3DAxes(canvas, projectedPoints, yawColor, pitchColor, rollColor)

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
            val scale = 1f / (point[2] / (500f * scaleFactor) + 1f)  // Add offset to prevent division by zero
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
                drawSimpleHeadPoseAxes(canvas, facePose, noseX + 100f * scaleFactor, noseY)
                // Draw improved head pose axes and 3D box
                // drawImprovedHeadPoseAxes(canvas, facePose, noseX + 100f * scaleFactor, noseY)
            }
            
            // Draw simple gaze lines from nose tip
            if (showGaze) {
                val facePose = facePoseAnalyzer.analyzeFacePose(faceLandmarks, imageWidth, imageHeight)
                drawSimpleGaze(canvas, facePose, offsetX, offsetY, gazeLength = 80f)
            }
            
            // Draw simple face info near nose tip
            // if (showFacePoseInfo) {
            //     val facePose = facePoseAnalyzer.analyzeFacePose(faceLandmarks, imageWidth, imageHeight)
            //     drawSimpleFaceInfo(canvas, noseX, noseY, faceIndex, facePose)
            // }
        } else {
            // In non-simple mode, also draw the 3D box visualization
            // if (showGaze) {
            //     val noseTip = faceLandmarks[1]  // Nose tip landmark
            //     val noseX = noseTip.x() * imageWidth * scaleFactor + offsetX
            //     val noseY = noseTip.y() * imageHeight * scaleFactor + offsetY
            //     val facePose = facePoseAnalyzer.analyzeFacePose(faceLandmarks, imageWidth, imageHeight)
            //     draw3DBoxVisualization(canvas, facePose, noseX, noseY)
            // }
        }
    }

    private fun drawFacePoseInfo(
        canvas: Canvas,
        faceLandmarks: List<NormalizedLandmark>,
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
            baseTextY + lineHeight * 14f, // Extend background to cover new visualization
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

        // Draw head pose visualization (original)
        if (showHeadPoseAxes) {
            val headPoseX = baseTextX
            val headPoseY = baseTextY + lineHeight * 2
            drawHeadPoseVisualization(canvas, facePose, headPoseX, headPoseY)
        }

        // Draw gaze visualization
        val gazeX = baseTextX
        val gazeY = baseTextY + lineHeight * 9
        drawGazeVisualization(canvas, facePose, gazeX, gazeY)

        // Draw head pose cube visualization centered in the pitch area
        // val cubeX = baseTextX + columnWidth * 0.5f // Position horizontally within the column
        // val cubeY = baseTextY + lineHeight * 4.5f // Position vertically centered around pitch
        // drawHeadPoseCube(canvas, facePose, cubeX, cubeY)

        // Draw 3D face mesh visualization with rotation
        // val meshVizX = baseTextX + columnWidth * 3f // Position horizontally within the column, shifted right
        // val meshVizY = baseTextY + lineHeight * 5f // Position vertically, adjusted to accommodate cube and text
        // draw3DFaceMeshVisualization(canvas, faceLandmarks, facePose, meshVizX, meshVizY)

        // Draw head pose angles
        canvas.drawText(
            "Yaw: ${String.format("%.1f", facePose.yaw)}°",
            baseTextX, baseTextY + lineHeight * 3, textPaint)
        canvas.drawText(
            "Pitch: ${String.format("%.1f", facePose.pitch)}°",
            baseTextX, baseTextY + lineHeight * 4, textPaint)
        canvas.drawText(
            "Roll: ${String.format("%.1f", facePose.roll)}°",
            baseTextX, baseTextY + lineHeight * 5, textPaint)

        // Draw improved head pose angles
        textPaint.textSize = 25f
        textPaint.color = Color.RED
        canvas.drawText(
            "Improved Yaw: ${String.format("%.1f", facePose.improvedYaw)}°",
            baseTextX, baseTextY + lineHeight * 6.5f, textPaint)
        textPaint.color = Color.GREEN
        canvas.drawText(
            "Improved Pitch: ${String.format("%.1f", facePose.improvedPitch)}°",
            baseTextX, baseTextY + lineHeight * 7.5f, textPaint)
        textPaint.color = Color.BLUE
        canvas.drawText(
            "Improved Roll: ${String.format("%.1f", facePose.improvedRoll)}°",
            baseTextX, baseTextY + lineHeight * 8.5f, textPaint)
        textPaint.textSize = 30f
        textPaint.color = Color.BLACK

        // Draw v2 head pose angles
        textPaint.textSize = 25f
        textPaint.color = Color.RED
        canvas.drawText(
            "v2 Yaw: ${String.format("%.1f", facePose.yaw)}°",
            baseTextX, baseTextY + lineHeight * 6.5f, textPaint)
        textPaint.color = Color.GREEN
        canvas.drawText(
            "v2 Pitch: ${String.format("%.1f", facePose.pitch)}°",
            baseTextX, baseTextY + lineHeight * 7.5f, textPaint)
        textPaint.color = Color.BLUE
        canvas.drawText(
            "v2 Roll: ${String.format("%.1f", facePose.v2Roll)}°",
            baseTextX, baseTextY + lineHeight * 8.5f, textPaint)
        textPaint.textSize = 30f
        textPaint.color = Color.BLACK

        // Draw eye positions
        val leftEye = facePose.eyePositions.leftEye
        val rightEye = facePose.eyePositions.rightEye
        canvas.drawText(
            "Left Eye: (${String.format("%.2f", leftEye.x)}, ${String.format("%.2f", leftEye.y)})",
            baseTextX, baseTextY + lineHeight * 10, textPaint)
        canvas.drawText(
            "Right Eye: (${String.format("%.2f", rightEye.x)}, ${String.format("%.2f", rightEye.y)})",
            baseTextX, baseTextY + lineHeight * 11, textPaint)

        // Draw iris positions
        val leftIris = facePose.irisPositions.leftIris
        val rightIris = facePose.irisPositions.rightIris
        canvas.drawText(
            "Left Iris: (${String.format("%.2f", leftIris.x)}, ${String.format("%.2f", leftIris.y)})",
            baseTextX, baseTextY + lineHeight * 12, textPaint)
        canvas.drawText(
            "Right Iris: (${String.format("%.2f", rightIris.x)}, ${String.format("%.2f", rightIris.y)})",
            baseTextX, baseTextY + lineHeight * 13, textPaint)

        // Draw screen distance if available
        facePose.screenDistance?.let {
            canvas.drawText(
                "Screen Distance: ${String.format("%.2f", it)}",
                baseTextX, baseTextY + lineHeight * 14, textPaint)
        }

        // Draw fatigue metrics
        facePose.fatigueMetrics?.let {
            // Set text color based on fatigue status
            textPaint.color = if (it.isFatigued) Color.RED else Color.GREEN

            canvas.drawText(
                "Eye Openness: ${String.format("%.2f", it.averageEyeOpenness)}",
                baseTextX, baseTextY + lineHeight * 15, textPaint)
            canvas.drawText(
                "Blink Count: ${it.blinkCount}",
                baseTextX, baseTextY + lineHeight * 16, textPaint)
            canvas.drawText(
                "Fatigue Score: ${String.format("%.2f", it.fatigueScore)}",
                baseTextX, baseTextY + lineHeight * 17, textPaint)
            canvas.drawText(
                "Status: ${if (it.isFatigued) "FATIGUED" else "ALERT"}",
                baseTextX, baseTextY + lineHeight * 18, textPaint)

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
        faceLandmarks: List<NormalizedLandmark>,
        facePose: FacePoseAnalyzer.FacePose,
        centerX: Float,
        centerY: Float
    ) {
        // Calculate bounding box of the landmarks in 3D
        var minX = Float.MAX_VALUE
        var maxX = Float.MIN_VALUE
        var minY = Float.MAX_VALUE
        var maxY = Float.MIN_VALUE
        var minZ = Float.MAX_VALUE
        var maxZ = Float.MIN_VALUE

        faceLandmarks.forEach { landmark ->
            minX = min(minX, landmark.x())
            maxX = max(maxX, landmark.x())
            minY = min(minY, landmark.y())
            maxY = max(maxY, landmark.y())
            minZ = min(minZ, landmark.z())
            maxZ = max(maxZ, landmark.z())
        }

        // Scale the bounding box dimensions
        val cubeWidth = (maxX - minX) * imageWidth * scaleFactor
        val cubeHeight = (maxY - minY) * imageHeight * scaleFactor
        val cubeDepth = (maxZ - minZ) * 500f * scaleFactor // Scale Z differently as it's relative

        val sphereRadius = 15f * scaleFactor
        val yawColor = Color.rgb(255, 100, 100)    // Light red for yaw
        val pitchColor = Color.rgb(100, 255, 100)  // Light green for pitch
        val rollColor = Color.rgb(100, 100, 255)   // Light blue for roll

        // Calculate gaze direction from iris positions
        val leftIris = facePose.irisPositions.leftIris
        val rightIris = facePose.irisPositions.rightIris
        
        // Calculate average iris position relative to center (0.5, 0.5)
        val avgIrisX = ((leftIris.x + rightIris.x) / 2 - 0.5f) * 2 // Scale to [-1, 1]
        val avgIrisY = ((leftIris.y + rightIris.y) / 2 - 0.5f) * 2 // Scale to [-1, 1]
        
        // Convert iris position to angles (full 360 degrees)
        val irisYaw = avgIrisX * 180f // Scale to [-180, 180] range
        val irisPitch = avgIrisY * 180f // Scale to [-180, 180] range
        
        // Use iris-based angles for cube rotation
        val yawRad = Math.toRadians(facePose.yaw.toDouble())
        val pitchRad = Math.toRadians(facePose.pitch.toDouble())
        val rollRad = Math.toRadians(facePose.roll.toDouble())

        // Position cube so the center is at the nose tip
        // val cubeShiftZ = -cubeSize / 2f // Shift the cube backward by half its depth

        // Define cube vertices with front face centered at (0,0,0) in local space
        val vertices = arrayOf(
            floatArrayOf(-cubeWidth / 2f, -cubeHeight / 2f, 0f),  // 0: left-bottom-front
            floatArrayOf(cubeWidth / 2f, -cubeHeight / 2f, 0f),   // 1: right-bottom-front
            floatArrayOf(cubeWidth / 2f, cubeHeight / 2f, 0f),    // 2: right-top-front
            floatArrayOf(-cubeWidth / 2f, cubeHeight / 2f, 0f),   // 3: left-top-front
            floatArrayOf(-cubeWidth / 2f, -cubeHeight / 2f, -cubeDepth), // 4: left-bottom-back
            floatArrayOf(cubeWidth / 2f, -cubeHeight / 2f, -cubeDepth),  // 5: right-bottom-back
            floatArrayOf(cubeWidth / 2f, cubeHeight / 2f, -cubeDepth),   // 6: right-top-back
            floatArrayOf(-cubeWidth / 2f, cubeHeight / 2f, -cubeDepth)   // 7: left-top-back
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
        }.toTypedArray()

        // Project vertices to 2D with scaling
        val projectedVertices = rotatedVertices.map { vertex ->
            // Adjust perspective scaling factor for a better visual fit
            val scale = 1f / (vertex[2] / (500f * scaleFactor) + 1f) // Adjusted back to 500f
            floatArrayOf(
                vertex[0] * scale + centerX,
                vertex[1] * scale + centerY
            )
        }.toTypedArray()

        // Define faces (vertex indices in order)
        val faces = arrayOf(
            intArrayOf(0, 1, 2, 3), // Front face (based on new vertex order)
            intArrayOf(4, 5, 6, 7), // Back face (based on new vertex order)
            intArrayOf(0, 4, 7, 3), // Left face (-X)
            intArrayOf(1, 5, 6, 2), // Right face (+X)
            intArrayOf(3, 2, 6, 7), // Top face (+Y)
            intArrayOf(0, 1, 5, 4)  // Bottom face (-Y)
        )

        // Define colors for each face
        val faceColors = arrayOf(
            Color.BLACK, // Front face
            Color.WHITE,  // Back face
            Color.WHITE,   // Left face
            Color.WHITE,   // Right face
            Color.WHITE, // Top face
            Color.WHITE  // Bottom face
        )

        // Draw faces (simple painter's algorithm - draw back faces first)
        val faceDepth = faces.map { face -> projectedVertices[face[0]][1] + projectedVertices[face[2]][1] }.withIndex().sortedByDescending { it.value }

        val facePaint = Paint().apply {
            style = Paint.Style.FILL
            isAntiAlias = true
        }

        faceDepth.forEach { (faceIndex, _) ->
            val face = faces[faceIndex]
            facePaint.color = faceColors[faceIndex]
            // Adjust alpha for the front face to make it cover landmarks more
            facePaint.alpha = if (faceIndex == 0) 200 else 100 // Front face (index 0) is less transparent

            val path = android.graphics.Path()
            path.moveTo(projectedVertices[face[0]][0], projectedVertices[face[0]][1])
            path.lineTo(projectedVertices[face[1]][0], projectedVertices[face[1]][1])
            path.lineTo(projectedVertices[face[2]][0], projectedVertices[face[2]][1])
            path.lineTo(projectedVertices[face[3]][0], projectedVertices[face[3]][1])
            path.close()
            canvas.drawPath(path, facePaint)
        }

        // Draw edges
        val linePaint = Paint().apply {
            strokeWidth = 2f * scaleFactor
            color = Color.argb(150, 255, 255, 255) // Semi-transparent white for edges
            style = Paint.Style.STROKE
            isAntiAlias = true
        }

        val edges = arrayOf(
            0 to 1, 1 to 2, 2 to 3, 3 to 0, // Front face
            4 to 5, 5 to 6, 6 to 7, 7 to 4, // Back face
            0 to 4, 1 to 5, 2 to 6, 3 to 7  // Connecting edges
        )

        edges.forEach { (startIdx, endIdx) ->
            val p1 = projectedVertices[startIdx]
            val p2 = projectedVertices[endIdx]
            canvas.drawLine(p1[0], p1[1], p2[0], p2[1], linePaint)
        }

        // Draw rotating sphere in center
        val spherePaint = Paint().apply {
            style = Paint.Style.FILL
            isAntiAlias = true
        }

        // Draw main sphere
        spherePaint.color = Color.argb(200, 255, 255, 255)
        // Position the sphere at the nose tip's 2D projection
        canvas.drawCircle(centerX, centerY, sphereRadius, spherePaint)

        // Draw rotation indicators on sphere
        val sphereIndicatorPaint = Paint().apply {
            strokeWidth = 2f * scaleFactor
            style = Paint.Style.STROKE
            isAntiAlias = true
        }

        // Draw yaw indicator (horizontal circle)
        sphereIndicatorPaint.color = yawColor
        // Position indicator at the nose tip's 2D projection
        canvas.drawCircle(centerX, centerY, sphereRadius * 0.8f, sphereIndicatorPaint)

        // Draw pitch indicator (vertical circle)
        sphereIndicatorPaint.color = pitchColor
        canvas.save()
        // Rotate around the nose tip's 2D projection
        canvas.rotate(90f, centerX, centerY)
        canvas.drawCircle(centerX, centerY, sphereRadius * 0.8f, sphereIndicatorPaint)
        canvas.restore()

        // Draw roll indicator (rotating line)
        sphereIndicatorPaint.color = rollColor
        canvas.save()
        // Rotate around the nose tip's 2D projection
        canvas.rotate(facePose.roll, centerX, centerY)
        canvas.drawLine(
            centerX - sphereRadius, // Start point X (relative to nose tip center)
            centerY, // Start point Y (relative to nose tip center)
            centerX + sphereRadius, // End point X (relative to nose tip center)
            centerY, // End point Y (relative to nose tip center)
            sphereIndicatorPaint
        )
        canvas.restore()

        // Draw angle values on sphere
        val textPaint = Paint().apply {
            textSize = 12f * scaleFactor
            isAntiAlias = true
            color = Color.BLACK
        }

        // Draw yaw value
        textPaint.color = yawColor
        canvas.drawText(
            "Y:${String.format("%.0f", facePose.yaw)}°",
            // Position text relative to the nose tip's 2D projection
            centerX - sphereRadius * 0.7f, // Adjusted position
            centerY - sphereRadius * 0.7f, // Adjusted position
            textPaint
        )

        // Draw pitch value
        textPaint.color = pitchColor
        canvas.drawText(
            "P:${String.format("%.0f", facePose.pitch)}°",
            // Position text relative to the nose tip's 2D projection
            centerX + sphereRadius * 0.7f, // Adjusted position
            centerY + sphereRadius * 0.7f, // Adjusted position
            textPaint
        )

        // Draw roll value
        textPaint.color = rollColor
        canvas.drawText(
            "R:${String.format("%.0f", facePose.v2Roll)}°", // Display v2 roll
            centerX,
            // Position text relative to the nose tip's 2D projection
            centerY + sphereRadius * 0.7f, // Adjusted position
            textPaint
        )

        // Draw individual rotation axes from the center sphere
        val axisLength = sphereRadius * 1.5f // Length of individual axes
        val axisPaint = Paint().apply {
            strokeWidth = 3f * scaleFactor
            isAntiAlias = true
        }

        // Yaw axis (rotate around Y, draw X axis initially)
        axisPaint.color = yawColor
        canvas.save()
        canvas.translate(centerX, centerY)
        canvas.rotate(facePose.yaw, 0f, 0f) // Rotate around center
        canvas.drawLine(0f, 0f, axisLength, 0f, axisPaint)
        canvas.restore()

        // Pitch axis (rotate around X, draw Y axis initially)
        axisPaint.color = pitchColor
        canvas.save()
        canvas.translate(centerX, centerY)
        canvas.rotate(facePose.pitch, 0f, 0f) // Rotate around center
        canvas.drawLine(0f, 0f, 0f, -axisLength, axisPaint) // Y is down in 2D
        canvas.restore()

        // Roll axis (rotate around Z, draw line in XY plane initially)
        axisPaint.color = rollColor
        canvas.save()
        canvas.translate(centerX, centerY)
        canvas.rotate(facePose.v2Roll, 0f, 0f)
        canvas.drawLine(-axisLength, 0f, axisLength, 0f, axisPaint) // Line in X direction
        canvas.restore()
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
        // draw3DBoxVisualization(canvas, facePose, centerX, centerY)
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

        val yawColor = Color.rgb(255, 100, 100)    // Red for yaw (左右)
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

    private fun draw3DFaceMeshVisualization(
        canvas: Canvas,
        faceLandmarks: List<NormalizedLandmark>,
        facePose: FacePoseAnalyzer.FacePose,
        centerX: Float,
        centerY: Float
    ) {
        // Find the approximate center of the face mesh in 2D for positioning
        // val mesh2DCenterX = faceLandmarks.map { it.x() }.average().toFloat() * imageWidth * scaleFactor + offsetX
        // val mesh2DCenterY = faceLandmarks.map { it.y() }.average().toFloat() * imageHeight * scaleFactor + offsetY

        // Use nose tip as the pivot for rotation and center of the visualization
        val noseTip = faceLandmarks.getOrNull(1) ?: return // Landmark 1 is nose tip

        val axisColorX = Color.RED
        val axisColorY = Color.GREEN
        val axisColorZ = Color.BLUE
        val gridColor = Color.GRAY

        val meshPaint = Paint().apply {
            color = Color.BLUE
            strokeWidth = 2f * scaleFactor
            style = Paint.Style.STROKE
            isAntiAlias = true
        }
        val pointPaint = Paint().apply {
            color = Color.BLUE
            style = Paint.Style.FILL
            isAntiAlias = true
        }
        val axisPaint = Paint().apply {
            strokeWidth = 3f * scaleFactor
            isAntiAlias = true
        }
        val gridLinePaint = Paint().apply {
            color = gridColor
            strokeWidth = 1f * scaleFactor
            style = Paint.Style.STROKE
            isAntiAlias = true
        }
         val textPaint = Paint().apply {
            color = Color.BLACK
            textSize = 15f * scaleFactor
            isAntiAlias = true
        }

        // Get improved head pose angles
        val yaw = facePose.improvedYaw
        val pitch = facePose.improvedPitch
        val roll = facePose.improvedRoll

        // Convert angles to radians
        val yawRad = Math.toRadians(yaw.toDouble())
        val pitchRad = Math.toRadians(pitch.toDouble())
        val rollRad = Math.toRadians(roll.toDouble())

        // Process landmarks: Translate, Rotate, Translate back, Project
        val projectedLandmarks = faceLandmarks.map { landmark ->
            // 1. Translate landmarks so nose tip is at origin (using nose tip 3D coords)
            var x = (landmark.x() - noseTip.x()) * imageWidth * scaleFactor
            var y = (landmark.y() - noseTip.y()) * imageHeight * scaleFactor
            var z = landmark.z() * 500f * scaleFactor // Use a scaled Z for perspective

            // 2. Apply rotations (Roll, then Pitch, then Yaw - standard aerospace sequence)

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

            // 3. Translate back (position the rotated origin at centerX, centerY)
            val finalX = x + centerX
            val finalY = y + centerY
            val finalZ = z // Z is used for perspective only, not translated back for drawing

            // 4. Project to 2D (simple perspective)
             val scale = 1f / (finalZ / (500f * scaleFactor) + 1f) // Adjust 500f for perspective strength
            floatArrayOf(
                finalX * scale + (1-scale)*centerX, // adjust for perspective origin
                finalY * scale + (1-scale)*centerY // adjust for perspective origin
            )
        }.toTypedArray()

        // Draw connectors
        FaceLandmarker.FACE_LANDMARKS_CONNECTORS.filterNotNull().forEach { connector ->
            val startLandmarkProj = projectedLandmarks.getOrNull(connector.start())
            val endLandmarkProj = projectedLandmarks.getOrNull(connector.end())
            if (startLandmarkProj != null && endLandmarkProj != null) {
                canvas.drawLine(startLandmarkProj[0], startLandmarkProj[1], endLandmarkProj[0], endLandmarkProj[1], meshPaint)
            }
        }

        // Draw landmarks
        projectedLandmarks.forEach { landmarkProj ->
            canvas.drawCircle(landmarkProj[0], landmarkProj[1], 4f * scaleFactor, pointPaint)
        }

        // Draw XYZ grid and axes (fixed relative to centerX, centerY)
        val gridSize = 50f * scaleFactor
        val gridExtent = 2 // Smaller grid for visualization

        // Draw grid lines parallel to XZ plane
        for (i in -gridExtent..gridExtent) {
             for (j in -gridExtent..gridExtent) {
                 val p1 = projectPointForGrid(floatArrayOf(i * gridSize, -gridExtent * gridSize, j * gridSize), centerX, centerY, scaleFactor)
                 val p2 = projectPointForGrid(floatArrayOf(i * gridSize, gridExtent * gridSize, j * gridSize), centerX, centerY, scaleFactor)
                 canvas.drawLine(p1[0], p1[1], p2[0], p2[1], gridLinePaint)

                 val p3 = projectPointForGrid(floatArrayOf(-gridExtent * gridSize, i * gridSize, j * gridSize), centerX, centerY, scaleFactor)
                 val p4 = projectPointForGrid(floatArrayOf(gridExtent * gridSize, i * gridSize, j * gridSize), centerX, centerY, scaleFactor)
                 canvas.drawLine(p3[0], p3[1], p4[0], p4[1], gridLinePaint)
             }
         }

        // Draw grid lines parallel to XY plane (for front/back faces of the grid volume)
        for (i in -gridExtent..gridExtent) {
            for (j in -gridExtent..gridExtent) {
                 val p1 = projectPointForGrid(floatArrayOf(i * gridSize, j * gridSize, -gridExtent * gridSize), centerX, centerY, scaleFactor)
                 val p2 = projectPointForGrid(floatArrayOf(i * gridSize, j * gridSize, gridExtent * gridSize), centerX, centerY, scaleFactor)
                 canvas.drawLine(p1[0], p1[1], p2[0], p2[1], gridLinePaint)
            }
        }

        // Draw axes
        val axisLength = (gridExtent + 0.5f) * gridSize
        val originProj = projectPointForGrid(floatArrayOf(0f, 0f, 0f), centerX, centerY, scaleFactor)
        val xAxisEndProj = projectPointForGrid(floatArrayOf(axisLength, 0f, 0f), centerX, centerY, scaleFactor)
        val yAxisEndProj = projectPointForGrid(floatArrayOf(0f, -axisLength, 0f), centerX, centerY, scaleFactor) // Y is typically down in 2D graphics
        val zAxisEndProj = projectPointForGrid(floatArrayOf(0f, 0f, axisLength), centerX, centerY, scaleFactor)

        axisPaint.color = axisColorX // X-axis (Red)
        canvas.drawLine(originProj[0], originProj[1], xAxisEndProj[0], xAxisEndProj[1], axisPaint)
        axisPaint.color = axisColorY // Y-axis (Green)
        canvas.drawLine(originProj[0], originProj[1], yAxisEndProj[0], yAxisEndProj[1], axisPaint)
        axisPaint.color = axisColorZ // Z-axis (Blue)
        canvas.drawLine(originProj[0], originProj[1], zAxisEndProj[0], zAxisEndProj[1], axisPaint)

        // Draw axis labels
        canvas.drawText("X", xAxisEndProj[0], xAxisEndProj[1], textPaint)
        canvas.drawText("Y", yAxisEndProj[0], yAxisEndProj[1], textPaint)
        canvas.drawText("Z", zAxisEndProj[0], zAxisEndProj[1], textPaint)
    }

    private fun projectPointForGrid(
        point: FloatArray,
        centerX: Float,
        centerY: Float,
        scaleFactor: Float
    ): FloatArray {
        // Simple perspective projection helper for grid points
        val scale = 1f / (point[2] / (300f * scaleFactor) + 1f) // Adjust 300f for perspective strength
        val projectedX = point[0] * scale + centerX
        val projectedY = point[1] * scale + centerY
        return floatArrayOf(projectedX, projectedY)
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

    // Helper function to get landmark screen coordinates
    private fun getLandmarkScreenCoords(landmark: NormalizedLandmark, offsetX: Float, offsetY: Float): PointF {
        val x = landmark.x() * imageWidth * scaleFactor + offsetX
        val y = landmark.y() * imageHeight * scaleFactor + offsetY
        return PointF(x, y)
    }

    // Draws a horizontal line rotated by yawAngle near the specified coordinates
    private fun drawYawIndicatorNearEar(
        canvas: Canvas,
        x: Float,
        y: Float,
        yawAngle: Float,
        scaleFactor: Float
    ) {
        val indicatorLength = 50f * scaleFactor // Length of the indicator line
        val yawColor = Color.rgb(255, 100, 100) // Red color

        val paint = Paint().apply {
            color = yawColor
            strokeWidth = 2f * scaleFactor // Adjust thickness to match rotation indicators
            isAntiAlias = true
            alpha = 240 // Match alpha from rotation indicators
        }

        canvas.save()
        canvas.translate(x, y) // Move canvas origin to the ear position
        //canvas.rotate(yawAngle) // Rotate based on yaw angle (This rotates the whole line, not just the dot)

        // Draw the horizontal line
        canvas.drawLine(-indicatorLength / 2f, 0f, indicatorLength / 2f, 0f, paint)

        // Draw the moving dot
        val dotPaint = Paint().apply {
            color = Color.BLACK // Black dot
            style = Paint.Style.FILL
            isAntiAlias = true
        }
        val dotRadius = 3f * scaleFactor

        // Calculate dot position based on yaw angle
        // Map yawAngle (e.g., -180 to 180) to a position along the indicator line (-indicatorLength/2 to +indicatorLength/2)
        val maxIndicatorYaw = 60f // Adjusting max indicator range for original angles
        val clampedYaw = yawAngle.coerceIn(-maxIndicatorYaw, maxIndicatorYaw)
        // Map clamped yaw angle [-maxIndicatorYaw, maxIndicatorYaw] to [-indicatorLength/2f, +indicatorLength/2f]
        // When yaw is -maxIndicatorYaw, dotX should be -indicatorLength/2f
        // When yaw is 0, dotX should be 0
        // When yaw is +maxIndicatorYaw, dotX should be +indicatorLength/2f
        val dotX = (clampedYaw / maxIndicatorYaw) * (indicatorLength / 2f) // This formula seems correct for linear mapping
        val dotY = 0f // The dot moves horizontally along the line

        // Draw the dot at the calculated position in the non-rotated canvas
        canvas.drawCircle(dotX, dotY, dotRadius, dotPaint)

        canvas.restore()
    }

    // Draws a vertical line rotated by pitchAngle near the specified coordinates
    private fun drawPitchIndicatorNearEar(
        canvas: Canvas,
        x: Float,
        y: Float,
        pitchAngle: Float,
        scaleFactor: Float
    ) {
        val indicatorLength = 30f * scaleFactor // Length of the indicator line
        val pitchColor = Color.rgb(100, 255, 100) // Green color

        val paint = Paint().apply {
            color = pitchColor
            strokeWidth = 2f * scaleFactor // Adjust thickness to match rotation indicators
            isAntiAlias = true
            alpha = 240 // Match alpha from rotation indicators
        }

        canvas.save()
        canvas.translate(x, y) // Move canvas origin to the ear position

        // Draw a vertical line centered at the origin (Y is down in 2D)
        canvas.drawLine(0f, -indicatorLength / 2f, 0f, indicatorLength / 2f, paint)

        // Draw the moving dot
        val dotPaint = Paint().apply {
            color = Color.BLACK // Black dot
            style = Paint.Style.FILL
            isAntiAlias = true
        }
        val dotRadius = 3f * scaleFactor

        // Calculate dot position based on pitch angle
        // Map pitchAngle (e.g., -180 to 180) to a position along the indicator line (-indicatorLength/2 to +indicatorLength/2)
        val maxIndicatorPitch = 60f // Adjusting max indicator range for original angles
        val clampedPitch = pitchAngle.coerceIn(-maxIndicatorPitch, maxIndicatorPitch)
        // Map clamped pitch angle [-maxIndicatorPitch, maxIndicatorPitch] to [-indicatorLength/2f, +indicatorLength/2f]
        // Note: Positive pitch means looking down, which corresponds to a larger Y value on the screen (Y is down).
        // So, a positive pitch angle should map to a positive Y position relative to the center (0,0).
        val dotX = 0f
        val dotY = (clampedPitch / maxIndicatorPitch) * (indicatorLength / 2f)

        // Draw the dot at the calculated position relative to the fixed vertical line (origin is at x, y)
        canvas.drawCircle(dotX, dotY, dotRadius, dotPaint)

        canvas.restore()
    }

    // Draws a semicircle rotated by rollAngle near the specified coordinates
    private fun drawRollIndicatorNearEar(
        canvas: Canvas,
        x: Float,
        y: Float,
        rollAngle: Float,
        scaleFactor: Float
    ) {
        val indicatorRadius = 15f * scaleFactor // Radius of the semicircle
        val rollColor = Color.rgb(100, 100, 255) // Blue color

        val paint = Paint().apply {
            color = rollColor
            strokeWidth = 2f * scaleFactor // Adjust thickness to match rotation indicators
            style = Paint.Style.STROKE
            isAntiAlias = true
            alpha = 240 // Match alpha from rotation indicators
        }

        canvas.save()
        canvas.translate(x, y) // Move canvas origin to the ear position
        // Rotate based on roll angle (semicircle opens downwards initially)
        canvas.rotate(rollAngle)
        // Draw a semicircle arc
        val rectF = android.graphics.RectF(
            -indicatorRadius,
            -indicatorRadius,
            indicatorRadius,
            indicatorRadius
        )
        // Arc from 180 to 360 degrees (bottom half of circle)
        canvas.drawArc(rectF, 180f, 180f, false, paint)

        // Draw the moving dot on the semicircle
        val dotPaint = Paint().apply {
            color = Color.BLACK // Black dot
            style = Paint.Style.FILL
            isAntiAlias = true
        }
        val dotRadius = 3f * scaleFactor

        // Calculate dot position on the semicircle based on roll angle
        // Map rollAngle (e.g., -180 to 180) to an angle on the semicircle (180 to 360 degrees)
        // We need to map the relevant range of roll angle to the 180-degree sweep of the arc.
        // Let's assume roll angle is most relevant in a range like -90 to +90 degrees.
        // Map -90 to 180 degrees on arc, 0 to 270 degrees on arc, +90 to 360 degrees on arc.
        // A simpler approach: Map the angle directly to a position along the arc length.
        // The total arc length is PI * radius.
        // Let's assume roll angle maps linearly to the arc from -90deg roll -> start of arc, +90deg roll -> end of arc.
        // Map rollAngle from [-90, 90] to [0, PI].
        val maxIndicatorRoll = 90f // Adjusting max indicator range for original angles (semicircle covers 180 deg)
        val clampedRoll = rollAngle.coerceIn(-maxIndicatorRoll, maxIndicatorRoll)
        // Normalize clamped roll to [0, 1]
        val normalizedRoll = (clampedRoll + maxIndicatorRoll) / (2 * maxIndicatorRoll)

        // Calculate angle on the semicircle arc in radians (0 to PI) for the dot position
        // The arc starts at 180 degrees (PI radians) and goes to 360 degrees (2*PI radians).
        // A 0 normalizedRoll should be at the start (180 deg), 1 at the end (360 deg).
        // Angle on arc = 180 + normalizedRoll * 180 (degrees) -> PI + normalizedRoll * PI (radians)
        // Map normalized roll [0, 1] to angle [180, 360] degrees on the arc.
        // 0 maps to 180 degrees (PI radians), 1 maps to 360 degrees (2*PI radians)
        val dotAngleRad = (180f + normalizedRoll * 180f) * (PI.toFloat() / 180f) // Radians

        // Calculate Cartesian coordinates (relative to origin at x, y) on the circle
        // In 2D graphics, angle 0 is usually right, 90 is down, 180 is left, 270 is up.
        // We need to calculate coordinates on a circle with radius `indicatorRadius`.
        val dotX = indicatorRadius * cos(dotAngleRad).toFloat()
        val dotY = indicatorRadius * sin(dotAngleRad).toFloat()

        // Draw the dot at the calculated position relative to the fixed semicircle (origin is at x, y)
        canvas.drawCircle(dotX, dotY, dotRadius, dotPaint)

        canvas.restore()
    }
}
