/*
 * Copyright 2023 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.google.mediapipe.examples.facelandmarker.fragment
import com.mitac.mediapipe.examples.facelandmarker.R

import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.SystemClock
import android.provider.MediaStore
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.AdapterView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.fragment.app.Fragment
import androidx.fragment.app.activityViewModels
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.viewpager2.widget.ViewPager2
import com.google.mediapipe.examples.facelandmarker.FaceLandmarkerHelper
import com.google.mediapipe.examples.facelandmarker.MainViewModel
import com.mitac.mediapipe.examples.facelandmarker.databinding.FragmentGalleryBinding
import com.google.mediapipe.tasks.vision.core.RunningMode
import java.util.Locale
import java.util.concurrent.Executors
import java.util.concurrent.ScheduledExecutorService
import java.util.concurrent.TimeUnit

class GalleryFragment : Fragment(), FaceLandmarkerHelper.LandmarkerListener {

    enum class MediaType {
        IMAGE,
        VIDEO,
        UNKNOWN
    }

    private var _fragmentGalleryBinding: FragmentGalleryBinding? = null
    private val fragmentGalleryBinding
        get() = _fragmentGalleryBinding!!
    private lateinit var faceLandmarkerHelper: FaceLandmarkerHelper
    private val viewModel: MainViewModel by activityViewModels()
    private val faceBlendshapesResultAdapter by lazy {
        FaceBlendshapesResultAdapter()
    }

    /** Blocking ML operations are performed using this executor */
    private lateinit var backgroundExecutor: ScheduledExecutorService

    private val getContent =
        registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri: Uri? ->
            // Handle the returned Uri
            uri?.let { mediaUri ->
                when (val mediaType = loadMediaType(mediaUri)) {
                    MediaType.IMAGE -> runDetectionOnImage(mediaUri)
                    MediaType.VIDEO -> runDetectionOnVideo(mediaUri)
                    MediaType.UNKNOWN -> {
                        updateDisplayView(mediaType)
                        Toast.makeText(
                            requireContext(),
                            "Unsupported data type.",
                            Toast.LENGTH_SHORT
                        ).show()
                    }
                }
            }
        }

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _fragmentGalleryBinding =
            FragmentGalleryBinding.inflate(inflater, container, false)

        return fragmentGalleryBinding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        fragmentGalleryBinding.fabGetContent.setOnClickListener {
            getContent.launch(arrayOf("image/*", "video/*"))
        }
        fragmentGalleryBinding.recyclerviewResults.setLayoutManager(LinearLayoutManager(requireContext()))
        fragmentGalleryBinding.recyclerviewResults.setAdapter(faceBlendshapesResultAdapter)

        initBottomSheetControls()

        val overlay = fragmentGalleryBinding.overlay

        // checkboxes
        val listener = android.widget.CompoundButton.OnCheckedChangeListener { buttonView, isChecked ->
            when(buttonView.id) {
                R.id.checkbox_simple_mode -> overlay.simpleDrawMode = isChecked
                R.id.checkbox_landmarks -> overlay.showFaceLandmarks = isChecked
                R.id.checkbox_connectors -> overlay.showConnectors = isChecked
                R.id.checkbox_head_pose_axes -> overlay.showHeadPoseAxes = isChecked
                R.id.checkbox_gaze -> overlay.showGaze = isChecked
                R.id.checkbox_face_pose_info -> overlay.showFacePoseInfo = isChecked
            }
        }

        fragmentGalleryBinding.checkboxSimpleMode.setOnCheckedChangeListener(listener)
        fragmentGalleryBinding.checkboxLandmarks.setOnCheckedChangeListener(listener)
        fragmentGalleryBinding.checkboxConnectors.setOnCheckedChangeListener(listener)
        fragmentGalleryBinding.checkboxHeadPoseAxes.setOnCheckedChangeListener(listener)
        fragmentGalleryBinding.checkboxGaze.setOnCheckedChangeListener(listener)
        fragmentGalleryBinding.checkboxFacePoseInfo.setOnCheckedChangeListener(listener)
    }

    override fun onPause() {
        fragmentGalleryBinding.overlay.clear()
        if (fragmentGalleryBinding.videoView.isPlaying) {
            fragmentGalleryBinding.videoView.stopPlayback()
        }
        fragmentGalleryBinding.videoView.setVisibility(View.GONE)
        fragmentGalleryBinding.imageResult.setVisibility(View.GONE)
        fragmentGalleryBinding.tvPlaceholder.setVisibility(View.VISIBLE)

        // Shutdown the background executor
        if (::backgroundExecutor.isInitialized) {
            backgroundExecutor.shutdown()
        }

        activity?.runOnUiThread {
            faceBlendshapesResultAdapter.updateResults(null)
            faceBlendshapesResultAdapter.notifyDataSetChanged()
        }
        super.onPause()
    }

    private fun initBottomSheetControls() {
        // init bottom sheet settings
        fragmentGalleryBinding.bottomSheetLayout.maxFacesValue.setText(viewModel.currentMaxFaces.toString())
        fragmentGalleryBinding.bottomSheetLayout.detectionThresholdValue.setText(String.format(Locale.US, "%.2f", viewModel.currentMinFaceDetectionConfidence))
        fragmentGalleryBinding.bottomSheetLayout.trackingThresholdValue.setText(String.format(Locale.US, "%.2f", viewModel.currentMinFaceTrackingConfidence))
        fragmentGalleryBinding.bottomSheetLayout.presenceThresholdValue.setText(String.format(Locale.US, "%.2f", viewModel.currentMinFacePresenceConfidence))

        // When clicked, lower detection score threshold floor
        fragmentGalleryBinding.bottomSheetLayout.detectionThresholdMinus.setOnClickListener {
            if (viewModel.currentMinFaceDetectionConfidence >= 0.2) {
                viewModel.setMinFaceDetectionConfidence(viewModel.currentMinFaceDetectionConfidence - 0.1f)
                updateControlsUi()
            }
        }

        // When clicked, raise detection score threshold floor
        fragmentGalleryBinding.bottomSheetLayout.detectionThresholdPlus.setOnClickListener {
            if (viewModel.currentMinFaceDetectionConfidence <= 0.8) {
                viewModel.setMinFaceDetectionConfidence(viewModel.currentMinFaceDetectionConfidence + 0.1f)
                updateControlsUi()
            }
        }

        // When clicked, lower face tracking score threshold floor
        fragmentGalleryBinding.bottomSheetLayout.trackingThresholdMinus.setOnClickListener {
            if (viewModel.currentMinFaceTrackingConfidence >= 0.2) {
                viewModel.setMinFaceTrackingConfidence(
                    viewModel.currentMinFaceTrackingConfidence - 0.1f
                )
                updateControlsUi()
            }
        }

        // When clicked, raise face tracking score threshold floor
        fragmentGalleryBinding.bottomSheetLayout.trackingThresholdPlus.setOnClickListener {
            if (viewModel.currentMinFaceTrackingConfidence <= 0.8) {
                viewModel.setMinFaceTrackingConfidence(
                    viewModel.currentMinFaceTrackingConfidence + 0.1f
                )
                updateControlsUi()
            }
        }

        // When clicked, lower face presence score threshold floor
        fragmentGalleryBinding.bottomSheetLayout.presenceThresholdMinus.setOnClickListener {
            if (viewModel.currentMinFacePresenceConfidence >= 0.2) {
                viewModel.setMinFacePresenceConfidence(
                    viewModel.currentMinFacePresenceConfidence - 0.1f
                )
                updateControlsUi()
            }
        }

        // When clicked, raise face presence score threshold floor
        fragmentGalleryBinding.bottomSheetLayout.presenceThresholdPlus.setOnClickListener {
            if (viewModel.currentMinFacePresenceConfidence <= 0.8) {
                viewModel.setMinFacePresenceConfidence(
                    viewModel.currentMinFacePresenceConfidence + 0.1f
                )
                updateControlsUi()
            }
        }

        // When clicked, reduce the number of objects that can be detected at a time
        fragmentGalleryBinding.bottomSheetLayout.maxFacesMinus.setOnClickListener {
            if (viewModel.currentMaxFaces > 1) {
                viewModel.setMaxFaces(viewModel.currentMaxFaces - 1)
                updateControlsUi()
            }
        }

        // When clicked, increase the number of objects that can be detected at a time
        fragmentGalleryBinding.bottomSheetLayout.maxFacesPlus.setOnClickListener {
            if (viewModel.currentMaxFaces < 2) {
                viewModel.setMaxFaces(viewModel.currentMaxFaces + 1)
                updateControlsUi()
            }
        }

        // When clicked, change the underlying hardware used for inference. Current options are CPU
        // GPU, and NNAPI
        fragmentGalleryBinding.bottomSheetLayout.spinnerDelegate.setSelection(
            viewModel.currentDelegate,
            false
        )
        fragmentGalleryBinding.bottomSheetLayout.spinnerDelegate.setOnItemSelectedListener(
            object : AdapterView.OnItemSelectedListener {
                override fun onItemSelected(
                    p0: AdapterView<*>?,
                    p1: View?,
                    p2: Int,
                    p3: Long
                ) {

                    viewModel.setDelegate(p2)
                    updateControlsUi()
                }

                override fun onNothingSelected(p0: AdapterView<*>?) {
                    /* no op */
                }
            })
    }

    // Update the values displayed in the bottom sheet. Reset detector.
    private fun updateControlsUi() {
        if (fragmentGalleryBinding.videoView.isPlaying) {
            fragmentGalleryBinding.videoView.stopPlayback()
        }
        fragmentGalleryBinding.videoView.setVisibility(View.GONE)
        fragmentGalleryBinding.imageResult.setVisibility(View.GONE)
        fragmentGalleryBinding.overlay.clear()
        fragmentGalleryBinding.bottomSheetLayout.maxFacesValue.setText(viewModel.currentMaxFaces.toString())
        fragmentGalleryBinding.bottomSheetLayout.detectionThresholdValue.setText(String.format(Locale.US, "%.2f", viewModel.currentMinFaceDetectionConfidence))
        fragmentGalleryBinding.bottomSheetLayout.trackingThresholdValue.setText(String.format(Locale.US, "%.2f", viewModel.currentMinFaceTrackingConfidence))
        fragmentGalleryBinding.bottomSheetLayout.presenceThresholdValue.setText(String.format(Locale.US, "%.2f", viewModel.currentMinFacePresenceConfidence))

        fragmentGalleryBinding.overlay.clear()
        fragmentGalleryBinding.tvPlaceholder.setVisibility(View.VISIBLE)
    }

    // Load and display the image.
    private fun runDetectionOnImage(uri: Uri) {
        setUiEnabled(false)
        backgroundExecutor = Executors.newSingleThreadScheduledExecutor()
        updateDisplayView(MediaType.IMAGE)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            val source = ImageDecoder.createSource(
                requireActivity().contentResolver,
                uri
            )
            ImageDecoder.decodeBitmap(source)
        } else {
            MediaStore.Images.Media.getBitmap(
                requireActivity().contentResolver,
                uri
            )
        }
            .copy(Bitmap.Config.ARGB_8888, true)
            ?.let { bitmap ->
                fragmentGalleryBinding.imageResult.setImageBitmap(bitmap)

                // Run face landmarker on the input image
                backgroundExecutor.execute {

                    faceLandmarkerHelper =
                        FaceLandmarkerHelper(
                            context = requireContext(),
                            runningMode = RunningMode.IMAGE,
                            minFaceDetectionConfidence = viewModel.currentMinFaceDetectionConfidence,
                            minFaceTrackingConfidence = viewModel.currentMinFaceTrackingConfidence,
                            minFacePresenceConfidence = viewModel.currentMinFacePresenceConfidence,
                            maxNumFaces = viewModel.currentMaxFaces,
                            currentDelegate = viewModel.currentDelegate
                        )

                    faceLandmarkerHelper.detectImage(bitmap)?.let { result ->
                        activity?.runOnUiThread {
                            if (fragmentGalleryBinding.recyclerviewResults.scrollState != ViewPager2.SCROLL_STATE_DRAGGING) {
                                faceBlendshapesResultAdapter.updateResults(result.result)
                                faceBlendshapesResultAdapter.notifyDataSetChanged()
                            }
                            fragmentGalleryBinding.overlay.setResults(
                                result.result,
                                bitmap.height,
                                bitmap.width,
                                RunningMode.IMAGE
                            )

                            setUiEnabled(true)
                            fragmentGalleryBinding.bottomSheetLayout.inferenceTimeVal.setText(String.format("%d ms", result.inferenceTime))
                        }
                    } ?: run { Log.e(TAG, "Error running face landmarker.") }

                    faceLandmarkerHelper.clearFaceLandmarker()
                }
            }
    }

    private fun runDetectionOnVideo(uri: Uri) {
        setUiEnabled(false)
        updateDisplayView(MediaType.VIDEO)

        fragmentGalleryBinding.videoView.setVideoURI(uri)
        // mute the audio
        fragmentGalleryBinding.videoView.setOnPreparedListener { mp: android.media.MediaPlayer -> mp.setVolume(0f, 0f) }
        // Add completion listener for looping
        fragmentGalleryBinding.videoView.setOnCompletionListener { mp: android.media.MediaPlayer ->
            // Restart video from beginning
            fragmentGalleryBinding.videoView.start()
        }
        fragmentGalleryBinding.videoView.requestFocus()

        backgroundExecutor = Executors.newSingleThreadScheduledExecutor()
        backgroundExecutor.execute {
            faceLandmarkerHelper =
                FaceLandmarkerHelper(
                    context = requireContext(),
                    runningMode = RunningMode.VIDEO,
                    minFaceDetectionConfidence = viewModel.currentMinFaceDetectionConfidence,
                    minFaceTrackingConfidence = viewModel.currentMinFaceTrackingConfidence,
                    minFacePresenceConfidence = viewModel.currentMinFacePresenceConfidence,
                    maxNumFaces = viewModel.currentMaxFaces,
                    currentDelegate = viewModel.currentDelegate
                )

            activity?.runOnUiThread {
                fragmentGalleryBinding.videoView.setVisibility(View.GONE)
                fragmentGalleryBinding.progressBar.setVisibility(View.VISIBLE)
            }

            faceLandmarkerHelper.detectVideoFile(uri, VIDEO_INTERVAL_MS)
                ?.let { resultBundle ->
                    activity?.runOnUiThread { displayVideoResult(resultBundle) }
                }
                ?: run { Log.e(TAG, "Error running face landmarker.") }

            faceLandmarkerHelper.clearFaceLandmarker()
        }
    }

    // Setup and display the video.
    private fun displayVideoResult(result: FaceLandmarkerHelper.VideoResultBundle) {
        fragmentGalleryBinding.videoView.setVisibility(View.VISIBLE)
        fragmentGalleryBinding.progressBar.setVisibility(View.GONE)

        fragmentGalleryBinding.videoView.start()
        var videoStartTimeMs = SystemClock.uptimeMillis()
        var lastResultIndex = -1

        backgroundExecutor.scheduleAtFixedRate(
            {
                activity?.runOnUiThread {
                    val videoElapsedTimeMs = SystemClock.uptimeMillis() - videoStartTimeMs
                    val resultIndex = videoElapsedTimeMs.div(VIDEO_INTERVAL_MS).toInt()

                    if (fragmentGalleryBinding.videoView.visibility == View.GONE) {
                        // The video playback has finished so we stop drawing bounding boxes
                        backgroundExecutor.shutdown()
                    } else {
                        // Handle video looping
                        if (resultIndex >= result.results.size) {
                            // Reset the start time when video loops
                            videoStartTimeMs = SystemClock.uptimeMillis()
                            lastResultIndex = -1
                        } else if (resultIndex != lastResultIndex) {
                            // Only update if we have a new frame
                            fragmentGalleryBinding.overlay.setResults(
                                result.results[resultIndex],
                                result.inputImageHeight,
                                result.inputImageWidth,
                                RunningMode.VIDEO
                            )

                            if (fragmentGalleryBinding.recyclerviewResults.scrollState != ViewPager2.SCROLL_STATE_DRAGGING) {
                                faceBlendshapesResultAdapter.updateResults(result.results[resultIndex])
                                faceBlendshapesResultAdapter.notifyDataSetChanged()
                            }

                            setUiEnabled(true)
                            fragmentGalleryBinding.bottomSheetLayout.inferenceTimeVal.setText(String.format("%d ms", result.inferenceTime))
                            
                            lastResultIndex = resultIndex
                        }
                    }
                }
            },
            0,
            VIDEO_INTERVAL_MS,
            TimeUnit.MILLISECONDS
        )
    }

    private fun updateDisplayView(mediaType: MediaType) {
        fragmentGalleryBinding.imageResult.setVisibility(if (mediaType == MediaType.IMAGE) View.VISIBLE else View.GONE)
        fragmentGalleryBinding.videoView.setVisibility(if (mediaType == MediaType.VIDEO) View.VISIBLE else View.GONE)
        fragmentGalleryBinding.tvPlaceholder.setVisibility(if (mediaType == MediaType.UNKNOWN) View.VISIBLE else View.GONE)
    }

    // Check the type of media that user selected.
    private fun loadMediaType(uri: Uri): MediaType {
        val mimeType = context?.contentResolver?.getType(uri)
        mimeType?.let {
            if (mimeType.startsWith("image")) return MediaType.IMAGE
            if (mimeType.startsWith("video")) return MediaType.VIDEO
        }

        return MediaType.UNKNOWN
    }

    private fun setUiEnabled(enabled: Boolean) {
        fragmentGalleryBinding.fabGetContent.setEnabled(enabled)
        fragmentGalleryBinding.bottomSheetLayout.detectionThresholdMinus.setEnabled(enabled)
        fragmentGalleryBinding.bottomSheetLayout.detectionThresholdPlus.setEnabled(enabled)
        fragmentGalleryBinding.bottomSheetLayout.trackingThresholdMinus.setEnabled(enabled)
        fragmentGalleryBinding.bottomSheetLayout.trackingThresholdPlus.setEnabled(enabled)
        fragmentGalleryBinding.bottomSheetLayout.presenceThresholdMinus.setEnabled(enabled)
        fragmentGalleryBinding.bottomSheetLayout.presenceThresholdPlus.setEnabled(enabled)
        fragmentGalleryBinding.bottomSheetLayout.maxFacesPlus.setEnabled(enabled)
        fragmentGalleryBinding.bottomSheetLayout.maxFacesMinus.setEnabled(enabled)
        fragmentGalleryBinding.bottomSheetLayout.spinnerDelegate.setEnabled(enabled)
    }

    private fun classifyingError() {
        activity?.runOnUiThread {
            fragmentGalleryBinding.progressBar.setVisibility(View.GONE)
            setUiEnabled(true)
            updateDisplayView(MediaType.UNKNOWN)
        }
    }

    override fun onError(error: String, errorCode: Int) {
        classifyingError()
        activity?.runOnUiThread {
            Toast.makeText(requireContext(), error, Toast.LENGTH_SHORT).show()
            if (errorCode == FaceLandmarkerHelper.GPU_ERROR) {
                fragmentGalleryBinding.bottomSheetLayout.spinnerDelegate.setSelection(
                    FaceLandmarkerHelper.DELEGATE_CPU,
                    false
                )
            }
        }
    }

    override fun onResults(resultBundle: FaceLandmarkerHelper.ResultBundle) {
        // no-op
    }

    companion object {
        private const val TAG = "GalleryFragment"

        // Value used to get frames at specific intervals for inference (e.g. every 300ms)
        private const val VIDEO_INTERVAL_MS = 300L
    }
}