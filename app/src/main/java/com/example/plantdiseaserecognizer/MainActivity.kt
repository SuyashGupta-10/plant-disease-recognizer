package com.example.plantdiseaserecognizer

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.view.View
import android.graphics.BitmapFactory
import android.os.Bundle
import android.widget.Toast
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.example.plantdiseaserecognizer.databinding.ActivityMainBinding
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import kotlin.math.exp

class MainActivity : AppCompatActivity() {
    private lateinit var binding: ActivityMainBinding
    private lateinit var env: OrtEnvironment
    private lateinit var session: OrtSession
    private var modelReady = false

    private lateinit var requestCameraPermission: ActivityResultLauncher<String>

    // Camera preview launcher
    private val takePicture = registerForActivityResult(
        ActivityResultContracts.TakePicturePreview()
    ) { bmp ->
        if (bmp == null) {
            Toast.makeText(this, "Failed to capture image", Toast.LENGTH_SHORT).show()
        } else {
            handleBitmap(bmp)
        }
    }

    // Gallery picker launcher
    private val chooseImage = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri ->
        if (uri == null) return@registerForActivityResult
        try {
            contentResolver.openInputStream(uri)?.use { stream ->
                BitmapFactory.decodeStream(stream)
            }?.let { bmp ->
                handleBitmap(bmp)
            } ?: Toast.makeText(this, "Could not decode image", Toast.LENGTH_SHORT).show()
        } catch (e: Exception) {
            e.printStackTrace()
            Toast.makeText(this, "Error loading image", Toast.LENGTH_SHORT).show()
        }
    }

    // Class labels in the exact same order
    private val labels = arrayOf(
        "Bell Pepper- Bacterial Spot",
        "Bell Pepper- Healthy",
        "Potato- Early Blight",
        "Potato- Late Blight",
        "Potato- Healthy",
        "Tomato- Bacterial Spot",
        "Tomato- Early Blight",
        "Tomato- Late Blight",
        "Tomato- Leaf Mold",
        "Tomato- Septoria Leaf Spot",
        "Tomato- Spider Mites (Two-spotted spider_mite)",
        "Tomato- Target Spot",
        "Tomato- Tomato Yellow Leaf Curl Virus (TYLCV)",
        "Tomato- Tomato Mosaic Virus",
        "Tomato- Healthy"
    )

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // Disable buttons until the model is loaded
        binding.btnCamera.isEnabled = false
        binding.btnGallery.isEnabled = false

        // 1) Register runtime permission launcher
        requestCameraPermission = registerForActivityResult(
            ActivityResultContracts.RequestPermission()
        ) { granted ->
            if (granted) {
                takePicture.launch(null)
            } else {
                Toast.makeText(this, "Camera permission needed", Toast.LENGTH_SHORT).show()
            }
        }

        // 2) Load ONNX model off the UI thread
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                val envLocal     = OrtEnvironment.getEnvironment()
                val modelBytes   = assets.open("Plant_disease_MobileNetV3.onnx").readBytes()
                val sessionLocal = envLocal.createSession(modelBytes)

                withContext(Dispatchers.Main) {
                    env        = envLocal
                    session    = sessionLocal
                    modelReady = true

                    binding.btnCamera.isEnabled = true
                    binding.btnGallery.isEnabled = true
                    Toast.makeText(this@MainActivity, "Model loaded!", Toast.LENGTH_SHORT).show()
                }
            } catch (ex: Exception) {
                ex.printStackTrace()
                withContext(Dispatchers.Main) {
                    val msg = ex.localizedMessage ?: "Unknown error"
                    Toast.makeText(this@MainActivity, "Failed to load model: $msg", Toast.LENGTH_LONG).show()
                }
            }
        }

        // 3) Camera button click
        binding.btnCamera.setOnClickListener {
            if (!modelReady) {
                Toast.makeText(this, "Model still loading…", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                == PackageManager.PERMISSION_GRANTED
            ) {
                takePicture.launch(null)
            } else {
                requestCameraPermission.launch(Manifest.permission.CAMERA)
            }
        }

        // 4) Gallery button click
        binding.btnGallery.setOnClickListener {
            if (!modelReady) {
                Toast.makeText(this, "Model still loading…", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }
            chooseImage.launch("image/*")
        }
    }

    private fun handleBitmap(bitmap: Bitmap) {
        binding.imgPreview.setImageBitmap(bitmap)

        lifecycleScope.launch {
            // 1) inference off the UI thread
            val resultText = withContext(Dispatchers.IO) { runInference(bitmap) }

            // 2) now on Main, show the card and split it
            withContext(Dispatchers.Main) {
                binding.cardResult.visibility = View.VISIBLE

                // e.g. resultText == "Predicted: Tomato_Late_blight (conf 86.76%)"
                val core = resultText
                    .removePrefix("Predicted: ")
                    .trim()                                           // "Tomato_Late_blight (conf 86.76%)"

                // 1) Label is everything before the first " ("
                val label = core.substringBefore(" (")

                // 2) Regex to pull the number inside parentheses, works for "conf" or "Confidence"
                val pctRegex = """\((?:conf|Confidence):?\s*([\d.]+)%""".toRegex(RegexOption.IGNORE_CASE)
                val pct = pctRegex.find(core)?.groupValues?.get(1) ?: "0"

                // 3) Build the grey confidence line
                val confidenceText = "$pct% Precision"

                // 4) Update your two TextViews
                binding.tvResultLabel.text      = label
                binding.tvResultConfidence.text = confidenceText
            }
        }
    }


    private fun runInference(bitmap: Bitmap): String {
        return try {
            // 1) Resize to 128×128
            val inputSize = 128
            val resized = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)

            // 2) Read pixels
            val pixelCount = inputSize * inputSize
            val pixels = IntArray(pixelCount)
            resized.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)

            // 3) Allocate a direct ByteBuffer and get a FloatBuffer view
            val byteBuffer = ByteBuffer
                .allocateDirect(4 * 1 * 3 * pixelCount)    // 4 bytes per float
                .order(ByteOrder.nativeOrder())
            val floatBuffer: FloatBuffer = byteBuffer.asFloatBuffer()

            // 4) Channel-first packing + Normalize(mean=0.5, std=0.5)
            val MEAN = floatArrayOf(0.5f, 0.5f, 0.5f)
            val STD  = floatArrayOf(0.5f, 0.5f, 0.5f)
            for (c in 0 until 3) {
                for (y in 0 until inputSize) {
                    for (x in 0 until inputSize) {
                        val px = pixels[y * inputSize + x]
                        val v = when (c) {
                            0 -> ((px shr 16 and 0xFF) / 255f - MEAN[0]) / STD[0]
                            1 -> ((px shr  8 and 0xFF) / 255f - MEAN[1]) / STD[1]
                            else -> ((px        and 0xFF) / 255f - MEAN[2]) / STD[2]
                        }
                        floatBuffer.put(v)
                    }
                }
            }
            floatBuffer.rewind()

            // 5) Create tensor from the FloatBuffer
            val shape = longArrayOf(1, 3, inputSize.toLong(), inputSize.toLong())
            OnnxTensor.createTensor(env, floatBuffer, shape).use { tensor ->
                session.run(mapOf(session.inputNames.first() to tensor)).use { results ->
                    val outOnnx     = results[0] as OnnxTensor
                    @Suppress("UNCHECKED_CAST")
                    val outputArray = outOnnx.getValue() as Array<FloatArray>
                    val scores      = outputArray[0]

                    // 6) Softmax + pick highest probability
                    val probs = softmax(scores)
                    val idx   = probs.indices.maxByOrNull { probs[it] } ?: -1
                    val label = labels.getOrNull(idx) ?: "Unknown"
                    val conf  = "%.2f".format(probs[idx] * 100)

                    return "Predicted: $label (conf $conf%)"
                }
            }
        } catch (e: Exception) {
            e.printStackTrace()
            "Error: ${e.localizedMessage}"
        }
    }

    private fun softmax(logits: FloatArray): FloatArray {
        val maxLogit = logits.maxOrNull() ?: 0f
        val exps     = logits.map { exp((it - maxLogit).toDouble()) }
        val sum      = exps.sum()
        return exps.map { (it / sum).toFloat() }.toFloatArray()
    }
}
