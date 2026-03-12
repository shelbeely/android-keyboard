package org.futo.inputmethod.latin.xlm

import android.content.Context
import android.os.Build
import com.google.ai.edge.aicore.GenerativeModel
import com.google.ai.edge.aicore.generationConfig
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * Represents the different tone styles available for tone transformation via Gemini Nano.
 */
enum class ToneStyle {
    FORMAL,
    CASUAL,
    PROFESSIONAL,
    FRIENDLY,
    CONCISE
}

/**
 * Represents the availability state of Gemini Nano / AICore on the current device.
 */
enum class GeminiAvailability {
    /** Availability has not been checked yet */
    UNKNOWN,
    /** Gemini Nano is available and the model is ready */
    AVAILABLE,
    /** Gemini Nano / AICore is not available on this device */
    UNAVAILABLE
}

/**
 * Manages on-device Gemini Nano inference via Android's AICore system service.
 *
 * AICore is available on Pixel 9+ devices running Android 14 (API 34) or higher.
 * This manager detects availability, initialises the model, and provides text-transformation
 * helpers consumed by [GeminiNanoAction].
 */
object GeminiNanoManager {

    @Volatile
    private var model: GenerativeModel? = null

    @Volatile
    var availability: GeminiAvailability = GeminiAvailability.UNKNOWN
        private set

    /** Returns true when Gemini Nano is ready to handle requests. */
    fun isAvailable(): Boolean = availability == GeminiAvailability.AVAILABLE

    /**
     * Checks whether Gemini Nano / AICore is available on this device and prepares the inference
     * model.  Safe to call multiple times; subsequent calls return the cached result immediately.
     *
     * @return true if the model is ready for inference, false otherwise.
     */
    suspend fun initialize(context: Context): Boolean {
        // AICore requires Android 14 (API 34) or higher
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
            availability = GeminiAvailability.UNAVAILABLE
            return false
        }

        if (availability == GeminiAvailability.AVAILABLE) return true
        if (availability == GeminiAvailability.UNAVAILABLE) return false

        return withContext(Dispatchers.IO) {
            try {
                val m = GenerativeModel(
                    generationConfig = generationConfig {
                        this.context = context.applicationContext
                        temperature = 0.2f
                        topK = 16
                        maxOutputTokens = 1024
                    }
                )
                m.prepareInferenceModel()
                model = m
                availability = GeminiAvailability.AVAILABLE
                true
            } catch (_: Exception) {
                availability = GeminiAvailability.UNAVAILABLE
                false
            }
        }
    }

    /**
     * Resets the cached availability state so that [initialize] will retry on the next call.
     * Useful after a system update or AICore service restart.
     */
    fun resetAvailability() {
        model = null
        availability = GeminiAvailability.UNKNOWN
    }

    // -----------------------------------------------------------------------------------------
    // Internal inference helper
    // -----------------------------------------------------------------------------------------

    private suspend fun generate(prompt: String): String? {
        val m = model ?: return null
        return withContext(Dispatchers.IO) {
            try {
                m.generateContent(prompt).text?.trim()?.ifBlank { null }
            } catch (_: Exception) {
                null
            }
        }
    }

    // -----------------------------------------------------------------------------------------
    // Public text-transformation APIs
    // -----------------------------------------------------------------------------------------

    /**
     * Rewrites [text] to improve clarity and fluency while keeping the same meaning and language.
     * Returns the rewritten text, or null if inference failed.
     */
    suspend fun rewriteText(text: String): String? = generate(
        "Rewrite the following text to improve clarity and fluency. " +
                "Keep the same meaning and language. " +
                "Output ONLY the rewritten text without any explanation:\n\n$text"
    )

    /**
     * Rewrites [text] with the requested [tone].
     * Returns the rewritten text, or null if inference failed.
     */
    suspend fun changeTone(text: String, tone: ToneStyle): String? {
        val description = when (tone) {
            ToneStyle.FORMAL -> "formal and professional"
            ToneStyle.CASUAL -> "casual and conversational"
            ToneStyle.PROFESSIONAL -> "business-professional"
            ToneStyle.FRIENDLY -> "friendly and warm"
            ToneStyle.CONCISE -> "concise and brief"
        }
        return generate(
            "Rewrite the following text with a $description tone. " +
                    "Keep the same language. " +
                    "Output ONLY the rewritten text without any explanation:\n\n$text"
        )
    }

    /**
     * Summarises [text] in 1-2 sentences.
     * Returns the summary, or null if inference failed.
     */
    suspend fun summarize(text: String): String? = generate(
        "Summarize the following text in 1-2 concise sentences. " +
                "Output ONLY the summary without any explanation:\n\n$text"
    )

    /**
     * Corrects spelling, grammar and punctuation in [text] without changing its meaning.
     * Returns the corrected text, or null if inference failed.
     */
    suspend fun proofread(text: String): String? = generate(
        "Fix all spelling, grammar, and punctuation errors in the following text. " +
                "Keep the same language and meaning. " +
                "Output ONLY the corrected text without any explanation:\n\n$text"
    )

    /**
     * Generates a natural 1-2 sentence continuation of [text] that matches its style and tone.
     * Returns only the new continuation (not the original text), or null if inference failed.
     */
    suspend fun continueWriting(text: String): String? = generate(
        "Continue the following text naturally with 1-2 sentences. " +
                "Match the style, tone, and language. " +
                "Output ONLY the new continuation without repeating the original text:\n\n$text"
    )

    /**
     * Generates up to 3 short, natural reply suggestions for [text].
     * Returns a list of reply strings (may be empty if inference failed or produced no output).
     */
    suspend fun generateSmartReplies(text: String): List<String> {
        val response = generate(
            "Generate exactly 3 short, natural reply options for the following message. " +
                    "Output ONLY 3 replies, one per line, with no numbering or bullet points:\n\n$text"
        ) ?: return emptyList()
        return response.lines()
            .map { it.trimStart('1', '2', '3', '.', ')', '-', '*', ' ').trim() }
            .filter { it.isNotBlank() }
            .take(3)
    }
}
