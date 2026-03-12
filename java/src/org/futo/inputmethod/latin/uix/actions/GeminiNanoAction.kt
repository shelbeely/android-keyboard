package org.futo.inputmethod.latin.uix.actions

import android.content.ClipboardManager
import android.content.Context
import android.os.Build
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.FilterChip
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Tab
import androidx.compose.material3.TabRow
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.MutableState
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.suspendCancellableCoroutine
import org.futo.inputmethod.latin.R
import org.futo.inputmethod.latin.uix.Action
import org.futo.inputmethod.latin.uix.ActionWindow
import org.futo.inputmethod.latin.uix.CloseResult
import org.futo.inputmethod.latin.uix.KeyboardManagerForAction
import org.futo.inputmethod.latin.uix.PersistentActionState
import com.google.mlkit.genai.common.DownloadCallback
import com.google.mlkit.genai.common.FeatureAvailability
import com.google.mlkit.genai.proofreading.Proofreading
import com.google.mlkit.genai.proofreading.ProofreadingClient
import com.google.mlkit.genai.proofreading.ProofreadingRequest
import com.google.mlkit.genai.summarization.Summarization
import com.google.mlkit.genai.summarization.SummarizationClient
import com.google.mlkit.genai.summarization.SummarizationRequest
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException

// ──────────────────────────────────────────────────────────────────────────────
// Availability model
// ──────────────────────────────────────────────────────────────────────────────

/** Represents the runtime availability of a Gemini Nano ML-Kit feature. */
sealed class GeminiFeatureStatus {
    object Checking : GeminiFeatureStatus()
    object Available : GeminiFeatureStatus()
    object Downloadable : GeminiFeatureStatus()
    object Downloading : GeminiFeatureStatus()
    object NotSupported : GeminiFeatureStatus()
}

// ──────────────────────────────────────────────────────────────────────────────
// Thin helpers around the ML-Kit GenAI Tasks API
// ──────────────────────────────────────────────────────────────────────────────

/** Minimum API level required for AICore / Gemini Nano. */
private val GEMINI_NANO_MIN_API = Build.VERSION_CODES.UPSIDE_DOWN_CAKE // API 34

private fun Int.toGeminiFeatureStatus(): GeminiFeatureStatus = when (this) {
    FeatureAvailability.AVAILABLE -> GeminiFeatureStatus.Available
    FeatureAvailability.DOWNLOADABLE -> GeminiFeatureStatus.Downloadable
    FeatureAvailability.DOWNLOADING -> GeminiFeatureStatus.Downloading
    else -> GeminiFeatureStatus.NotSupported
}

/** Suspending wrapper: check whether the proofreading feature is available. */
private suspend fun checkProofreadingAvailability(context: Context): GeminiFeatureStatus {
    if (Build.VERSION.SDK_INT < GEMINI_NANO_MIN_API) return GeminiFeatureStatus.NotSupported
    return suspendCancellableCoroutine { cont ->
        Proofreading.getClient(context)
            .checkFeatureAvailability()
            .addOnSuccessListener { status -> cont.resume(status.toGeminiFeatureStatus()) }
            .addOnFailureListener { cont.resume(GeminiFeatureStatus.NotSupported) }
    }
}

/** Suspending wrapper: check whether the summarization feature is available. */
private suspend fun checkSummarizationAvailability(context: Context): GeminiFeatureStatus {
    if (Build.VERSION.SDK_INT < GEMINI_NANO_MIN_API) return GeminiFeatureStatus.NotSupported
    return suspendCancellableCoroutine { cont ->
        Summarization.getClient(context)
            .checkFeatureAvailability()
            .addOnSuccessListener { status -> cont.resume(status.toGeminiFeatureStatus()) }
            .addOnFailureListener { cont.resume(GeminiFeatureStatus.NotSupported) }
    }
}

/** Request download of the proofreading model. */
private fun requestProofreadingDownload(context: Context) {
    if (Build.VERSION.SDK_INT < GEMINI_NANO_MIN_API) return
    Proofreading.getClient(context).requestFeatureDownload(object : DownloadCallback {
        override fun onDownloadStarted(bytesToDownload: Long) {}
        override fun onDownloadFailed(e: Exception) {}
        override fun onDownloadCompleted() {}
        override fun onDownloadProgress(downloadedBytes: Long) {}
    })
}

/** Request download of the summarization model. */
private fun requestSummarizationDownload(context: Context) {
    if (Build.VERSION.SDK_INT < GEMINI_NANO_MIN_API) return
    Summarization.getClient(context).requestFeatureDownload(object : DownloadCallback {
        override fun onDownloadStarted(bytesToDownload: Long) {}
        override fun onDownloadFailed(e: Exception) {}
        override fun onDownloadCompleted() {}
        override fun onDownloadProgress(downloadedBytes: Long) {}
    })
}

// ──────────────────────────────────────────────────────────────────────────────
// Tone constants
// ──────────────────────────────────────────────────────────────────────────────

private data class ToneOption(val labelRes: Int, val style: Int)

private val TONE_OPTIONS = listOf(
    ToneOption(R.string.action_gemini_nano_tone_formal,    ProofreadingRequest.REWRITE_STYLE_FORMAL),
    ToneOption(R.string.action_gemini_nano_tone_friendly,  ProofreadingRequest.REWRITE_STYLE_FRIENDLY),
    ToneOption(R.string.action_gemini_nano_tone_concise,   ProofreadingRequest.REWRITE_STYLE_CONCISE),
    ToneOption(R.string.action_gemini_nano_tone_elaborate, ProofreadingRequest.REWRITE_STYLE_ELABORATE),
)

// ──────────────────────────────────────────────────────────────────────────────
// Persistent state (holds open ML-Kit clients across window open/close)
// ──────────────────────────────────────────────────────────────────────────────

class GeminiNanoPersistentState(manager: KeyboardManagerForAction) : PersistentActionState {
    private val context: Context = manager.getContext()

    val proofreadingClient: ProofreadingClient? =
        if (Build.VERSION.SDK_INT >= GEMINI_NANO_MIN_API) Proofreading.getClient(context) else null

    val summarizationClient: SummarizationClient? =
        if (Build.VERSION.SDK_INT >= GEMINI_NANO_MIN_API) Summarization.getClient(context) else null

    override suspend fun cleanUp() { /* clients are lightweight, nothing to release proactively */ }

    override fun close() {
        proofreadingClient?.close()
        summarizationClient?.close()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// UI helpers
// ──────────────────────────────────────────────────────────────────────────────

/** Status banner shown while checking / when unavailable. */
@Composable
private fun StatusBanner(status: GeminiFeatureStatus, onRequestDownload: () -> Unit) {
    val msg = when (status) {
        GeminiFeatureStatus.Checking -> stringResource(R.string.action_gemini_nano_status_checking)
        GeminiFeatureStatus.NotSupported -> stringResource(R.string.action_gemini_nano_status_not_supported)
        GeminiFeatureStatus.Downloadable -> stringResource(R.string.action_gemini_nano_status_downloadable)
        GeminiFeatureStatus.Downloading -> stringResource(R.string.action_gemini_nano_status_downloading)
        GeminiFeatureStatus.Available -> return
    }
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        if (status == GeminiFeatureStatus.Checking) {
            CircularProgressIndicator(modifier = Modifier.padding(bottom = 8.dp))
        }
        Text(
            text = msg,
            style = MaterialTheme.typography.bodyMedium,
            textAlign = TextAlign.Center
        )
        if (status == GeminiFeatureStatus.Downloadable) {
            Spacer(modifier = Modifier.height(8.dp))
            Button(onClick = onRequestDownload) {
                Text(stringResource(R.string.action_gemini_nano_request_download_button))
            }
        }
    }
}

/** Tone-transform sub-panel. */
@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun TonePanel(
    inputText: MutableState<String>,
    selectedToneIndex: MutableState<Int>,
    result: MutableState<String>,
    isRunning: MutableState<Boolean>,
    error: MutableState<String?>,
    onRun: (String, Int) -> Unit,
    onInsert: (String) -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(8.dp)
    ) {
        OutlinedTextField(
            value = inputText.value,
            onValueChange = { inputText.value = it },
            modifier = Modifier.fillMaxWidth(),
            label = { Text(stringResource(R.string.action_gemini_nano_input_hint)) },
            maxLines = 4
        )
        Spacer(modifier = Modifier.height(8.dp))
        FlowRow(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            TONE_OPTIONS.forEachIndexed { idx, tone ->
                FilterChip(
                    selected = selectedToneIndex.value == idx,
                    onClick = { selectedToneIndex.value = idx },
                    label = { Text(stringResource(tone.labelRes)) }
                )
            }
        }
        Spacer(modifier = Modifier.height(8.dp))
        Button(
            onClick = {
                onRun(inputText.value, TONE_OPTIONS[selectedToneIndex.value].style)
            },
            enabled = !isRunning.value && inputText.value.isNotBlank(),
            modifier = Modifier.fillMaxWidth()
        ) {
            if (isRunning.value) {
                CircularProgressIndicator(
                    modifier = Modifier
                        .padding(end = 8.dp)
                        .height(16.dp),
                    strokeWidth = 2.dp
                )
            }
            Text(
                if (isRunning.value) stringResource(R.string.action_gemini_nano_running)
                else stringResource(R.string.action_gemini_nano_run_button)
            )
        }
        error.value?.let {
            Spacer(modifier = Modifier.height(8.dp))
            Text(
                text = it,
                color = MaterialTheme.colorScheme.error,
                style = MaterialTheme.typography.bodySmall
            )
        }
        if (result.value.isNotEmpty()) {
            Spacer(modifier = Modifier.height(8.dp))
            Text(
                stringResource(R.string.action_gemini_nano_result_label),
                style = MaterialTheme.typography.labelMedium
            )
            Surface(
                color = MaterialTheme.colorScheme.surfaceVariant,
                shape = MaterialTheme.shapes.small,
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(vertical = 4.dp)
            ) {
                Text(
                    text = result.value,
                    modifier = Modifier.padding(8.dp),
                    style = MaterialTheme.typography.bodyMedium
                )
            }
            Button(
                onClick = { onInsert(result.value) },
                modifier = Modifier.fillMaxWidth()
            ) {
                Text(stringResource(R.string.action_gemini_nano_insert_button))
            }
        }
    }
}

/** Rewrite sub-panel. */
@Composable
private fun RewritePanel(
    inputText: MutableState<String>,
    result: MutableState<String>,
    isRunning: MutableState<Boolean>,
    error: MutableState<String?>,
    onRun: (String) -> Unit,
    onInsert: (String) -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(8.dp)
    ) {
        OutlinedTextField(
            value = inputText.value,
            onValueChange = { inputText.value = it },
            modifier = Modifier.fillMaxWidth(),
            label = { Text(stringResource(R.string.action_gemini_nano_input_hint)) },
            maxLines = 4
        )
        Spacer(modifier = Modifier.height(8.dp))
        Button(
            onClick = { onRun(inputText.value) },
            enabled = !isRunning.value && inputText.value.isNotBlank(),
            modifier = Modifier.fillMaxWidth()
        ) {
            if (isRunning.value) {
                CircularProgressIndicator(
                    modifier = Modifier
                        .padding(end = 8.dp)
                        .height(16.dp),
                    strokeWidth = 2.dp
                )
            }
            Text(
                if (isRunning.value) stringResource(R.string.action_gemini_nano_running)
                else stringResource(R.string.action_gemini_nano_run_button)
            )
        }
        error.value?.let {
            Spacer(modifier = Modifier.height(8.dp))
            Text(
                text = it,
                color = MaterialTheme.colorScheme.error,
                style = MaterialTheme.typography.bodySmall
            )
        }
        if (result.value.isNotEmpty()) {
            Spacer(modifier = Modifier.height(8.dp))
            Text(
                stringResource(R.string.action_gemini_nano_result_label),
                style = MaterialTheme.typography.labelMedium
            )
            Surface(
                color = MaterialTheme.colorScheme.surfaceVariant,
                shape = MaterialTheme.shapes.small,
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(vertical = 4.dp)
            ) {
                Text(
                    text = result.value,
                    modifier = Modifier.padding(8.dp),
                    style = MaterialTheme.typography.bodyMedium
                )
            }
            Button(
                onClick = { onInsert(result.value) },
                modifier = Modifier.fillMaxWidth()
            ) {
                Text(stringResource(R.string.action_gemini_nano_insert_button))
            }
        }
    }
}

/** Summarize-clipboard sub-panel. */
@Composable
private fun SummarizePanel(
    clipboardText: MutableState<String>,
    result: MutableState<String>,
    isRunning: MutableState<Boolean>,
    error: MutableState<String?>,
    onRun: (String) -> Unit,
    onInsert: (String) -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(8.dp)
    ) {
        Surface(
            color = MaterialTheme.colorScheme.surfaceVariant,
            shape = MaterialTheme.shapes.small,
            modifier = Modifier
                .fillMaxWidth()
                .height(80.dp)
        ) {
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(8.dp)
            ) {
                Text(
                    text = clipboardText.value.ifEmpty {
                        stringResource(R.string.action_gemini_nano_clipboard_hint)
                    },
                    style = MaterialTheme.typography.bodySmall,
                    color = if (clipboardText.value.isEmpty())
                        MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.6f)
                    else
                        MaterialTheme.colorScheme.onSurfaceVariant,
                    maxLines = 4
                )
            }
        }
        Spacer(modifier = Modifier.height(8.dp))
        Button(
            onClick = { onRun(clipboardText.value) },
            enabled = !isRunning.value && clipboardText.value.isNotBlank(),
            modifier = Modifier.fillMaxWidth()
        ) {
            if (isRunning.value) {
                CircularProgressIndicator(
                    modifier = Modifier
                        .padding(end = 8.dp)
                        .height(16.dp),
                    strokeWidth = 2.dp
                )
            }
            Text(
                if (isRunning.value) stringResource(R.string.action_gemini_nano_running)
                else stringResource(R.string.action_gemini_nano_run_button)
            )
        }
        error.value?.let {
            Spacer(modifier = Modifier.height(8.dp))
            Text(
                text = it,
                color = MaterialTheme.colorScheme.error,
                style = MaterialTheme.typography.bodySmall
            )
        }
        if (result.value.isNotEmpty()) {
            Spacer(modifier = Modifier.height(8.dp))
            Text(
                stringResource(R.string.action_gemini_nano_result_label),
                style = MaterialTheme.typography.labelMedium
            )
            Surface(
                color = MaterialTheme.colorScheme.surfaceVariant,
                shape = MaterialTheme.shapes.small,
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(vertical = 4.dp)
            ) {
                Text(
                    text = result.value,
                    modifier = Modifier.padding(8.dp),
                    style = MaterialTheme.typography.bodyMedium
                )
            }
            Button(
                onClick = { onInsert(result.value) },
                modifier = Modifier.fillMaxWidth()
            ) {
                Text(stringResource(R.string.action_gemini_nano_insert_button))
            }
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Action window
// ──────────────────────────────────────────────────────────────────────────────

private class GeminiNanoActionWindow(
    val manager: KeyboardManagerForAction,
    val state: GeminiNanoPersistentState
) : ActionWindow() {
    private val context = manager.getContext()

    // ── Combined availability (worst of proofreading + summarization) ─────────
    private val availabilityStatus: MutableState<GeminiFeatureStatus> =
        mutableStateOf(GeminiFeatureStatus.Checking)

    // ── Tab selection (0 = Tone, 1 = Rewrite, 2 = Summarize) ─────────────────
    private val selectedTab: MutableState<Int> = mutableIntStateOf(0)

    // ── Per-tab state ──────────────────────────────────────────────────────────
    private val toneInput   = mutableStateOf("")
    private val toneResult  = mutableStateOf("")
    private val toneRunning = mutableStateOf(false)
    private val toneError   = mutableStateOf<String?>(null)
    private val selectedToneIndex = mutableIntStateOf(0)

    private val rewriteInput   = mutableStateOf("")
    private val rewriteResult  = mutableStateOf("")
    private val rewriteRunning = mutableStateOf(false)
    private val rewriteError   = mutableStateOf<String?>(null)

    private val clipboardText   = mutableStateOf("")
    private val summarizeResult = mutableStateOf("")
    private val summarizeRunning = mutableStateOf(false)
    private val summarizeError  = mutableStateOf<String?>(null)

    init {
        // Seed text fields with context from before cursor
        val transaction = manager.createInputTransaction()
        val contextText = transaction.textContext.beforeCursor
            ?.trimEnd()
            ?.takeLast(500) // limit to last ~500 chars
            ?.toString() ?: ""
        transaction.cancel()

        if (contextText.isNotBlank()) {
            toneInput.value    = contextText
            rewriteInput.value = contextText
        }

        // Read clipboard text
        val clipMgr = context.getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
        clipMgr.primaryClip
            ?.getItemAt(0)
            ?.coerceToText(context)
            ?.toString()
            ?.takeIf { it.isNotBlank() }
            ?.let { clipboardText.value = it }

        // Async availability check
        manager.getLifecycleScope().launch(Dispatchers.IO) {
            val proofStatus = checkProofreadingAvailability(context)
            val sumStatus   = checkSummarizationAvailability(context)
            // Overall status: available only if both are available
            availabilityStatus.value = when {
                proofStatus == GeminiFeatureStatus.Available &&
                sumStatus   == GeminiFeatureStatus.Available -> GeminiFeatureStatus.Available
                proofStatus == GeminiFeatureStatus.Downloadable ||
                sumStatus   == GeminiFeatureStatus.Downloadable -> GeminiFeatureStatus.Downloadable
                proofStatus == GeminiFeatureStatus.Downloading ||
                sumStatus   == GeminiFeatureStatus.Downloading -> GeminiFeatureStatus.Downloading
                else -> GeminiFeatureStatus.NotSupported
            }
        }
    }

    // ── Inference helpers ──────────────────────────────────────────────────────

    private fun runToneTransform(text: String, style: Int) {
        if (state.proofreadingClient == null) return
        if (text.isBlank() || text.split("\\s+".toRegex()).size < 2) {
            toneError.value = context.getString(R.string.action_gemini_nano_input_too_short)
            return
        }
        toneRunning.value = true
        toneError.value   = null
        toneResult.value  = ""
        manager.getLifecycleScope().launch(Dispatchers.IO) {
            try {
                val request = ProofreadingRequest.builder(text)
                    .setRewriteStyle(style)
                    .build()
                val result = suspendCancellableCoroutine<String> { cont ->
                    state.proofreadingClient.runInference(request)
                        .addOnSuccessListener { res ->
                            cont.resume(res.results.firstOrNull()?.text ?: text)
                        }
                        .addOnFailureListener { e ->
                            cont.resumeWithException(e)
                        }
                }
                toneResult.value = result
            } catch (e: Exception) {
                toneError.value = context.getString(R.string.action_gemini_nano_error, e.localizedMessage ?: e.javaClass.simpleName)
            } finally {
                toneRunning.value = false
            }
        }
    }

    private fun runRewrite(text: String) {
        if (state.proofreadingClient == null) return
        if (text.isBlank() || text.split("\\s+".toRegex()).size < 2) {
            rewriteError.value = context.getString(R.string.action_gemini_nano_input_too_short)
            return
        }
        rewriteRunning.value = true
        rewriteError.value   = null
        rewriteResult.value  = ""
        manager.getLifecycleScope().launch(Dispatchers.IO) {
            try {
                val request = ProofreadingRequest.builder(text)
                    .setRewriteStyle(ProofreadingRequest.REWRITE_STYLE_NORMAL)
                    .build()
                val result = suspendCancellableCoroutine<String> { cont ->
                    state.proofreadingClient.runInference(request)
                        .addOnSuccessListener { res ->
                            cont.resume(res.results.firstOrNull()?.text ?: text)
                        }
                        .addOnFailureListener { e ->
                            cont.resumeWithException(e)
                        }
                }
                rewriteResult.value = result
            } catch (e: Exception) {
                rewriteError.value = context.getString(R.string.action_gemini_nano_error, e.localizedMessage ?: e.javaClass.simpleName)
            } finally {
                rewriteRunning.value = false
            }
        }
    }

    private fun runSummarize(text: String) {
        if (state.summarizationClient == null) return
        if (text.isBlank()) {
            summarizeError.value = context.getString(R.string.action_gemini_nano_no_clipboard_text)
            return
        }
        summarizeRunning.value = true
        summarizeError.value   = null
        summarizeResult.value  = ""
        manager.getLifecycleScope().launch(Dispatchers.IO) {
            try {
                val request = SummarizationRequest.builder(text).build()
                val result = suspendCancellableCoroutine<String> { cont ->
                    state.summarizationClient.runInference(request)
                        .addOnSuccessListener { res ->
                            cont.resume(res.summary)
                        }
                        .addOnFailureListener { e ->
                            cont.resumeWithException(e)
                        }
                }
                summarizeResult.value = result
            } catch (e: Exception) {
                summarizeError.value = context.getString(R.string.action_gemini_nano_error, e.localizedMessage ?: e.javaClass.simpleName)
            } finally {
                summarizeRunning.value = false
            }
        }
    }

    private fun insertResult(text: String) {
        manager.typeText(text)
        manager.closeActionWindow()
    }

    // ── Composable UI ──────────────────────────────────────────────────────────

    @Composable
    override fun windowName(): String = stringResource(R.string.action_gemini_nano_title)

    @Composable
    override fun WindowContents(keyboardShown: Boolean) {
        val status = availabilityStatus.value

        Column(modifier = Modifier.fillMaxSize()) {
            // Availability banner (or nothing when available)
            StatusBanner(
                status = status,
                onRequestDownload = {
                    requestProofreadingDownload(context)
                    requestSummarizationDownload(context)
                    availabilityStatus.value = GeminiFeatureStatus.Downloading
                }
            )

            if (status != GeminiFeatureStatus.Available) return@Column

            // Tab bar
            val tabLabels = listOf(
                stringResource(R.string.action_gemini_nano_tab_tone),
                stringResource(R.string.action_gemini_nano_tab_rewrite),
                stringResource(R.string.action_gemini_nano_tab_summarize)
            )
            TabRow(selectedTabIndex = selectedTab.value) {
                tabLabels.forEachIndexed { index, label ->
                    Tab(
                        selected = selectedTab.value == index,
                        onClick  = { selectedTab.value = index },
                        text     = { Text(label) }
                    )
                }
            }

            // Tab content
            when (selectedTab.value) {
                0 -> TonePanel(
                    inputText          = toneInput,
                    selectedToneIndex  = selectedToneIndex,
                    result             = toneResult,
                    isRunning          = toneRunning,
                    error              = toneError,
                    onRun              = { text, style -> runToneTransform(text, style) },
                    onInsert           = ::insertResult
                )
                1 -> RewritePanel(
                    inputText  = rewriteInput,
                    result     = rewriteResult,
                    isRunning  = rewriteRunning,
                    error      = rewriteError,
                    onRun      = ::runRewrite,
                    onInsert   = ::insertResult
                )
                2 -> SummarizePanel(
                    clipboardText = clipboardText,
                    result        = summarizeResult,
                    isRunning     = summarizeRunning,
                    error         = summarizeError,
                    onRun         = ::runSummarize,
                    onInsert      = ::insertResult
                )
            }
        }
    }

    override fun close(): CloseResult {
        // Cancel any outstanding (blocking) work; coroutine jobs are automatically
        // cancelled when the lifecycle scope is cleared by the framework.
        return CloseResult.Default
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Action definition
// ──────────────────────────────────────────────────────────────────────────────

val GeminiNanoAction = Action(
    icon             = R.drawable.sparkle,
    name             = R.string.action_gemini_nano_title,
    simplePressImpl  = null,
    persistentState  = { GeminiNanoPersistentState(it) },
    windowImpl       = { manager, persistentState ->
        GeminiNanoActionWindow(
            manager = manager,
            state   = persistentState as GeminiNanoPersistentState
        )
    }
)
