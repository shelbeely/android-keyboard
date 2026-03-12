package org.futo.inputmethod.latin.uix.actions

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.RowScope
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.wrapContentHeight
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.FilterChip
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.MutableState
import androidx.compose.runtime.mutableStateOf
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.launch
import org.futo.inputmethod.latin.R
import org.futo.inputmethod.latin.uix.Action
import org.futo.inputmethod.latin.uix.ActionInputTransaction
import org.futo.inputmethod.latin.uix.ActionWindow
import org.futo.inputmethod.latin.uix.KeyboardManagerForAction
import org.futo.inputmethod.latin.uix.PersistentActionState
import org.futo.inputmethod.latin.uix.PersistentStateInitialization
import org.futo.inputmethod.latin.uix.USE_GEMINI_NANO
import org.futo.inputmethod.latin.uix.getSetting
import org.futo.inputmethod.latin.uix.settings.UserSettingsMenu
import org.futo.inputmethod.latin.uix.settings.userSettingToggleDataStore
import org.futo.inputmethod.latin.xlm.GeminiAvailability
import org.futo.inputmethod.latin.xlm.GeminiNanoManager
import org.futo.inputmethod.latin.xlm.ToneStyle

// ------------------------------------------------------------------------------------------------
// Settings menu exposed from the title bar
// ------------------------------------------------------------------------------------------------

val GeminiNanoSettings = UserSettingsMenu(
    title = R.string.gemini_nano_settings_title,
    navPath = "geminiNano", registerNavPath = false,
    settings = listOf(
        userSettingToggleDataStore(
            title = R.string.gemini_nano_settings_enable,
            subtitle = R.string.gemini_nano_settings_enable_subtitle,
            setting = USE_GEMINI_NANO
        )
    )
)

// ------------------------------------------------------------------------------------------------
// Operation enum
// ------------------------------------------------------------------------------------------------

private enum class GeminiOperation(val titleRes: Int, val descRes: Int) {
    REWRITE(R.string.gemini_nano_op_rewrite, R.string.gemini_nano_op_rewrite_desc),
    TONE(R.string.gemini_nano_op_tone, R.string.gemini_nano_op_tone_desc),
    SUMMARIZE(R.string.gemini_nano_op_summarize, R.string.gemini_nano_op_summarize_desc),
    PROOFREAD(R.string.gemini_nano_op_proofread, R.string.gemini_nano_op_proofread_desc),
    CONTINUE(R.string.gemini_nano_op_continue, R.string.gemini_nano_op_continue_desc),
    SMART_REPLY(R.string.gemini_nano_op_smart_reply, R.string.gemini_nano_op_smart_reply_desc)
}

// ------------------------------------------------------------------------------------------------
// Text helpers
// ------------------------------------------------------------------------------------------------

/** Returns the last non-blank paragraph of [text], capped at 1 000 characters. */
private fun lastParagraph(text: CharSequence?): String {
    val str = text?.toString()?.trim() ?: return ""
    val paragraph = str.split("\n\n", "\n").lastOrNull { it.isNotBlank() } ?: str
    return paragraph.trim().takeLast(1000)
}

/** Returns the primary clipboard content as plain text, or null if empty. */
private fun Context.getClipboardText(): String? {
    val cm = getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
    return cm.primaryClip?.getItemAt(0)?.coerceToText(this)?.toString()?.trim()?.ifBlank { null }
}

/** Sets the primary clipboard to [text]. */
private fun Context.setClipboardText(text: String) {
    val cm = getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
    cm.setPrimaryClip(ClipData.newPlainText("AI result", text))
}

// ------------------------------------------------------------------------------------------------
// Persistent state – survives across window opens/closes while the keyboard is loaded
// ------------------------------------------------------------------------------------------------

private class GeminiNanoPersistentState(
    private val manager: KeyboardManagerForAction
) : PersistentActionState {

    val availabilityState: MutableState<GeminiAvailability> =
        mutableStateOf(GeminiNanoManager.availability)

    override suspend fun cleanUp() {}

    override fun close() {}

    override suspend fun onDeviceUnlocked() {
        checkAvailability()
    }

    suspend fun checkAvailability() {
        val ctx = manager.getContext()
        if (!ctx.getSetting(USE_GEMINI_NANO)) {
            availabilityState.value = GeminiAvailability.UNAVAILABLE
            return
        }
        val ready = GeminiNanoManager.initialize(ctx)
        availabilityState.value =
            if (ready) GeminiAvailability.AVAILABLE else GeminiAvailability.UNAVAILABLE
    }
}

// ------------------------------------------------------------------------------------------------
// Action window
// ------------------------------------------------------------------------------------------------

private class GeminiNanoWindow(
    private val manager: KeyboardManagerForAction,
    private val state: GeminiNanoPersistentState
) : ActionWindow() {

    // Capture text context at the moment the window is opened.
    private val inputTransaction: ActionInputTransaction = manager.createInputTransaction()
    private val initialBeforeCursor: CharSequence? = inputTransaction.textContext.beforeCursor

    // UI state
    private val selectedOp = mutableStateOf<GeminiOperation?>(null)
    private val selectedTone = mutableStateOf(ToneStyle.FORMAL)
    private val isProcessing = mutableStateOf(false)
    private val resultText = mutableStateOf<String?>(null)
    private val smartReplies = mutableStateOf<List<String>>(emptyList())
    private val errorOccurred = mutableStateOf(false)

    override fun close(): org.futo.inputmethod.latin.uix.CloseResult {
        inputTransaction.cancel()
        return super.close()
    }

    @Composable
    override fun windowName(): String = stringResource(R.string.action_gemini_nano_title)

    @Composable
    override fun WindowTitleBar(rowScope: RowScope) {
        with(rowScope) {
            Icon(
                painter = painterResource(R.drawable.ai_spark),
                contentDescription = null,
                modifier = Modifier
                    .size(20.dp)
                    .align(Alignment.CenterVertically),
                tint = MaterialTheme.colorScheme.primary
            )
            Text(
                windowName(),
                modifier = Modifier
                    .align(Alignment.CenterVertically)
                    .padding(start = 6.dp)
            )
            Spacer(modifier = Modifier.weight(1.0f))
        }
    }

    @OptIn(ExperimentalLayoutApi::class)
    @Composable
    override fun WindowContents(keyboardShown: Boolean) {
        val ctx = manager.getContext()
        val availability = state.availabilityState.value

        // Trigger availability check if still unknown
        LaunchedEffect(Unit) {
            if (availability == GeminiAvailability.UNKNOWN) {
                state.checkAvailability()
            }
        }

        Column(
            modifier = Modifier
                .fillMaxWidth()
                .wrapContentHeight()
                .verticalScroll(rememberScrollState())
                .padding(horizontal = 12.dp, vertical = 8.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            // ---- Availability banner ----
            when (availability) {
                GeminiAvailability.UNKNOWN -> {
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        CircularProgressIndicator(
                            modifier = Modifier.size(16.dp),
                            strokeWidth = 2.dp
                        )
                        Text(
                            text = stringResource(R.string.gemini_nano_checking),
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                    return@Column
                }

                GeminiAvailability.UNAVAILABLE -> {
                    Surface(
                        color = MaterialTheme.colorScheme.errorContainer,
                        shape = RoundedCornerShape(8.dp),
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Text(
                            text = stringResource(R.string.gemini_nano_unavailable),
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onErrorContainer,
                            modifier = Modifier.padding(10.dp)
                        )
                    }
                    return@Column
                }

                GeminiAvailability.AVAILABLE -> { /* proceed */ }
            }

            // ---- Operation selector ----
            FlowRow(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(6.dp),
                verticalArrangement = Arrangement.spacedBy(4.dp)
            ) {
                GeminiOperation.entries.forEach { op ->
                    FilterChip(
                        selected = selectedOp.value == op,
                        onClick = {
                            if (selectedOp.value != op) {
                                selectedOp.value = op
                                resultText.value = null
                                smartReplies.value = emptyList()
                                errorOccurred.value = false
                            }
                        },
                        label = {
                            Text(
                                stringResource(op.titleRes),
                                style = MaterialTheme.typography.labelSmall
                            )
                        }
                    )
                }
            }

            val op = selectedOp.value ?: return@Column

            // ---- Tone sub-selector (only for TONE operation) ----
            if (op == GeminiOperation.TONE) {
                FlowRow(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(6.dp),
                    verticalArrangement = Arrangement.spacedBy(4.dp)
                ) {
                    ToneStyle.entries.forEach { tone ->
                        val labelRes = when (tone) {
                            ToneStyle.FORMAL -> R.string.gemini_nano_tone_formal
                            ToneStyle.CASUAL -> R.string.gemini_nano_tone_casual
                            ToneStyle.PROFESSIONAL -> R.string.gemini_nano_tone_professional
                            ToneStyle.FRIENDLY -> R.string.gemini_nano_tone_friendly
                            ToneStyle.CONCISE -> R.string.gemini_nano_tone_concise
                        }
                        FilterChip(
                            selected = selectedTone.value == tone,
                            onClick = { selectedTone.value = tone },
                            label = {
                                Text(
                                    stringResource(labelRes),
                                    style = MaterialTheme.typography.labelSmall
                                )
                            }
                        )
                    }
                }
            }

            // ---- Determine source text ----
            val clipboardText = ctx.getClipboardText()

            val sourceText: String? = when (op) {
                GeminiOperation.SUMMARIZE -> clipboardText
                GeminiOperation.SMART_REPLY ->
                    initialBeforeCursor?.toString()?.trim()?.takeLast(1000)
                else -> lastParagraph(initialBeforeCursor)
            }

            val sourceLabelRes = when (op) {
                GeminiOperation.SUMMARIZE -> R.string.gemini_nano_clipboard_label
                else -> R.string.gemini_nano_source_label
            }

            // ---- Source text preview ----
            if (!sourceText.isNullOrBlank()) {
                Text(
                    text = stringResource(sourceLabelRes),
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
                Surface(
                    color = MaterialTheme.colorScheme.surfaceContainerLow,
                    shape = RoundedCornerShape(6.dp),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text(
                        text = sourceText.take(300).let {
                            if (sourceText.length > 300) "$it…" else it
                        },
                        style = MaterialTheme.typography.bodySmall.copy(
                            fontStyle = FontStyle.Italic
                        ),
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        modifier = Modifier.padding(8.dp),
                        maxLines = 4
                    )
                }
            } else {
                Text(
                    text = stringResource(R.string.gemini_nano_no_text),
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.error
                )
            }

            // ---- Run button (shown while idle, no result yet) ----
            if (!isProcessing.value &&
                resultText.value == null &&
                smartReplies.value.isEmpty() &&
                !errorOccurred.value
            ) {
                Button(
                    onClick = {
                        if (sourceText.isNullOrBlank()) return@Button
                        isProcessing.value = true
                        errorOccurred.value = false
                        manager.getLifecycleScope().launch {
                            runOperation(op, sourceText)
                        }
                    },
                    enabled = !sourceText.isNullOrBlank(),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text(stringResource(op.titleRes))
                }
            }

            // ---- Processing indicator ----
            if (isProcessing.value) {
                Box(modifier = Modifier.fillMaxWidth(), contentAlignment = Alignment.Center) {
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        CircularProgressIndicator(
                            modifier = Modifier.size(20.dp),
                            strokeWidth = 2.dp
                        )
                        Text(
                            stringResource(R.string.gemini_nano_processing),
                            style = MaterialTheme.typography.bodyMedium
                        )
                    }
                }
            }

            // ---- Error message ----
            if (errorOccurred.value) {
                Text(
                    text = stringResource(R.string.gemini_nano_error),
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.error
                )
                OutlinedButton(
                    onClick = { resetState() },
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text(stringResource(R.string.gemini_nano_result_dismiss))
                }
            }

            // ---- Result for text-output operations ----
            val result = resultText.value
            if (result != null) {
                Text(
                    text = stringResource(R.string.gemini_nano_result_label),
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
                Surface(
                    color = MaterialTheme.colorScheme.primaryContainer,
                    shape = RoundedCornerShape(6.dp),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text(
                        text = result,
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onPrimaryContainer,
                        modifier = Modifier.padding(10.dp)
                    )
                }

                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    Button(
                        onClick = {
                            insertResult(op, sourceText, result)
                            resetState()
                            manager.closeActionWindow()
                        },
                        modifier = Modifier.weight(1f)
                    ) {
                        Text(stringResource(R.string.gemini_nano_result_insert))
                    }

                    OutlinedButton(
                        onClick = {
                            ctx.setClipboardText(result)
                            resetState()
                        },
                        modifier = Modifier.weight(1f)
                    ) {
                        Text(stringResource(R.string.gemini_nano_result_copy))
                    }
                }

                Spacer(Modifier.height(4.dp))
                OutlinedButton(
                    onClick = { resetState() },
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text(stringResource(R.string.gemini_nano_result_dismiss))
                }
            }

            // ---- Smart reply list ----
            val replies = smartReplies.value
            if (replies.isNotEmpty()) {
                Text(
                    text = stringResource(R.string.gemini_nano_smart_reply_insert_hint),
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
                replies.forEach { reply ->
                    OutlinedButton(
                        onClick = {
                            manager.typeText(reply)
                            resetState()
                            manager.closeActionWindow()
                        },
                        modifier = Modifier.fillMaxWidth(),
                        shape = RoundedCornerShape(8.dp)
                    ) {
                        Text(
                            text = reply,
                            style = MaterialTheme.typography.bodySmall,
                            modifier = Modifier.fillMaxWidth()
                        )
                    }
                }
                Spacer(Modifier.height(4.dp))
                OutlinedButton(
                    onClick = { resetState() },
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text(stringResource(R.string.gemini_nano_result_dismiss))
                }
            }
        }
    }

    // ---- Private helpers ----

    private suspend fun runOperation(op: GeminiOperation, sourceText: String) {
        when (op) {
            GeminiOperation.SMART_REPLY -> {
                val replies = GeminiNanoManager.generateSmartReplies(sourceText)
                if (replies.isEmpty()) errorOccurred.value = true
                else smartReplies.value = replies
            }
            else -> {
                val result = when (op) {
                    GeminiOperation.REWRITE -> GeminiNanoManager.rewriteText(sourceText)
                    GeminiOperation.TONE -> GeminiNanoManager.changeTone(sourceText, selectedTone.value)
                    GeminiOperation.SUMMARIZE -> GeminiNanoManager.summarize(sourceText)
                    GeminiOperation.PROOFREAD -> GeminiNanoManager.proofread(sourceText)
                    GeminiOperation.CONTINUE -> GeminiNanoManager.continueWriting(sourceText)
                    GeminiOperation.SMART_REPLY -> null
                }
                if (result != null) resultText.value = result
                else errorOccurred.value = true
            }
        }
        isProcessing.value = false
    }

    private fun insertResult(op: GeminiOperation, sourceText: String?, result: String) {
        when (op) {
            // Append-style operations: just type the result after the cursor
            GeminiOperation.CONTINUE,
            GeminiOperation.SUMMARIZE -> manager.typeText(result)

            // Replace-style operations: delete source text, then type result
            GeminiOperation.REWRITE,
            GeminiOperation.TONE,
            GeminiOperation.PROOFREAD -> {
                if (!sourceText.isNullOrEmpty()) {
                    val beforeStr = initialBeforeCursor?.toString() ?: ""
                    // Only delete if the source text still appears at the end of before-cursor
                    val charsToDelete = if (beforeStr.endsWith(sourceText)) {
                        sourceText.length
                    } else {
                        0
                    }
                    if (charsToDelete > 0) manager.backspace(charsToDelete)
                }
                manager.typeText(result)
            }

            GeminiOperation.SMART_REPLY -> manager.typeText(result)
        }
    }

    private fun resetState() {
        resultText.value = null
        smartReplies.value = emptyList()
        errorOccurred.value = false
        isProcessing.value = false
    }
}

// ------------------------------------------------------------------------------------------------
// Action definition
// ------------------------------------------------------------------------------------------------

val GeminiNanoAction = Action(
    icon = R.drawable.ai_spark,
    name = R.string.action_gemini_nano_title,
    persistentState = { manager ->
        GeminiNanoPersistentState(manager)
    },
    persistentStateInitialization = PersistentStateInitialization.OnKeyboardLoad,
    windowImpl = { manager, persistent ->
        val state = persistent as GeminiNanoPersistentState
        GeminiNanoWindow(manager, state)
    },
    simplePressImpl = null,
    settingsMenu = GeminiNanoSettings
)
