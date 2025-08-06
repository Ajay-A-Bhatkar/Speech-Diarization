## **Speech Diarization Project Workflow**

### **1. Objective**

Build a system that takes audio recordings and:

- Automatically segments the audio according to who talks when (diarization).
- Transcribes the audio to text.
- Identifies known speakers based on training audio data.
- Outputs labeled transcription segments with speaker identity and gender information.
- Provides results for download (Excel).


### **2. Key Components**

1. **Audio Preprocessing**
    - Convert all audio to a standard format: mono channel, 16 kHz sample rate.
    - Uses `ffmpeg` tool wrapped in Python code via `ffmpeg-python`.
2. **Voice Activity Detection (VAD)**
    - Uses **Silero VAD** to find speech segments versus silence/background noise.
    - Helps focus processing on speech parts only.
3. **Speaker Embedding Extraction**
    - Uses **SpeechBrain's ECAPA-TDNN** speaker encoder.
    - Converts audio segments into fixed-length speaker embeddings (vector representations capturing speaker voice characteristics).
4. **Training a Speaker Classifier**
    - Uses **XGBoost** classifier on extracted embeddings.
    - Trains on labeled, segmented audio files where each folder corresponds to a known speaker.
    - Labels are encoded to numeric IDs (`LabelEncoder`).
5. **Speech Transcription**
    - Uses **WhisperX** model (e.g., `"medium"` sized).
    - Transcribes audio content to text and performs forced alignment for precise word timings.
6. **Speaker Diarization**
    - Uses **WhisperX's** diarization pipeline with a pretrained **pyannote.audio** model accessed via Hugging Face (requires API token).
    - Segments audio by speaker clusters without knowing speaker identity.
7. **Assigning Speaker Labels**
    - Extract embeddings from diarized segments.
    - Classify embeddings with trained XGBoost model to assign known speaker names.
    - Unknown or unrecognized segments are labeled as `"unknown"`.
8. **Gender Detection**
    - Uses **gender-guesser** package on predicted speaker names for gender tagging.
9. **Output formatting**
    - Combines speaker labels, gender, segment start and end times.
    - Saves results to an Excel file for download.

### **3. Flow of Your Code**

#### **A. Setup**

- Import required libraries.
- Set device (`cuda` if GPU available else `cpu`).
- Fixed Hugging Face token for diarization model.
- Load the SpeechBrain speaker encoder with caching to reduce reload overhead.


#### **B. Step 1: Train Speaker Classifier**

- User uploads a **ZIP file** containing **training audio**, organized by speaker folders.
- ZIP is extracted into a temporary directory.
- For each audio file:
    - Load and convert to mono 16kHz.
    - Split into fixed-duration segments (~3 seconds).
    - Extract speaker embeddings from each segment.
    - Collect embeddings and corresponding speaker labels.
- Labels are encoded into integers.
- Train XGBoost classifier on extracted embeddings.
- Trained model and label encoder saved locally and cached in session state.


#### **C. Step 2: Diarize and Transcribe Input Audio**

- User uploads an input audio file to analyze.
- Audio preprocessed (mono, 16kHz) with `ffmpeg`.
- Run diarization pipeline (via WhisperX + pyannote model) on preprocessed audio.
- Diarization output: speaker segments with times.
- Run transcription with WhisperX:
    - Detect language automatically.
    - Transcribe using appropriate language model.
    - Align word timings for precise timestamps.
- For each diarized segment:
    - Extract embedding from audio.
    - Classify speaker identity using trained XGBoost model.
    - Guess speaker gender.
- Collect results as DataFrame, including start/end times, speaker name, gender.


#### **D. Output**

- Display diarization segments and speaker-labeled transcript segments in Streamlit.
- Provide downloadable Excel file (.xlsx) with segments and speaker info.


### **4. Key Code Highlights and Concepts**

**Device and compute type:**

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "float32"
```

Ensures proper precision to avoid memory errors during model inference.

**Silero VAD:**
Imports and runs to segment speech regions.

**Speaker embedding extraction:**
Extract segment embeddings:

```python
def extract_embeddings(folder_path, segment_duration=3):
    ...
    emb = classifier.encode_batch(segment)
```

**Training XGBoost classifier:**

```python
le = LabelEncoder()
y_encoded = le.fit_transform(y)
clf = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
clf.fit(X, y_encoded)
```

**Using WhisperX for diarization and transcription:**

```python
diarize_model = DiarizationPipeline(use_auth_token=HF_TOKEN, device=device)
diarized_segments = diarize_model(audio_path)

model_transcribe = whisperx.load_model("medium", device=device, compute_type=compute_type)
transcription = model_transcribe.transcribe(audio_path, batch_size=2, language=detected_lang)

model_align, metadata = whisperx.load_align_model(language_code=detected_lang, device=device)
aligned = whisperx.align(transcription["segments"], model_align, metadata, audio_path, device)
```

**Classify each diarized segment speaker:**

```python
for idx, row in diarized_segments.iterrows():
    segment_audio = load_segment(audio, row['start'], row['end'])
    emb = classifier.encode_batch(segment_audio).detach().cpu().numpy()
    pred = clf.predict([emb])
    speaker_name = label_encoder.inverse_transform(pred)
```

**Display and export final results:**

Results combined into a Pandas DataFrame and saved as an Excel:

```python
df.to_excel("diarization_output.xlsx", index=False)
```


### **5. Important Notes**

- Hugging Face token **must be valid** and associated with accepted terms for diarization models.
- Memory management: use smaller WhisperX models and small batch sizes if hardware is limited.
- Preprocessing audio to uniform sample rate helps standardize results.
- Embeddings are sensitive to segment length and quality — best if training samples are representative.


### **6. Presentation Tips**

- Start with **Problem Statement:** Why diarization + speaker identification is important.
- Explain high-level **workflow:** Preprocessing → VAD → Embedding extraction → Classification → Transcription → Output.
- Show code snippet flows for each major block.
- Highlight innovations:
    - Using WhisperX for diarization and alignment.
    - Speaker embeddings and classifier trained on custom audio data.
    - Web app interface via Streamlit for interactiveness.
- Discuss challenges and runtime considerations (memory usage, compute).
- Show demo outputs (screenshots or live demo) including diarized segments, speaker labels, and downloadable transcript.
- End with future directions: adding real-time streaming, UI improvements, multi-language support.

If you want, you can share that notebook or more details, and I can help generate slides or a presentation script! Let me know.

<div style="text-align: center">⁂</div>

[^1]: SpeechDiarization.ipynb

