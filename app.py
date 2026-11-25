import hashlib
import io
import os
import subprocess
import tempfile
import time
import streamlit as st
from google import genai
from google.genai import types
from google.genai.errors import APIError
from docx import Document
from fpdf import FPDF

st.set_page_config(
    page_title="Audio to Script Studio",
    page_icon="🎙️",
    layout="wide",
)

DEFAULT_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.5-transcribe",
    "gemini-4-hydrogen",
]

INLINE_SIZE_LIMIT = 15 * 1024 * 1024  # 15 MB


# --- Text & Encoding Helpers ---

def to_safe_text(text: str) -> str:
    if not text:
        return ""
    return (
        text.replace("—", "-")
        .replace("–", "-")
        .replace("’", "'")
        .replace("‘", "'")
        .replace("“", '"')
        .replace("”", '"')
        .encode("ascii", "backslashreplace")
        .decode("ascii")
    )


def guess_mime_type(suffix: str) -> str:
    ext = suffix.lower()
    mapping = {
        ".mp3": "audio/mp3",
        ".wav": "audio/wav",
        ".m4a": "audio/mp4",
        ".ogg": "audio/ogg",
        ".flac": "audio/flac",
        ".aac": "audio/aac",
        ".webm": "audio/webm",
    }
    return mapping.get(ext, "audio/mp3")


# --- Model Discovery ---

def get_available_models(api_key: str) -> list[str]:
    """Lists models available for the given API key that support generateContent."""
    clean_key = api_key.strip()
    if not clean_key:
        return DEFAULT_MODELS

    cache_key = f"models_{hashlib.md5(clean_key.encode()).hexdigest()}"
    if cache_key in st.session_state:
        return st.session_state[cache_key]

    try:
        client = genai.Client(api_key=clean_key)
        found = []
        for m in client.models.list():
            name = getattr(m, "name", "")
            model_id = name.replace("models/", "", 1)
            actions = getattr(m, "supported_actions", None) or getattr(m, "supported_generation_methods", None) or []

            # Skip image, TTS, embedding, live, and veo models
            lower_id = model_id.lower()
            if any(skip in lower_id for skip in ("image", "tts", "embedding", "live", "veo", "lyria", "imagen")):
                continue

            if not actions or "generateContent" in actions or "generate_content" in actions:
                if model_id:
                    found.append(model_id)

        # Put preferred fast models first if present
        ordered = []
        for pref in DEFAULT_MODELS:
            if pref in found and pref not in ordered:
                ordered.append(pref)
        for m_id in found:
            if m_id not in ordered:
                ordered.append(m_id)

        result = ordered or DEFAULT_MODELS
        st.session_state[cache_key] = result
        return result
    except Exception:
        return DEFAULT_MODELS


# --- Document Exporters ---

def create_word_doc(title: str, speaker_info: str, transcript_text: str) -> bytes:
    doc = Document()
    doc.add_heading(title, level=1)

    if speaker_info.strip():
        p = doc.add_paragraph()
        run = p.add_run(f"Speaker / Context: {speaker_info}")
        run.bold = True

    doc.add_paragraph("")

    for paragraph in transcript_text.split("\n\n"):
        clean_p = paragraph.strip()
        if clean_p:
            doc.add_paragraph(clean_p)

    buffer = io.BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer.getvalue()


def create_pdf_doc(title: str, speaker_info: str, transcript_text: str) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Helvetica", style="B", size=16)
    pdf.multi_cell(0, 10, to_safe_text(title))
    pdf.ln(2)

    if speaker_info.strip():
        pdf.set_font("Helvetica", style="I", size=11)
        pdf.multi_cell(0, 7, to_safe_text(f"Speaker / Context: {speaker_info}"))
        pdf.ln(4)

    pdf.set_font("Helvetica", size=11)
    for paragraph in transcript_text.split("\n\n"):
        clean_p = paragraph.strip()
        if clean_p:
            pdf.multi_cell(0, 6.5, to_safe_text(clean_p))
            pdf.ln(3)

    return bytes(pdf.output())


# --- Speed & Reliability Helpers ---

def has_ffmpeg_cached() -> bool:
    if "has_ffmpeg" not in st.session_state:
        try:
            subprocess.run(
                ["ffmpeg", "-version"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
            st.session_state["has_ffmpeg"] = True
        except Exception:
            st.session_state["has_ffmpeg"] = False
    return st.session_state["has_ffmpeg"]


def optimize_audio_if_needed(raw_bytes: bytes, original_name: str, compress_large: bool, status_cb=None) -> tuple[bytes, str]:
    suffix = os.path.splitext(original_name)[1].lower() or ".mp3"
    if not compress_large or not has_ffmpeg_cached() or len(raw_bytes) < 5 * 1024 * 1024:
        return raw_bytes, suffix

    if status_cb:
        status_cb("Compressing large audio to 16kHz mono MP3...")

    in_path = None
    out_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f_in:
            f_in.write(raw_bytes)
            in_path = f_in.name

        out_path = in_path + "_fast.mp3"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                in_path,
                "-ac",
                "1",
                "-ar",
                "16000",
                "-b:a",
                "40k",
                out_path,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        with open(out_path, "rb") as f_out:
            opt_bytes = f_out.read()
        if opt_bytes and len(opt_bytes) < len(raw_bytes):
            return opt_bytes, ".mp3"
    except Exception:
        pass
    finally:
        for p in (in_path, out_path):
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass

    return raw_bytes, suffix


def build_audio_content(client: genai.Client, audio_bytes: bytes, suffix: str, cache_key: str, status_cb=None):
    if len(audio_bytes) < INLINE_SIZE_LIMIT:
        return types.Part.from_bytes(
            data=audio_bytes,
            mime_type=guess_mime_type(suffix),
        )

    cached = st.session_state.get("gemini_file_cache")
    if cached and cached.get("key") == cache_key:
        try:
            return client.files.get(name=cached["name"])
        except Exception:
            pass

    if status_cb:
        status_cb("Uploading large audio via Files API...")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        uploaded_ref = client.files.upload(file=tmp_path)

        for _ in range(40):
            ref = client.files.get(name=uploaded_ref.name)
            state_name = getattr(getattr(ref, "state", None), "name", getattr(ref, "state", ""))
            if state_name == "ACTIVE":
                st.session_state["gemini_file_cache"] = {"key": cache_key, "name": ref.name}
                return ref
            if state_name == "FAILED":
                raise RuntimeError("Gemini failed to process the audio file.")
            time.sleep(0.5)

        return uploaded_ref
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def generate_fast(
    client: genai.Client,
    primary_model: str,
    model_pool: list[str],
    contents: list,
    enable_fallback: bool = True,
    max_retries: int = 2,
    status_cb=None,
):
    models_to_try = [primary_model]
    if enable_fallback:
        for m in model_pool:
            if m != primary_model:
                models_to_try.append(m)

    last_error = None
    for m_idx, model_name in enumerate(models_to_try):
        for attempt in range(max_retries):
            try:
                if status_cb:
                    status_cb(f"Generating script with {model_name} (try {attempt + 1}/{max_retries})...")
                resp = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(temperature=0.0),
                )
                return resp, model_name
            except APIError as e:
                last_error = e
                # Do not retry 404 on the same model; move immediately to fallback
                if e.code == 404:
                    break
                if e.code in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
                    sleep_s = 1.5 * (attempt + 1)
                    if status_cb:
                        status_cb(f"{model_name} returned {e.code}. Retrying in {sleep_s:.1f}s...")
                    time.sleep(sleep_s)
                    continue
                break
            except Exception as e:
                last_error = e
                break

        if enable_fallback and m_idx < len(models_to_try) - 1 and status_cb:
            status_cb(f"Moving to {models_to_try[m_idx + 1]}...")

    raise last_error


# --- Sidebar ---
with st.sidebar:
    st.header("🔑 API Configuration")
    api_key = st.text_input(
        "Enter Gemini API Key",
        type="password",
        placeholder="AIzaSy...",
    )

    available_models = get_available_models(api_key)
    model_choice = st.selectbox(
        "Primary Gemini Model",
        options=available_models,
        index=0,
    )

    auto_fallback = st.checkbox("Quick fallback if busy/unavailable", value=True)
    compress_large = st.checkbox("Compress files > 5 MB (needs ffmpeg)", value=True)

    output_style = st.selectbox(
        "Transcript Format",
        options=[
            "Clean Paragraphs (Polished Script)",
            "Speaker Diarization (Speaker Labels + Paragraphs)",
            "Verbatim with Timestamps [MM:SS]",
        ],
    )


# --- Main UI ---
st.title("🎙️ Audio-to-Script Dashboard")
st.caption("Fast transcription with PDF & Word (.docx) export")

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("1. Upload Audio")
    doc_title = st.text_input("Document Title", value="Audio Transcript")
    speaker_hint = st.text_input(
        "Speaker / Context Hint (Optional)",
        placeholder="e.g., Dr. T. V. Somanathan - Leadership Workshop",
    )

    uploaded_audio = st.file_uploader(
        "Choose an audio file",
        type=["mp3", "wav", "m4a", "ogg", "flac", "aac", "webm"],
    )

    if uploaded_audio is not None:
        st.audio(uploaded_audio)

    transcribe_btn = st.button("⚡ Generate Script Fast", type="primary", use_container_width=True)

if "transcript" not in st.session_state:
    st.session_state.transcript = ""

if transcribe_btn:
    if not api_key:
        st.error("Please enter your Gemini API Key in the sidebar.")
    elif uploaded_audio is None:
        st.warning("Please upload an audio file first.")
    else:
        start_t = time.perf_counter()
        status_box = st.status("Running...", expanded=True)
        safe_cb = lambda msg: status_box.write(to_safe_text(msg))
        try:
            raw_bytes = uploaded_audio.getvalue()
            raw_hash = hashlib.md5(raw_bytes).hexdigest()

            audio_bytes, suffix = optimize_audio_if_needed(
                raw_bytes,
                uploaded_audio.name,
                compress_large=compress_large,
                status_cb=safe_cb,
            )
            cache_key = f"{raw_hash}_{hashlib.md5(audio_bytes).hexdigest()}"

            client = genai.Client(api_key=api_key.strip())
            audio_part = build_audio_content(
                client=client,
                audio_bytes=audio_bytes,
                suffix=suffix,
                cache_key=cache_key,
                status_cb=safe_cb,
            )

            style_prompts = {
                "Clean Paragraphs (Polished Script)": (
                    "Transcribe this audio accurately into clean paragraphs. "
                    "Remove filler words ('uh', 'um') while keeping exact meaning."
                ),
                "Speaker Diarization (Speaker Labels + Paragraphs)": (
                    "Transcribe this audio with clear speaker labels and clean paragraphs."
                ),
                "Verbatim with Timestamps [MM:SS]": (
                    "Transcribe this audio verbatim with [MM:SS] timestamps at each segment."
                ),
            }

            prompt = f"{style_prompts[output_style]}\n"
            if speaker_hint.strip():
                prompt += f"Context / Speaker: {speaker_hint.strip()}\n"
            prompt += "Return ONLY the transcript text."

            response, used_model = generate_fast(
                client=client,
                primary_model=model_choice,
                model_pool=available_models,
                contents=[audio_part, prompt],
                enable_fallback=auto_fallback,
                status_cb=safe_cb,
            )

            elapsed = time.perf_counter() - start_t
            st.session_state.transcript = response.text or ""
            status_box.update(
                label=f"Completed in {elapsed:.1f}s ({to_safe_text(used_model)})",
                state="complete",
                expanded=False,
            )

        except Exception as e:
            safe_err = to_safe_text(str(e) or repr(e))
            status_box.update(label="Failed", state="error", expanded=True)
            st.error(f"Error: {safe_err}")

with col2:
    st.subheader("2. Review & Download")

    if st.session_state.transcript:
        edited_transcript = st.text_area(
            "Script Text",
            value=st.session_state.transcript,
            height=400,
        )
        st.session_state.transcript = edited_transcript

        word_bytes = create_word_doc(doc_title, speaker_hint, edited_transcript)
        pdf_bytes = create_pdf_doc(doc_title, speaker_hint, edited_transcript)
        safe_filename = doc_title.strip().lower().replace(" ", "_") or "transcript"

        d1, d2 = st.columns(2)
        with d1:
            st.download_button(
                "📄 Download PDF",
                data=pdf_bytes,
                file_name=f"{safe_filename}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        with d2:
            st.download_button(
                "📝 Download Word (.docx)",
                data=word_bytes,
                file_name=f"{safe_filename}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True,
            )
    else:
        st.info("Generated text and PDF/Word download buttons will appear here.")