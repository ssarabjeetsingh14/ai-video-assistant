import json
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarize import summarize, generate_title
from core.extractor import (
    extract_action_items,
    extract_key_decisions,
    extract_questions,
)
from core.rag_engine import build_rag_chain, ask_question

load_dotenv()

st.set_page_config(
    page_title="AI Video Assistant",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ------------------------------- CSS ----------------------------------------
st.markdown(
    r"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    :root {
      --bg:#080d1c; --panel:#101a32; --panel2:#14213d; --line:#293b63;
      --text:#f5f6ff; --muted:#9baaca; --violet:#9b5cff; --pink:#f04de8;
      --blue:#4b9cff; --mint:#31d8b0; --gold:#ffb547;
    }
    html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
    .stApp {
      color:var(--text);
      background:
        radial-gradient(ellipse at 7% 0%, rgba(112,64,255,.22), transparent 30%),
        radial-gradient(ellipse at 96% 8%, rgba(31,133,255,.14), transparent 28%),
        radial-gradient(ellipse at 55% 75%, rgba(174,45,218,.07), transparent 35%),
        var(--bg);
    }
    [data-testid="stHeader"] { background:rgba(8,13,28,.55); }
    [data-testid="stSidebar"], [data-testid="collapsedControl"] { display:none !important; }
    .block-container { max-width:1500px; padding-top:1.7rem; padding-bottom:3rem; }
    h1,h2,h3,h4 { font-family:'Space Grotesk',sans-serif !important; color:var(--text); }
    p, label, .stMarkdown { color:#dce4ff; }
    .topline { display:flex; justify-content:space-between; align-items:center; margin-bottom:25px; }
    .brand { display:flex; align-items:center; gap:12px; }
    .brand-icon {
      width:43px;height:43px;border-radius:14px;display:flex;align-items:center;
      justify-content:center;font-size:24px;color:white;font-weight:700;
      background:linear-gradient(135deg,#7b48ff,#d946ef);
      box-shadow:0 0 28px rgba(151,76,255,.42);
    }
    .brand-name {font:700 20px 'Space Grotesk';letter-spacing:-.5px;color:#fff;}
    .brand-sub {font-size:10px;color:#8f9fc4;letter-spacing:2px;text-transform:uppercase;}
    .live-pill {
      display:inline-flex;align-items:center;gap:8px;border:1px solid #254e61;
      color:#76e8d3;background:rgba(36,205,169,.07);padding:8px 13px;
      border-radius:99px;font-size:11px;font-weight:700;letter-spacing:.7px;
    }
    .live-dot {width:7px;height:7px;border-radius:50%;background:#39e0b5;box-shadow:0 0 10px #39e0b5;}
    .hero {
      position:relative;overflow:hidden;border:1px solid rgba(133,118,255,.35);
      border-radius:26px;padding:34px 36px;margin-bottom:24px;
      background:linear-gradient(112deg,rgba(25,31,66,.96),rgba(11,23,48,.95) 68%,rgba(44,19,65,.78));
      box-shadow:0 18px 70px rgba(0,0,0,.17);
    }
    .hero:after {
      content:'✦';position:absolute;right:6%;top:-65px;font-size:250px;
      color:rgba(158,103,255,.09);transform:rotate(12deg);pointer-events:none;
    }
    .eyebrow {font-size:11px;letter-spacing:2.5px;text-transform:uppercase;color:#c0a3ff;font-weight:700;margin-bottom:13px;}
    .hero-title {font:700 clamp(31px,4vw,52px)/1.07 'Space Grotesk';letter-spacing:-2px;color:#fff;margin-bottom:14px;}
    .gradient-text {background:linear-gradient(90deg,#ad72ff,#f05be7,#7bb7ff);-webkit-background-clip:text;background-clip:text;color:transparent;}
    .hero-copy {font-size:15px;line-height:1.75;color:#aebcda;max-width:680px;}
    .feature-row {display:flex;flex-wrap:wrap;gap:9px;margin-top:22px;}
    .feature-chip {border:1px solid rgba(130,154,207,.19);background:rgba(14,27,54,.7);padding:9px 12px;border-radius:12px;font-size:12px;color:#dce4ff;}
    .feature-chip span {margin-right:7px;}
    .section-head {display:flex;justify-content:space-between;align-items:end;gap:12px;margin:7px 0 16px;}
    .section-title {font:600 22px 'Space Grotesk';color:#f4f5ff;}
    .section-sub {font-size:13px;color:#91a1c4;margin-top:4px;}
    .glass-card {
      border:1px solid rgba(113,143,207,.24);border-radius:19px;padding:22px;
      background:linear-gradient(145deg,rgba(19,31,58,.94),rgba(11,21,42,.91));
      box-shadow:0 12px 32px rgba(0,0,0,.10);margin-bottom:16px;
    }
    .card-kicker {font-size:10px;letter-spacing:1.7px;color:#a4b4d5;font-weight:700;text-transform:uppercase;margin-bottom:11px;}
    .mini-feature {height:100%;min-height:95px;border:1px solid rgba(111,137,193,.2);border-radius:15px;padding:15px;background:rgba(15,27,51,.8);}
    .mini-icon {font-size:19px;margin-bottom:9px;}
    .mini-title {font-weight:700;color:#f2f4ff;font-size:13px;}
    .mini-copy {font-size:11px;color:#91a1c4;line-height:1.5;margin-top:4px;}
    .stTextInput input,.stTextArea textarea,.stSelectbox [data-baseweb="select"] {
      background:#0a142a !important;color:#f3f5ff !important;
      border:1px solid #30456e !important;border-radius:11px !important;
    }
    .stTextInput input:focus,.stTextArea textarea:focus {border-color:#9b5cff !important;box-shadow:0 0 0 1px #9b5cff !important;}
    .stButton button,.stDownloadButton button {
      color:#fff !important;border:1px solid rgba(190,126,255,.7) !important;
      border-radius:12px !important;min-height:43px;font-weight:700 !important;
      background:linear-gradient(100deg,#7948ff,#bd3ee9) !important;
      box-shadow:0 7px 22px rgba(130,63,255,.2);transition:all .2s ease;
    }
    .stButton button:hover,.stDownloadButton button:hover {filter:brightness(1.12);transform:translateY(-1px);}
    .stRadio [role="radiogroup"] {gap:12px;}
    .stRadio label {color:#e7eaff !important;}
    .stTabs [data-baseweb="tab-list"] {gap:5px;border-bottom:1px solid #263759;padding-bottom:4px;}
    .stTabs [data-baseweb="tab"] {padding:12px 14px;border-radius:10px 10px 0 0;color:#a9b7d5;font-weight:600;}
    .stTabs [aria-selected="true"] {color:#e4c4ff !important;border-bottom:2px solid #d34bff !important;background:rgba(147,68,255,.10);}
    [data-testid="stExpander"] {border:1px solid #2b3d62;border-radius:12px;background:rgba(15,27,51,.6);}
    [data-testid="stFileUploader"] {border:1px dashed #5a4b94;border-radius:12px;background:rgba(10,20,42,.55);padding:9px;}
    [data-testid="stChatMessage"] {border:1px solid #2b3d62;background:rgba(18,31,58,.75);border-radius:15px;}
    .metric-card {border:1px solid #2b3c62;border-radius:15px;padding:17px;background:linear-gradient(145deg,#14213e,#0f1a32);}
    .metric-label {font-size:10px;letter-spacing:1.3px;text-transform:uppercase;color:#95a8cd;font-weight:700;}
    .metric-value {font:700 22px 'Space Grotesk';color:#fff;margin-top:8px;}
    .metric-note {font-size:11px;color:#8497bb;margin-top:4px;}
    .insight-card {border:1px solid #2a3c62;border-radius:16px;padding:18px;background:rgba(16,29,55,.78);height:100%;}
    .insight-heading {font-weight:700;color:#f4f5ff;margin-bottom:10px;}
    .empty-state {text-align:center;border:1px dashed #2b3d62;border-radius:18px;padding:35px 15px;color:#96a6c8;background:rgba(16,28,52,.4);}
    .empty-state b {display:block;font:600 18px 'Space Grotesk';color:#e9edff;margin:10px 0;}
    hr {border-color:#253658;}
    footer {visibility:hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------ State ----------------------------------------
for key, default in {
    "result": None,
    "chat_history": [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


def run_pipeline(source: str, language: str) -> dict:
    chunks = process_input(source)
    transcript = transcribe_all(chunks, language=language)
    title = generate_title(transcript)
    summary_text = summarize(transcript)
    action_items = extract_action_items(transcript)
    decisions = extract_key_decisions(transcript)
    questions = extract_questions(transcript)
    rag_chain = build_rag_chain(transcript)
    return {
        "title": title,
        "transcript": transcript,
        "summary": summary_text,
        "action_items": action_items,
        "key_decisions": decisions,
        "open_questions": questions,
        "rag_chain": rag_chain,
    }


def render_items(value):
    if value is None or value == "":
        st.caption("Nothing was extracted for this section.")
    elif isinstance(value, (list, tuple)):
        if not value:
            st.caption("Nothing was extracted for this section.")
        else:
            for item in value:
                st.markdown(f"- {item}")
    else:
        st.markdown(str(value))


def as_export_text(result):
    return (
        f"TITLE\n{result.get('title', '')}\n\n"
        f"SUMMARY\n{result.get('summary', '')}\n\n"
        f"ACTION ITEMS\n{result.get('action_items', '')}\n\n"
        f"KEY DECISIONS\n{result.get('key_decisions', '')}\n\n"
        f"OPEN QUESTIONS\n{result.get('open_questions', '')}\n\n"
        f"TRANSCRIPT\n{result.get('transcript', '')}\n"
    )


# ------------------------------ Header ---------------------------------------
st.markdown(
    """
    <div class="topline">
      <div class="brand">
        <div class="brand-icon">✦</div>
        <div><div class="brand-name">AI Video Assistant</div>
        <div class="brand-sub">Understand more. Rewatch less.</div></div>
      </div>
      <div class="live-pill"><span class="live-dot"></span> AI WORKSPACE</div>
    </div>
    <div class="hero">
      <div class="eyebrow">✦ Your personal media intelligence studio</div>
      <div class="hero-title">Turn hours into <span class="gradient-text">insights.</span></div>
      <div class="hero-copy">Transform videos, lectures, and meetings into clear summaries, searchable transcripts, decisions, and next steps. Then ask questions and get answers from your content.</div>
      <div class="feature-row">
        <div class="feature-chip"><span>◉</span>Smart transcription</div>
        <div class="feature-chip"><span>✧</span>AI summaries</div>
        <div class="feature-chip"><span>✓</span>Actionable takeaways</div>
        <div class="feature-chip"><span>⌕</span>Chat with your content</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------ Input ----------------------------------------
if st.session_state.result is None:
    head_l, head_r = st.columns([0.7, 0.3])
    with head_l:
        st.markdown('<div class="section-title">Create a new analysis</div><div class="section-sub">Start with a link or drop in a recording.</div>', unsafe_allow_html=True)
    with head_r:
        st.markdown('<div style="text-align:right;color:#8396bc;font-size:12px;padding-top:8px;">SUPPORTED: MP3 · WAV · M4A · MP4 · MOV · WEBM</div>', unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="glass-card"><div class="card-kicker">01 / Add your source</div>', unsafe_allow_html=True)
        source_mode = st.radio(
            "Choose your source",
            ["YouTube URL", "Upload media file"],
            horizontal=True,
            label_visibility="collapsed",
        )
        youtube_url = ""
        uploaded_file = None
        if source_mode == "YouTube URL":
            youtube_url = st.text_input(
                "YouTube link",
                placeholder="Paste a YouTube URL here  ·  https://youtube.com/watch?v=...",
                label_visibility="collapsed",
            )
        else:
            uploaded_file = st.file_uploader(
                "Drop your audio or video here, or click to browse",
                type=["mp3", "wav", "m4a", "mp4", "mpeg", "mpga", "webm", "ogg", "flac", "mov"],
            )
        c1, c2 = st.columns([0.38, 0.62], vertical_alignment="bottom")
        with c1:
            language = st.selectbox(
                "Transcription language",
                ["english", "hinglish"],
                format_func=lambda x: "English" if x == "english" else "Hinglish",
            )
        with c2:
            analyze = st.button("✦  Start Analysis  →", use_container_width=True, type="primary")
        st.markdown('</div>', unsafe_allow_html=True)

    feature_cols = st.columns(4, gap="medium")
    features = [
        ("◉", "Transcribe", "Convert spoken words into readable text."),
        ("✧", "Summarize", "Get the core ideas at a glance."),
        ("✓", "Extract actions", "Find tasks, decisions, and questions."),
        ("⌕", "Ask anything", "Chat with your video using RAG."),
    ]
    for col, (icon, title, copy) in zip(feature_cols, features):
        with col:
            st.markdown(
                f'<div class="mini-feature"><div class="mini-icon">{icon}</div>'
                f'<div class="mini-title">{title}</div><div class="mini-copy">{copy}</div></div>',
                unsafe_allow_html=True,
            )

    if analyze:
        source = None
        temp_path = None
        if source_mode == "YouTube URL":
            if not youtube_url.strip():
                st.warning("Paste a YouTube URL to get started.")
            else:
                source = youtube_url.strip()
        elif uploaded_file is None:
            st.warning("Upload an audio or video file to get started.")
        else:
            suffix = Path(uploaded_file.name).suffix or ".mp4"
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
                    temp.write(uploaded_file.getbuffer())
                    temp_path = temp.name
                source = temp_path
            except Exception as exc:
                st.error(f"Could not prepare uploaded media: {exc}")

        if source:
            # A persistent visual step tracker. Each step changes to complete only
            # after its corresponding pipeline function returns successfully.
            st.markdown(
                """
                <div class="glass-card">
                  <div class="card-kicker">LIVE PIPELINE</div>
                  <div class="section-title">Building your media insights</div>
                  <div class="section-sub">Progress updates as each stage finishes.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            step_names = [
                "Prepare media",
                "Transcribe audio",
                "Generate title & summary",
                "Extract action items",
                "Extract key decisions",
                "Extract open questions",
                "Build RAG knowledge base",
            ]
            step_details = [
                "Downloading or preparing audio and splitting it into chunks.",
                "Converting speech into a transcript.",
                "Creating a title and concise summary.",
                "Finding tasks and follow-ups.",
                "Identifying decisions made in the content.",
                "Finding unresolved questions.",
                "Indexing transcript content for chat.",
            ]
            progress = st.progress(0, text="Getting ready…")
            step_placeholders = []
            for i, (step_name, detail) in enumerate(zip(step_names, step_details)):
                ph = st.empty()
                step_placeholders.append(ph)
                ph.markdown(
                    f'<div class="glass-card" style="padding:14px 18px;opacity:.72;">'
                    f'<div style="display:flex;gap:14px;align-items:center;">'
                    f'<div style="width:34px;height:34px;border-radius:50%;display:flex;align-items:center;justify-content:center;'
                    f'background:#172541;border:1px solid #30456e;color:#91a5ca;font-weight:700;">{i+1:02}</div>'
                    f'<div style="flex:1"><div style="font-weight:700;color:#e6ebff">{step_name}</div>'
                    f'<div style="font-size:12px;color:#91a1c4;margin-top:3px">{detail}</div></div>'
                    f'<div style="color:#8293b7;font-size:12px">Waiting</div></div></div>',
                    unsafe_allow_html=True,
                )

            def set_step(index, state):
                symbols = {"running": "◌", "complete": "✓", "error": "!"}
                labels = {"running": "In progress", "complete": "Completed", "error": "Failed"}
                colors = {"running": "#bda2ff", "complete": "#42dfb4", "error": "#ff7a90"}
                backgrounds = {
                    "running": "rgba(145,92,255,.12)",
                    "complete": "rgba(49,216,176,.08)",
                    "error": "rgba(255,90,120,.09)",
                }
                border_colors = {
                    "running": "#7955cf",
                    "complete": "#267f70",
                    "error": "#a9465b",
                }
                step_placeholders[index].markdown(
                    f'<div class="glass-card" style="padding:14px 18px;margin-bottom:10px;'
                    f'border-color:{border_colors[state]};background:{backgrounds[state]};">'
                    f'<div style="display:flex;gap:14px;align-items:center;">'
                    f'<div style="width:34px;height:34px;flex-shrink:0;border-radius:50%;display:flex;align-items:center;'
                    f'justify-content:center;background:{backgrounds[state]};border:1px solid {border_colors[state]};'
                    f'color:{colors[state]};font-weight:700;font-size:17px;">{symbols[state]}</div>'
                    f'<div style="flex:1"><div style="font-weight:700;color:#f2f4ff">{step_names[index]}</div>'
                    f'<div style="font-size:12px;color:#a0afd0;margin-top:3px">{step_details[index]}</div></div>'
                    f'<div style="color:{colors[state]};font-size:12px;font-weight:700">{labels[state]}</div>'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )

            try:
                # Run stages individually so the UI can show genuine progress.
                set_step(0, "running")
                progress.progress(3, text="Step 1 of 7 · Preparing media")
                chunks = process_input(source)
                set_step(0, "complete")
                progress.progress(15, text="Step 1 of 7 · Media prepared")

                set_step(1, "running")
                progress.progress(18, text="Step 2 of 7 · Transcribing audio")
                transcript = transcribe_all(chunks, language=language)
                set_step(1, "complete")
                progress.progress(40, text="Step 2 of 7 · Transcription complete")

                set_step(2, "running")
                progress.progress(43, text="Step 3 of 7 · Generating title and summary")
                title = generate_title(transcript)
                summary_text = summarize(transcript)
                set_step(2, "complete")
                progress.progress(58, text="Step 3 of 7 · Summary complete")

                set_step(3, "running")
                progress.progress(61, text="Step 4 of 7 · Extracting action items")
                action_items = extract_action_items(transcript)
                set_step(3, "complete")
                progress.progress(69, text="Step 4 of 7 · Action items extracted")

                set_step(4, "running")
                progress.progress(72, text="Step 5 of 7 · Extracting key decisions")
                decisions = extract_key_decisions(transcript)
                set_step(4, "complete")
                progress.progress(80, text="Step 5 of 7 · Decisions extracted")

                set_step(5, "running")
                progress.progress(83, text="Step 6 of 7 · Extracting open questions")
                questions = extract_questions(transcript)
                set_step(5, "complete")
                progress.progress(90, text="Step 6 of 7 · Questions extracted")

                set_step(6, "running")
                progress.progress(93, text="Step 7 of 7 · Building searchable knowledge base")
                rag_chain = build_rag_chain(transcript)
                set_step(6, "complete")
                progress.progress(100, text="All 7 steps completed")

                result = {
                    "title": title,
                    "transcript": transcript,
                    "summary": summary_text,
                    "action_items": action_items,
                    "key_decisions": decisions,
                    "open_questions": questions,
                    "rag_chain": rag_chain,
                }
                st.session_state.result = result
                st.session_state.chat_history = []
                st.success("Analysis complete! Opening your results…")
                st.rerun()
            except Exception as exc:
                # Mark the currently active stage as failed.
                for idx, ph in enumerate(step_placeholders):
                    if "In progress" in str(ph):
                        set_step(idx, "error")
                        break
                st.error("Analysis failed. The step tracker shows the progress reached. Check the terminal traceback, API keys, and FFmpeg/FFprobe configuration.")
                st.exception(exc)
            finally:
                if temp_path:
                    try:
                        Path(temp_path).unlink(missing_ok=True)
                    except Exception:
                        pass

else:
    result = st.session_state.result
    title = result.get("title") or "Untitled media analysis"

    result_head_l, result_head_r = st.columns([0.78, 0.22], vertical_alignment="center")
    with result_head_l:
        st.markdown(f'<div class="section-title">{title}</div><div class="section-sub">Your media has been analyzed. Explore the insights or ask a follow-up question.</div>', unsafe_allow_html=True)
    with result_head_r:
        if st.button("＋ New analysis", use_container_width=True):
            st.session_state.result = None
            st.session_state.chat_history = []
            st.rerun()

    m1, m2, m3, m4 = st.columns(4, gap="medium")
    metric_data = [
        (m1, "01 / SUMMARY", "Ready", "Main ideas in one place"),
        (m2, "02 / ACTIONS", "Extracted", "Tasks and follow-ups"),
        (m3, "03 / DECISIONS", "Extracted", "Important outcomes"),
        (m4, "04 / KNOWLEDGE", "Searchable", "Ask questions with RAG"),
    ]
    for col, label, value, note in metric_data:
        with col:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">{label}</div>'
                f'<div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>',
                unsafe_allow_html=True,
            )

    st.write("")
    overview_tab, transcript_tab, actions_tab, decisions_tab, questions_tab, chat_tab, export_tab = st.tabs(
        ["✦ Overview", "▤ Transcript", "✓ Action Items", "◈ Key Decisions", "? Open Questions", "⌕ Chat with Video", "↓ Export"]
    )

    with overview_tab:
        left, right = st.columns([0.68, 0.32], gap="large")
        with left:
            st.markdown('<div class="section-title">Executive summary</div><div class="section-sub">The important ideas, without the noise.</div>', unsafe_allow_html=True)
            with st.container(border=True):
                st.markdown(result.get("summary") or "No summary was generated.")
        with right:
            st.markdown('<div class="section-title">At a glance</div><div class="section-sub">Your content, organized.</div>', unsafe_allow_html=True)
            with st.container(border=True):
                st.markdown("**✦ Generated title**")
                st.write(title)
                st.markdown("---")
                st.markdown("**▤ Transcript length**")
                transcript_text = str(result.get("transcript") or "")
                st.write(f"{len(transcript_text.split()):,} words")
                st.markdown("---")
                st.markdown("**⌕ Ask your content**")
                st.caption("Open Chat with Video to ask questions about the transcript.")
        a, b = st.columns(2, gap="large")
        with a:
            st.markdown('<div class="insight-card"><div class="insight-heading">✓ Action items</div>', unsafe_allow_html=True)
            render_items(result.get("action_items"))
            st.markdown('</div>', unsafe_allow_html=True)
        with b:
            st.markdown('<div class="insight-card"><div class="insight-heading">◈ Key decisions</div>', unsafe_allow_html=True)
            render_items(result.get("key_decisions"))
            st.markdown('</div>', unsafe_allow_html=True)

    with transcript_tab:
        st.markdown('<div class="section-title">Transcript explorer</div><div class="section-sub">Search the transcription or inspect the full text.</div>', unsafe_allow_html=True)
        transcript = str(result.get("transcript") or "")
        term = st.text_input("Search transcript", placeholder="Type a word or phrase…")
        if term.strip():
            pos = transcript.lower().find(term.lower().strip())
            if pos >= 0:
                st.success(f"Match found around character {pos:,}.")
                start = max(0, pos - 350)
                end = min(len(transcript), pos + len(term) + 550)
                st.markdown(f"> …{transcript[start:end]}…")
            else:
                st.info("No matching text was found.")
        st.text_area("Full transcript", value=transcript, height=430, label_visibility="collapsed")

    with actions_tab:
        st.markdown('<div class="section-title">Action items</div><div class="section-sub">Tasks and follow-ups identified in the content.</div>', unsafe_allow_html=True)
        with st.container(border=True):
            render_items(result.get("action_items"))

    with decisions_tab:
        st.markdown('<div class="section-title">Key decisions</div><div class="section-sub">The important conclusions and choices from the discussion.</div>', unsafe_allow_html=True)
        with st.container(border=True):
            render_items(result.get("key_decisions"))

    with questions_tab:
        st.markdown('<div class="section-title">Open questions</div><div class="section-sub">Unresolved questions surfaced from the transcript.</div>', unsafe_allow_html=True)
        with st.container(border=True):
            render_items(result.get("open_questions"))

    with chat_tab:
        st.markdown('<div class="section-title">Chat with your video</div><div class="section-sub">Ask about key ideas, decisions, examples, or details in the transcript.</div>', unsafe_allow_html=True)
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        prompt = st.chat_input("Ask a question about your video or meeting…")
        if prompt:
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Searching your content…"):
                    try:
                        answer = str(ask_question(result["rag_chain"], prompt))
                    except Exception as exc:
                        answer = f"I couldn't answer that question: {exc}"
                    st.markdown(answer)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})

    with export_tab:
        st.markdown('<div class="section-title">Export your insights</div><div class="section-sub">Save the analysis to read, share, or archive later.</div>', unsafe_allow_html=True)
        export_data = {
            "title": result.get("title"),
            "summary": result.get("summary"),
            "action_items": result.get("action_items"),
            "key_decisions": result.get("key_decisions"),
            "open_questions": result.get("open_questions"),
            "transcript": result.get("transcript"),
        }
        e1, e2 = st.columns(2, gap="large")
        with e1:
            st.markdown('<div class="glass-card"><div class="card-kicker">Structured export</div><h3>JSON file</h3><p>All generated insights in a structured format.</p></div>', unsafe_allow_html=True)
            st.download_button(
                "↓ Download JSON",
                data=json.dumps(export_data, ensure_ascii=False, indent=2, default=str),
                file_name="ai_video_analysis.json",
                mime="application/json",
                use_container_width=True,
            )
        with e2:
            st.markdown('<div class="glass-card"><div class="card-kicker">Readable export</div><h3>Text file</h3><p>Simple, portable notes including the transcript.</p></div>', unsafe_allow_html=True)
            st.download_button(
                "↓ Download TXT",
                data=as_export_text(result),
                file_name="ai_video_analysis.txt",
                mime="text/plain",
                use_container_width=True,
            )

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    '<div style="display:flex;justify-content:space-between;gap:10px;flex-wrap:wrap;color:#7284aa;font-size:11px;">'
    '<span>✦ AI VIDEO ASSISTANT</span><span>TRANSCRIBE · UNDERSTAND · TAKE ACTION</span></div>',
    unsafe_allow_html=True,
)
