"""
app.py — AI Resume & Cover Letter Generator  (v2)
==================================================
Groq API (free tier) — https://console.groq.com

What's new in v2:
  · Dark / Light theme chooser (in-app toggle)
  · 3 PDF templates: Modern, Classic, Minimal
  · Job Description ATS Analyzer with match score & gap keywords
  · AI Resume Improvement Tips (prioritised 🔴🟡🟢)
  · LinkedIn Bio Generator

Capstone Project: Using Generative AI with Streamlit
Academic sections preserved in the footer expander.
"""

import os
import streamlit as st
from dotenv import load_dotenv
from utils import (
    generate_prompt, call_llm, format_output,
    evaluate_resume, generate_pdf,
    analyze_jd, get_tips, gen_linkedin,
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
load_dotenv()

st.set_page_config(
    page_title="AI Resume Generator",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# THEME — toggled via sidebar, stored in session state
# ─────────────────────────────────────────────────────────────────────────────
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

DARK = st.session_state.dark_mode

# All colours defined once per theme — used in both CSS and f-strings
if DARK:
    T = {
        "bg":           "#0d0d1a",
        "bg_card":      "rgba(255,255,255,0.035)",
        "bg_input":     "rgba(255,255,255,0.055)",
        "border":       "rgba(255,255,255,0.08)",
        "border_input": "rgba(255,255,255,0.13)",
        "text":         "#eef2f7",
        "text_muted":   "#8896aa",
        "text_label":   "#7a8899",
        "accent":       "#5b8ef0",
        "accent2":      "#7c5bf0",
        "sidebar_bg":   "#08080f",
        "sidebar_txt":  "#7a8899",
        "sidebar_head": "#a78bfa",
        "score_bg":     "rgba(91,142,240,0.08)",
        "score_bdr":    "rgba(91,142,240,0.22)",
        "hero_grad":    "linear-gradient(135deg,#1a1a3e 0%,#0d0d1a 100%)",
        "hero_shine":   "rgba(91,142,240,0.15)",
        "tag_vb_bg":    "rgba(124,91,240,0.15)",
        "tag_vb_cl":    "#b39dfa",
        "tag_vb_bd":    "rgba(124,91,240,0.3)",
        "tag_sk_bg":    "rgba(16,185,129,0.12)",
        "tag_sk_cl":    "#5eead4",
        "tag_sk_bd":    "rgba(20,184,166,0.3)",
        "tag_gp_bg":    "rgba(239,68,68,0.12)",
        "tag_gp_cl":    "#fca5a5",
        "tag_gp_bd":    "rgba(239,68,68,0.3)",
        "tip_r_bg":     "rgba(239,68,68,0.1)",
        "tip_r_bd":     "#ef4444",
        "tip_y_bg":     "rgba(245,158,11,0.1)",
        "tip_y_bd":     "#f59e0b",
        "tip_g_bg":     "rgba(20,184,166,0.1)",
        "tip_g_bd":     "#14b8a6",
        "res_bg":       "rgba(255,255,255,0.025)",
        "progress_bg":  "rgba(255,255,255,0.06)",
    }
else:
    T = {
        "bg":           "#f4f6fb",
        "bg_card":      "#ffffff",
        "bg_input":     "#ffffff",
        "border":       "#dde3ee",
        "border_input": "#c8d0e0",
        "text":         "#1a202c",
        "text_muted":   "#5a6577",
        "text_label":   "#6b7a90",
        "accent":       "#3b5fc0",
        "accent2":      "#5b3db0",
        "sidebar_bg":   "#1a1a2e",
        "sidebar_txt":  "#7a8899",
        "sidebar_head": "#a78bfa",
        "score_bg":     "rgba(59,95,192,0.06)",
        "score_bdr":    "rgba(59,95,192,0.2)",
        "hero_grad":    "linear-gradient(135deg,#3b5fc0 0%,#5b3db0 60%,#3b5fc0 100%)",
        "hero_shine":   "rgba(255,255,255,0.12)",
        "tag_vb_bg":    "rgba(91,61,176,0.08)",
        "tag_vb_cl":    "#5b3db0",
        "tag_vb_bd":    "rgba(91,61,176,0.22)",
        "tag_sk_bg":    "rgba(13,148,136,0.08)",
        "tag_sk_cl":    "#0d9488",
        "tag_sk_bd":    "rgba(13,148,136,0.22)",
        "tag_gp_bg":    "rgba(220,38,38,0.07)",
        "tag_gp_cl":    "#dc2626",
        "tag_gp_bd":    "rgba(220,38,38,0.2)",
        "tip_r_bg":     "rgba(220,38,38,0.06)",
        "tip_r_bd":     "#dc2626",
        "tip_y_bg":     "rgba(217,119,6,0.07)",
        "tip_y_bd":     "#d97706",
        "tip_g_bg":     "rgba(13,148,136,0.07)",
        "tip_g_bd":     "#0d9488",
        "res_bg":       "#f8faff",
        "progress_bg":  "#e8edf5",
    }


# ─────────────────────────────────────────────────────────────────────────────
# CSS — injected after theme tokens are set
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
/* ── Font ── */
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');
html, body, [class*="css"] {{ font-family: 'DM Sans', sans-serif !important; }}

/* ── App background ── */
.stApp {{ background: {T['bg']} !important; color: {T['text']} !important; }}
.main .block-container {{ padding-top: 1rem; padding-bottom: 2rem; max-width: 1240px; }}

/* ── Hero ── */
.hero {{
    background: {T['hero_grad']};
    border-radius: 18px;
    padding: 3rem 2.5rem 2.4rem;
    text-align: center;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 16px 56px rgba(59,95,192,0.25);
}}
.hero::before {{
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(ellipse at 60% 0%, {T['hero_shine']} 0%, transparent 65%);
    pointer-events: none;
}}
.hero h1 {{
    color: #fff; font-size: 2.5rem; font-weight: 700;
    margin: 0; letter-spacing: -0.5px;
    text-shadow: 0 2px 16px rgba(0,0,0,0.25);
}}
.hero p {{ color: rgba(255,255,255,0.82); font-size: 1.02rem; margin-top: 0.6rem; }}
.badge-row {{ display:flex; justify-content:center; flex-wrap:wrap; gap:0.5rem; margin-top:1.1rem; }}
.badge {{
    background: rgba(255,255,255,0.13);
    border: 1px solid rgba(255,255,255,0.22);
    border-radius: 20px; padding: 0.25rem 0.85rem;
    color: #fff; font-size: 0.76rem; font-weight: 500;
    letter-spacing: 0.2px;
}}

/* ── Section label ── */
.sl {{
    color: {T['accent']}; font-size: 0.68rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 2px; margin-bottom: 0.55rem;
}}

/* ── Score card ── */
.score-card {{
    background: {T['score_bg']}; border: 1px solid {T['score_bdr']};
    border-radius: 14px; padding: 1.5rem; margin-bottom: 1rem;
}}
.score-num {{
    font-size: 3.8rem; font-weight: 700; line-height: 1;
    background: linear-gradient(135deg, {T['accent']}, {T['accent2']});
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
}}
.score-lbl {{ color: {T['text_muted']}; font-size: 0.8rem; margin-top: 0.2rem; }}
.metric-row {{ display:flex; gap:0.75rem; flex-wrap:wrap; margin-top:0.9rem; }}
.metric-pill {{
    background: {T['bg_card']}; border: 1px solid {T['border']};
    border-radius: 8px; padding: 0.4rem 0.9rem;
    color: {T['text']}; font-size: 0.8rem;
}}
.metric-pill b {{ color: {T['accent']}; }}

/* ── Keyword tags ── */
.tag-v {{ display:inline-block; background:{T['tag_vb_bg']}; border:1px solid {T['tag_vb_bd']};
          color:{T['tag_vb_cl']}; border-radius:6px; padding:.16rem .6rem;
          font-size:.76rem; margin:2px; font-weight:500; }}
.tag-s {{ display:inline-block; background:{T['tag_sk_bg']}; border:1px solid {T['tag_sk_bd']};
          color:{T['tag_sk_cl']}; border-radius:6px; padding:.16rem .6rem;
          font-size:.76rem; margin:2px; font-weight:500; }}
.tag-g {{ display:inline-block; background:{T['tag_gp_bg']}; border:1px solid {T['tag_gp_bd']};
          color:{T['tag_gp_cl']}; border-radius:6px; padding:.16rem .6rem;
          font-size:.76rem; margin:2px; font-weight:500; }}

/* ── Tip rows ── */
.tip-r {{ background:{T['tip_r_bg']}; border-left:3px solid {T['tip_r_bd']};
          border-radius:0 8px 8px 0; padding:.5rem .9rem; margin:.28rem 0;
          color:{T['text']}; font-size:.88rem; line-height:1.5; }}
.tip-y {{ background:{T['tip_y_bg']}; border-left:3px solid {T['tip_y_bd']};
          border-radius:0 8px 8px 0; padding:.5rem .9rem; margin:.28rem 0;
          color:{T['text']}; font-size:.88rem; line-height:1.5; }}
.tip-g {{ background:{T['tip_g_bg']}; border-left:3px solid {T['tip_g_bd']};
          border-radius:0 8px 8px 0; padding:.5rem .9rem; margin:.28rem 0;
          color:{T['text']}; font-size:.88rem; line-height:1.5; }}

/* ── Resume pre-text ── */
.resume-text {{
    font-family: 'DM Mono', monospace; font-size: 0.83rem; line-height: 1.78;
    color: {T['text']}; white-space: pre-wrap; background: {T['res_bg']};
    border: 1px solid {T['border']}; border-radius: 10px; padding: 1.25rem;
}}

/* ── ATS bar wrapper ── */
.ats-big {{
    font-size: 3rem; font-weight: 700; line-height: 1;
    background: linear-gradient(135deg, #10b981, {T['accent']});
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
}}

/* ── LinkedIn text block ── */
.linkedin-block {{
    background: {T['bg_card']}; border: 1px solid {T['border']};
    border-radius: 12px; padding: 1.3rem 1.5rem;
    color: {T['text']}; font-size: 0.93rem; line-height: 1.75;
    white-space: pre-wrap;
}}

/* ── Inputs ── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {{
    background: {T['bg_input']} !important; border: 1px solid {T['border_input']} !important;
    border-radius: 8px !important; color: {T['text']} !important;
    font-family: 'DM Sans', sans-serif !important;
}}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {{
    border-color: {T['accent']} !important;
    box-shadow: 0 0 0 2px rgba(91,142,240,.15) !important;
}}
.stSelectbox > div > div {{
    background: {T['bg_input']} !important; border: 1px solid {T['border_input']} !important;
    border-radius: 8px !important; color: {T['text']} !important;
}}

/* ── Buttons ── */
.stButton > button {{
    border-radius: 10px !important; font-weight: 600 !important;
    font-family: 'DM Sans', sans-serif !important; transition: all .2s !important;
}}
.stButton > button[kind="primary"] {{
    background: linear-gradient(135deg, {T['accent']}, {T['accent2']}) !important;
    border: none !important; color: #fff !important; font-size: 1rem !important;
}}
.stButton > button[kind="primary"]:hover {{
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(91,142,240,.38) !important;
}}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background: {T['sidebar_bg']} !important;
    border-right: 1px solid {T['border']} !important;
}}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown li,
[data-testid="stSidebar"] .stCaption {{ color: {T['sidebar_txt']} !important; }}
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4 {{ color: {T['sidebar_head']} !important; }}

/* ── Labels ── */
label {{ color: {T['text_label']} !important; font-size: .86rem !important; font-weight: 500 !important; }}

/* ── Tabs ── */
.stTabs [data-baseweb="tab"] {{ color: {T['text_muted']} !important; font-size: .9rem !important; }}
.stTabs [aria-selected="true"] {{ color: {T['accent']} !important; font-weight: 600 !important; }}

/* ── Misc ── */
.stAlert {{ border-radius: 10px !important; }}
hr {{ border-color: {T['border']} !important; }}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🚀 AI Resume Generator")

    # ── Theme toggle ──
    toggle_label = "☀️ Switch to Light Mode" if DARK else "🌙 Switch to Dark Mode"
    if st.button(toggle_label, use_container_width=True):
        st.session_state.dark_mode = not DARK
        st.rerun()

    st.markdown("---")
    st.markdown("### ⚙️ Settings")

    model_choice = st.selectbox(
        "Groq Model",
        ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
        index=0,
        help="70b = best quality · 8b-instant = fastest",
    )

    pdf_template = st.selectbox(
        "PDF Template",
        ["Modern", "Classic", "Minimal"],
        index=0,
        help="Visual style for your downloaded PDF resume",
    )

    st.markdown("---")
    st.markdown("### 🔬 AI/ML Stack")
    st.markdown("""
- **LLM** Groq LLaMA 3.3 70B
- **Task** Natural Language Generation
- **Eval** Rule-based NLP scoring
- **PDF** fpdf2 · 3 templates
- **UI** Streamlit + DM Sans
""")
    st.markdown("---")
    st.markdown("### 📖 References")
    st.markdown("""
- Vaswani et al. (2017) *Attention Is All You Need*
- Brown et al. (2020) *Language Models are Few-Shot Learners*
- Meta AI (2024) *LLaMA 3*
- Groq API Documentation (2024)
""")
    st.markdown("---")
    st.caption("Built with Python · Streamlit · Groq LLaMA")


# ─────────────────────────────────────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🚀 AI Resume & Career Generator</h1>
  <p>ATS-optimized resumes · Job fit analysis · LinkedIn bio · AI tips — all in one place</p>
  <div class="badge-row">
    <span class="badge">🧠 Groq LLaMA 3.3</span>
    <span class="badge">⚡ ATS Optimizer</span>
    <span class="badge">📄 3 PDF Templates</span>
    <span class="badge">🎯 JD Analyzer</span>
    <span class="badge">💼 LinkedIn Bio</span>
    <span class="badge">💡 AI Tips</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# API KEY
# ─────────────────────────────────────────────────────────────────────────────
api_key = os.getenv("GROQ_API_KEY", "")
if not api_key:
    api_key = st.text_input(
        "🔑 Groq API Key (free — no billing required)",
        type="password",
        placeholder="gsk_...",
        help="Get your free key at https://console.groq.com",
    )
    if not api_key:
        st.warning("Enter your Groq API key above to continue.", icon="🔑")
        st.info(
            "**Get a free key in 60 seconds:**\n"
            "1. Visit [console.groq.com](https://console.groq.com)\n"
            "2. Sign up → API Keys → Create API Key\n"
            "3. Paste it above",
            icon="ℹ️",
        )
        st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# INPUT FORM
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<p class="sl">📋 Candidate Profile</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

with col1:
    with st.container(border=True):
        st.markdown('<p class="sl">👤 Personal Information</p>', unsafe_allow_html=True)
        full_name = st.text_input("Full Name *",     placeholder="Alex Johnson")
        email     = st.text_input("Email Address *", placeholder="alex@email.com")
        phone     = st.text_input("Phone Number *",  placeholder="+1-555-0100")
        linkedin  = st.text_input("LinkedIn URL",    placeholder="linkedin.com/in/alexjohnson")

    with st.container(border=True):
        st.markdown('<p class="sl">🎓 Education</p>', unsafe_allow_html=True)
        education = st.text_area(
            "Education Details *",
            placeholder="B.Tech Computer Science, XYZ University, 2020–2024, GPA: 8.5",
            height=95,
        )

with col2:
    with st.container(border=True):
        st.markdown('<p class="sl">💼 Role & Skills</p>', unsafe_allow_html=True)
        job_role = st.text_input(
            "Target Job Role *",
            placeholder="Machine Learning Engineer",
        )
        skills = st.text_input(
            "Technical Skills * (comma-separated)",
            placeholder="Python, TensorFlow, PyTorch, SQL, Docker",
        )

    with st.container(border=True):
        st.markdown('<p class="sl">🚀 Experience & Projects</p>', unsafe_allow_html=True)
        projects = st.text_area(
            "Projects *",
            placeholder="Sentiment Analyzer — LSTM, 91% accuracy, PyTorch\nPrice Predictor — RMSE reduced 23%",
            height=100,
        )
        experience = st.text_area(
            "Work Experience (leave blank if none)",
            placeholder="Intern @ ABC Corp, Jun–Aug 2023 — Built REST APIs with FastAPI",
            height=100,
        )

# Optional JD input
with st.expander("🎯 Paste Job Description for ATS Analysis (optional)", expanded=False):
    st.caption("Paste the full job description below. The AI will score your resume match, find keyword gaps, and give tailoring suggestions.")
    job_description = st.text_area(
        "Job Description",
        placeholder="We are looking for a Machine Learning Engineer with experience in Python, TensorFlow...",
        height=180,
        label_visibility="collapsed",
    )


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────────────────────────────────────
def validate() -> list[str]:
    errors = []
    for field, val in {
        "Full Name": full_name, "Email": email, "Phone": phone,
        "Education": education, "Skills": skills,
        "Projects": projects,   "Target Job Role": job_role,
    }.items():
        if not val.strip():
            errors.append(f"**{field}** is required.")
    return errors


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
DEFAULTS = {
    "generated":       False,
    "output_sections": {},
    "evaluation":      {},
    "ats_result":      None,
    "tips":            [],
    "linkedin_bio":    "",
    # stored inputs for PDF download after regenerate
    "stored_name": "", "stored_email": "", "stored_phone": "",
    "stored_linkedin": "", "stored_role": "", "stored_skills": "",
    "stored_exp": "", "stored_proj": "",
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────────────────────────────────────────
# GENERATE BUTTON
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
_, btn_col, _ = st.columns([1, 2.2, 1])
with btn_col:
    gen_btn = st.button(
        "✨ Generate Resume & Career Kit",
        type="primary",
        use_container_width=True,
    )

if gen_btn:
    errs = validate()
    if errs:
        st.error("❌ Please fix the following:\n" + "\n".join(f"  - {e}" for e in errs))
    else:
        # Show a live progress bar so users see something happening
        progress = st.progress(0, text="Building your resume…")
        try:
            # Step 1 — Core generation
            progress.progress(10, text="🤖 Writing your resume & cover letter…")
            prompt   = generate_prompt(full_name, email, phone, linkedin,
                                       education, skills, projects, experience, job_role)
            raw      = call_llm(api_key=api_key, user_prompt=prompt, model=model_choice)
            sections = format_output(raw)
            res_text = sections.get("resume", "") + " " + sections.get("summary", "")
            evaluation = evaluate_resume(res_text, skills, job_role)

            st.session_state.update({
                "generated":       True,
                "output_sections": sections,
                "evaluation":      evaluation,
                "stored_name":     full_name,
                "stored_email":    email,
                "stored_phone":    phone,
                "stored_linkedin": linkedin,
                "stored_role":     job_role,
                "stored_skills":   skills,
                "stored_exp":      experience,
                "stored_proj":     projects,
            })
            progress.progress(40, text="📊 Scoring resume strength…")

            # Step 2 — ATS analysis (only if JD provided)
            if job_description.strip():
                progress.progress(55, text="🎯 Analysing job description match…")
                st.session_state.ats_result = analyze_jd(
                    api_key, job_description.strip(), res_text, model_choice
                )
            else:
                st.session_state.ats_result = None

            # Step 3 — Tips
            progress.progress(70, text="💡 Generating improvement tips…")
            st.session_state.tips = get_tips(api_key, res_text, job_role, model_choice)

            # Step 4 — LinkedIn bio
            progress.progress(85, text="💼 Crafting LinkedIn bio…")
            st.session_state.linkedin_bio = gen_linkedin(
                api_key, full_name, job_role, skills, experience, projects, model_choice
            )

            progress.progress(100, text="✅ Done!")
            progress.empty()
            st.success("✅ Your career kit is ready — explore the tabs below.")

        except Exception as e:
            progress.empty()
            err = str(e).lower()
            if "api_key" in err or "authentication" in err or "invalid_api_key" in err:
                st.error("🔑 **Invalid API Key** — verify at console.groq.com")
            elif "rate_limit" in err or "rate limit" in err:
                st.error("⏳ **Rate Limit** — wait a moment then try again.")
            elif "model" in err:
                st.error(f"🤖 **Model error** — try switching to `llama-3.1-8b-instant`.\n{e}")
            else:
                st.error(f"❌ **Error:** {e}")


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT — shown after successful generation
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.generated and st.session_state.output_sections:

    sections     = st.session_state.output_sections
    evaluation   = st.session_state.evaluation
    ats          = st.session_state.ats_result
    tips         = st.session_state.tips
    linkedin_bio = st.session_state.linkedin_bio

    name_dl = st.session_state.stored_name.replace(" ", "_")
    role_dl = st.session_state.stored_role.replace(" ", "_")

    st.markdown("---")

    # ── 6 output tabs ─────────────────────────────────────────────────────────
    tabs = st.tabs([
        "📊 Score",
        "📄 Resume",
        "✉️ Cover Letter",
        "🎯 ATS Analyzer",
        "💡 Tips",
        "💼 LinkedIn Bio",
    ])

    # ── TAB 0 : Score ─────────────────────────────────────────────────────────
    with tabs[0]:
        sc = evaluation["total_score"]
        gr = evaluation["grade"]
        cl = evaluation["color"]

        st.markdown(f"""
        <div class="score-card">
          <div style="display:flex;align-items:center;gap:1.8rem;flex-wrap:wrap;">
            <div>
              <div class="score-num">{sc}</div>
              <div class="score-lbl">Resume Strength / 100</div>
            </div>
            <div style="font-size:1.4rem;font-weight:700;color:{cl};">{gr}</div>
          </div>
          <div class="metric-row">
            <div class="metric-pill">⚡ Verbs <b>{evaluation['verb_score']}/30</b></div>
            <div class="metric-pill">🎯 Keywords <b>{evaluation['keyword_score']}/40</b></div>
            <div class="metric-pill">📝 Length <b>{evaluation['length_score']}/30</b></div>
            <div class="metric-pill">🔤 Words <b>{evaluation['word_count']}</b></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        tc1, tc2 = st.columns(2)
        with tc1:
            st.markdown("**⚡ Action Verbs Found**")
            if evaluation["found_verbs"]:
                st.markdown(
                    "".join(f'<span class="tag-v">{v}</span>' for v in evaluation["found_verbs"]),
                    unsafe_allow_html=True,
                )
            else:
                st.caption("None detected — add more action verbs.")
        with tc2:
            st.markdown("**✅ Skills Found in Resume**")
            if evaluation["skills_matched"]:
                st.markdown(
                    "".join(f'<span class="tag-s">{s}</span>' for s in evaluation["skills_matched"]),
                    unsafe_allow_html=True,
                )
            else:
                st.caption("No direct matches found.")

        st.markdown("<br>", unsafe_allow_html=True)
        st.progress(evaluation["verb_score"]    / 30,  text=f"Action Verbs: {evaluation['verb_score']}/30")
        st.progress(evaluation["keyword_score"] / 40,  text=f"Keyword Match: {evaluation['keyword_score']}/40")
        st.progress(evaluation["length_score"]  / 30,  text=f"Content Length: {evaluation['length_score']}/30")

    # ── TAB 1 : Resume ────────────────────────────────────────────────────────
    with tabs[1]:
        summary = sections.get("summary", "").strip()
        resume  = sections.get("resume",  "").strip()
        if summary:
            st.markdown("**Professional Summary**")
            st.info(summary)
        if resume:
            st.markdown("**Full Resume**")
            st.markdown(f'<div class="resume-text">{resume}</div>', unsafe_allow_html=True)
        if not summary and not resume:
            st.info("No resume content found.")

    # ── TAB 2 : Cover Letter ──────────────────────────────────────────────────
    with tabs[2]:
        cover = sections.get("cover_letter", "").strip()
        if cover:
            with st.container(border=True):
                st.markdown(cover)
        else:
            st.info("No cover letter generated.")

    # ── TAB 3 : ATS Analyzer ─────────────────────────────────────────────────
    with tabs[3]:
        if ats:
            ats_sc = ats["match_score"]
            if ats_sc >= 70:
                ats_cl, ats_lbl = "#10b981", "Strong Match 💪"
            elif ats_sc >= 45:
                ats_cl, ats_lbl = "#f59e0b", "Moderate Match 👍"
            else:
                ats_cl, ats_lbl = "#ef4444", "Weak Match ⚠️"

            st.markdown(f"""
            <div class="score-card">
              <div style="display:flex;align-items:center;gap:1.8rem;">
                <div>
                  <div class="ats-big" style="background:linear-gradient(135deg,{ats_cl},{ats_cl}aa);
                   -webkit-background-clip:text;-webkit-text-fill-color:transparent;">{ats_sc}</div>
                  <div class="score-lbl">ATS Match Score / 100</div>
                </div>
                <div style="font-size:1.2rem;font-weight:700;color:{ats_cl};">{ats_lbl}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            st.progress(ats_sc / 100, text=f"Match vs Job Description: {ats_sc}%")
            st.markdown("")

            kc1, kc2 = st.columns(2)
            with kc1:
                st.markdown("**❌ Missing Keywords — add these to your resume**")
                if ats["missing_keywords"]:
                    st.markdown(
                        "".join(f'<span class="tag-g">{k}</span>' for k in ats["missing_keywords"]),
                        unsafe_allow_html=True,
                    )
                else:
                    st.success("No major keyword gaps found!")
            with kc2:
                st.markdown("**✅ Keywords Already Present**")
                if ats["present_keywords"]:
                    st.markdown(
                        "".join(f'<span class="tag-s">{k}</span>' for k in ats["present_keywords"]),
                        unsafe_allow_html=True,
                    )
                else:
                    st.caption("No matching keywords found.")

            if ats["suggestions"]:
                st.markdown("**📋 Tailoring Suggestions**")
                for s in ats["suggestions"]:
                    st.markdown(f"- {s}")

            if ats["quick_wins"]:
                st.markdown("**⚡ Quick Wins**")
                for qw in ats["quick_wins"]:
                    st.markdown(f'<div class="tip-g">✅ {qw}</div>', unsafe_allow_html=True)

        else:
            st.info("💡 Paste a **Job Description** in the input above (before generating) to unlock ATS analysis.", icon="🎯")
            with st.container(border=True):
                st.markdown("""
**What ATS Analysis gives you:**
- 📊 Exact match score against the job posting
- ❌ Missing keywords recruiters are scanning for
- ✅ Keywords already working in your favour
- 📋 Specific rewrite suggestions
- ⚡ Quick wins to boost your score immediately
""")

    # ── TAB 4 : Tips ──────────────────────────────────────────────────────────
    with tabs[4]:
        st.markdown("**🔍 AI-Powered Improvement Tips**")
        st.caption("🔴 Fix before sending  ·  🟡 Fix this week  ·  🟢 Polish later")
        st.markdown("")
        if tips:
            for tip in tips:
                if tip.startswith("🔴"):
                    st.markdown(f'<div class="tip-r">{tip}</div>', unsafe_allow_html=True)
                elif tip.startswith("🟡"):
                    st.markdown(f'<div class="tip-y">{tip}</div>', unsafe_allow_html=True)
                elif tip.startswith("🟢"):
                    st.markdown(f'<div class="tip-g">{tip}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f"- {tip}")
        else:
            st.info("Tips will appear after generation.")

    # ── TAB 5 : LinkedIn Bio ──────────────────────────────────────────────────
    with tabs[5]:
        st.markdown("**💼 LinkedIn 'About' Section**")
        st.caption("First-person · Hook-first · Under 2,600 characters · Copy-paste ready")
        if linkedin_bio:
            st.markdown(f'<div class="linkedin-block">{linkedin_bio}</div>', unsafe_allow_html=True)
            st.markdown("")
            st.download_button(
                "⬇️ Download LinkedIn Bio (.txt)",
                data=linkedin_bio,
                file_name=f"{name_dl}_LinkedIn_Bio.txt",
                mime="text/plain",
            )
        else:
            st.info("LinkedIn bio will appear after generation.")

    # ─────────────────────────────────────────────────────────────────────────
    # DOWNLOAD BAR
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<p class="sl">⬇️ Downloads</p>', unsafe_allow_html=True)

    dl1, dl2, dl3 = st.columns(3)

    with dl1:
        try:
            pdf_bytes = generate_pdf(
                name=st.session_state.stored_name,
                email=st.session_state.stored_email,
                phone=st.session_state.stored_phone,
                linkedin=st.session_state.stored_linkedin,
                job_role=st.session_state.stored_role,
                sections=sections,
                evaluation=evaluation,
                template=pdf_template,
            )
            st.download_button(
                label=f"📄 Download PDF — {pdf_template}",
                data=pdf_bytes,
                file_name=f"{name_dl}_{role_dl}_{pdf_template}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception as pdf_err:
            st.error(f"PDF error: {pdf_err}")

    with dl2:
        sep = "-" * 40
        full_txt = (
            f"AI RESUME & CAREER KIT\nFor: {st.session_state.stored_name}\n{'='*60}\n\n"
            f"SUMMARY\n{sep}\n{sections.get('summary','')}\n\n"
            f"RESUME\n{sep}\n{sections.get('resume','')}\n\n"
            f"COVER LETTER\n{sep}\n{sections.get('cover_letter','')}\n\n"
            f"{'='*60}\n"
            f"Score: {evaluation['total_score']}/100  |  {evaluation['grade']}"
        )
        st.download_button(
            label="📝 Download .txt",
            data=full_txt,
            file_name=f"{name_dl}_{role_dl}_Resume.txt",
            mime="text/plain",
            use_container_width=True,
        )

    with dl3:
        if st.button("🔄 Regenerate", use_container_width=True):
            for k, v in DEFAULTS.items():
                st.session_state[k] = v
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# ACADEMIC FOOTER (preserved from original)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("📖 Academic Documentation — Capstone Project Details", expanded=False):
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Problem & Solution", "Algorithm", "Conclusion", "Future Scope", "References"
    ])
    with tab1:
        st.markdown("""
### 1. Problem Statement
Creating professional resumes and cover letters is a **time-intensive, skill-dependent** task.
Job seekers — especially students and fresh graduates — often lack:
- Writing expertise to articulate achievements clearly
- Knowledge of ATS (Applicant Tracking System) optimisation
- Ability to tailor documents for specific roles

### 2. Proposed Solution
We leverage **Groq's LLaMA** to automate:
- Professional summary generation (3–4 sentences)
- ATS-optimised resume creation with role-specific keywords
- Tailored cover letter drafting

### 3. System Development Approach
| Layer | Technology |
|-------|------------|
| Frontend UI | Streamlit (Python) |
| LLM Integration | Groq Python SDK |
| Prompt Design | Structured System + User Prompts |
| Evaluation | Rule-Based NLP Scoring |
| PDF Export | fpdf2 (3 templates) |
| Deployment | Streamlit Cloud / Local |
""")
    with tab2:
        st.markdown("""
### 4. Algorithm & ML Concepts

**4.1 Transformer Architecture (Vaswani et al., 2017)**
- Self-attention layers capture long-range token dependencies
- Decoder-only causal attention enables autoregressive generation
- Pre-training on large corpora enables zero-shot document generation

**4.2 Prompt Engineering**
- System prompt defines model persona and enforces output format
- User prompt encodes candidate profile as structured context tokens
- Section markers (SUMMARY / RESUME / COVER LETTER) guide parsing

**4.3 NLG Pipeline**
```
User Input → Tokenisation → Embedding → Transformer → Sampling → Output Text
```

**4.4 Resume Evaluation (Rule-Based Heuristics)**
- **Action Verb Score** (0–30): regex frequency extraction
- **Keyword Match Score** (0–40): substring search for multi-word skills
- **Content Length Score** (0–30): word count proxy for completeness

**4.5 ATS Gap Analysis**
- LLM compares job description tokens against resume text
- Returns match score, missing keywords, and targeted suggestions
""")
    with tab3:
        st.markdown("""
### 5. Conclusion
This capstone demonstrates that **context-aware generative AI** dramatically improves
professional document quality and creation speed.

Key findings:
- LLaMA-based generation produces ATS-optimised content superior to templates
- Prompt engineering gives fine-grained control over structure and tone
- Rule-based scoring provides transparent, interpretable feedback
- The modular architecture (utils.py) ensures maintainability and extensibility
- JD analysis and LinkedIn bio generation add real career-value beyond just a resume
""")
    with tab4:
        st.markdown("""
### 6. Future Scope

| Enhancement | Description |
|-------------|-------------|
| 🎯 Fine-tuning | Fine-tune LLaMA on HR-approved resume datasets |
| 🧠 Semantic Scoring | Replace rule-based scorer with BERT-based similarity |
| 🔄 RAG Integration | Retrieve role-specific examples via RAG |
| 🌐 Multi-language | Generate resumes in 10+ languages |
| 📱 Mobile App | React Native wrapper |
| 🔒 User Accounts | Save and version-control resume drafts |
""")
    with tab5:
        st.markdown("""
### 7. References

1. Vaswani, A., et al. (2017). *Attention Is All You Need*. NeurIPS. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
2. Brown, T., et al. (2020). *Language Models are Few-Shot Learners*. NeurIPS. [arXiv:2005.14165](https://arxiv.org/abs/2005.14165)
3. Meta AI. (2024). *LLaMA 3*. [ai.meta.com/llama](https://ai.meta.com/llama)
4. Groq. (2024). *Groq API Documentation*. https://console.groq.com/docs
5. Streamlit Inc. (2024). *Streamlit Documentation*. https://docs.streamlit.io
""")

st.markdown(f"""
<div style="text-align:center;color:{T['text_muted']};font-size:0.78rem;margin-top:1.5rem;padding-bottom:1rem;">
  🚀 AI Resume & Career Generator &nbsp;·&nbsp;
  Powered by Groq LLaMA &nbsp;·&nbsp;
  Built with Python &amp; Streamlit
</div>
""", unsafe_allow_html=True)