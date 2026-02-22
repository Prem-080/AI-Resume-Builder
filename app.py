"""
app.py — AI-Powered Resume & Cover Letter Generator
=====================================================
NOTE: Uses Groq API (free tier) — https://console.groq.com
      No billing required. Sign up and get a free API key instantly.
Capstone Project: Using Generative AI (GPT) with Streamlit

Academic Structure:
  1. Problem Statement     — Documented in README & comments
  2. Proposed Solution     — GPT-based NLG with prompt engineering
  3. System Dev Approach   — Modular Python + Streamlit + OpenAI API
  4. Algorithm & Deploy    — Transformer LLM + Streamlit Cloud
  5. Result                — Generated resume, score, and cover letter
  6. Conclusion            — Context-aware generation outperforms templates
  7. Future Scope          — Fine-tuning, multi-language, PDF export
  8. References            — OpenAI GPT, Vaswani et al. (2017), Streamlit docs
"""

import os
import streamlit as st
from dotenv import load_dotenv
from utils import generate_prompt, call_llm, format_output, evaluate_resume, generate_pdf

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION & ENVIRONMENT SETUP
# ─────────────────────────────────────────────────────────────────────────────
load_dotenv()

st.set_page_config(
    page_title="AI Resume & Cover Letter Generator",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS — Professional Dark-Mode Academic Theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Google Font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Background ── */
    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
        min-height: 100vh;
    }

    /* ── Main container ── */
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    /* ── Hero Header ── */
    .hero-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 40%, #f64f59 100%);
        border-radius: 16px;
        padding: 2.5rem 2rem;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(102,126,234,0.35);
        position: relative;
    }
    .hero-header h1 {
        color: #ffffff;
        font-size: 2.4rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 0 2px 10px rgba(0,0,0,0.3);
    }
    .hero-header p {
        color: rgba(255,255,255,0.88);
        font-size: 1.05rem;
        margin-top: 0.75rem;
        font-weight: 400;
    }
    .badge-row {
        display: flex;
        justify-content: center;
        gap: 0.6rem;
        flex-wrap: wrap;
        margin-top: 1.2rem;
    }
    .badge {
        background: rgba(255,255,255,0.15);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.25);
        border-radius: 20px;
        padding: 0.3rem 0.9rem;
        color: #fff;
        font-size: 0.8rem;
        font-weight: 500;
    }

    .section-label {
        color: #a78bfa;
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 0.5rem;
    }

    /* ── Score Card ── */
    .score-card {
        background: linear-gradient(135deg, rgba(16,185,129,0.12), rgba(59,130,246,0.12));
        border: 1px solid rgba(16,185,129,0.3);
        border-radius: 14px;
        padding: 1.5rem;
        margin-bottom: 1.2rem;
    }
    .score-number {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #10b981, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1;
    }
    .score-label {
        color: #94a3b8;
        font-size: 0.85rem;
        margin-top: 0.3rem;
    }
    .metric-row {
        display: flex;
        gap: 1rem;
        margin-top: 1rem;
        flex-wrap: wrap;
    }
    .metric-pill {
        background: rgba(255,255,255,0.06);
        border-radius: 8px;
        padding: 0.5rem 0.9rem;
        color: #e2e8f0;
        font-size: 0.82rem;
    }
    .metric-pill span {
        color: #a78bfa;
        font-weight: 600;
    }

    /* ── Input fields ── */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background: rgba(255,255,255,0.06) !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        border-radius: 8px !important;
        color: #e2e8f0 !important;
        font-family: 'Inter', sans-serif !important;
    }
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #7c3aed !important;
        box-shadow: 0 0 0 2px rgba(124,58,237,0.2) !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-family: 'Inter', sans-serif !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
        border: none !important;
        color: white !important;
        padding: 0.6rem 2rem !important;
        font-size: 1rem !important;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(124,58,237,0.45) !important;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: rgba(15,15,26,0.95) !important;
        border-right: 1px solid rgba(255,255,255,0.06) !important;
    }
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #a78bfa;
    }

    /* ── Alerts ── */
    .stAlert {
        border-radius: 10px !important;
    }

    /* ── Labels ── */
    label {
        color: #94a3b8 !important;
        font-size: 0.88rem !important;
        font-weight: 500 !important;
    }

    /* ── Divider ── */
    hr {
        border-color: rgba(255,255,255,0.08) !important;
    }

    /* ── Tag pills ── */
    .verb-tag {
        display: inline-block;
        background: rgba(167,139,250,0.15);
        border: 1px solid rgba(167,139,250,0.3);
        color: #c4b5fd;
        border-radius: 6px;
        padding: 0.15rem 0.55rem;
        font-size: 0.78rem;
        margin: 2px;
    }
    .skill-tag {
        display: inline-block;
        background: rgba(16,185,129,0.12);
        border: 1px solid rgba(16,185,129,0.3);
        color: #6ee7b7;
        border-radius: 6px;
        padding: 0.15rem 0.55rem;
        font-size: 0.78rem;
        margin: 2px;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — Project Info & Settings
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎓 Capstone Project")
    st.markdown("""
**AI-Powered Resume & Cover Letter Generator**
*Using Generative AI (GPT)*

---
### 📚 Academic Structure
1. Problem Statement
2. Proposed Solution
3. System Dev Approach
4. Algorithm & Deployment
5. Result & Evaluation
6. Conclusion
7. Future Scope
8. References

---
### 🔬 AI/ML Stack
- **Provider:** Groq (Free Tier)
- **Models:** LLaMA 3.3 70B · Mixtral
- **Technique:** Prompt Engineering
- **Architecture:** Transformer (LLM)
- **Task:** Natural Language Generation

---
### ⚙️ Settings
""")

    model_choice = st.selectbox(
        "Groq Model",
        ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
        index=0,
        help="llama-3.3-70b gives the best quality. llama-3.1-8b-instant is fastest."
    )

    st.markdown("---")
    st.markdown("### 📖 References")
    st.markdown("""
- Vaswani et al. (2017). *Attention Is All Need*.
- Brown et al. (2020). *Language Models are Few-Shot Learners*.
- Meta AI. (2024). *LLaMA 3: Open Foundation and Fine-Tuned Chat Models*. [ai.meta.com](https://ai.meta.com/llama)
- Jiang, A., et al. (2024). *Mixtral of Experts*. [arXiv:2401.04088](https://arxiv.org/abs/2401.04088)
- Groq API Documentation (2024).
- Streamlit Documentation (2024).
""")
    st.markdown("---")
    st.caption("Built with ❤️ using Python, Streamlit & Groq LLaMA")


# ─────────────────────────────────────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <h1>🤖 AI Resume & Cover Letter Generator</h1>
    <p>Capstone Project — Powered by Groq LLaMA & Prompt Engineering</p>
    <div class="badge-row">
        <span class="badge">🧠 Generative AI</span>
        <span class="badge">📄 NLP</span>
        <span class="badge">⚡ ATS-Optimized</span>
        <span class="badge">🎯 Prompt Engineering</span>
        <span class="badge">🐍 Python + Streamlit</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# API KEY HANDLING
# ─────────────────────────────────────────────────────────────────────────────
api_key = os.getenv("GROQ_API_KEY", "")

if not api_key:
    api_key = st.text_input(
        "🔑 Groq API Key (Free)",
        type="password",
        placeholder="gsk_...",
        help="Get a free key at https://console.groq.com — no billing required!",
    )
    if not api_key:
        st.warning(
            "⚠️ **Groq API Key Required** — Enter your free Groq API key above or set "
            "`GROQ_API_KEY` in your `.env` file to proceed.",
            icon="⚠️",
        )
        st.info(
            "💡 **How to get a FREE Groq API key (no billing needed):**\n"
            "1. Visit [console.groq.com](https://console.groq.com)\n"
            "2. Sign up with Google / GitHub\n"
            "3. Go to *API Keys* → *Create API Key*\n"
            "4. Copy and paste it above",
            icon="ℹ️",
        )
        st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# INPUT FORM — Two Column Layout
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<p class="section-label">📋 Candidate Profile</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

with col1:
    with st.container(border=True):
        st.markdown('<p class="section-label">👤 Personal Information</p>', unsafe_allow_html=True)
        full_name  = st.text_input("Full Name *", placeholder="e.g. Alex Johnson")
        email      = st.text_input("Email Address *", placeholder="alex.johnson@email.com")
        phone      = st.text_input("Phone Number *", placeholder="+1-555-0100")
        linkedin   = st.text_input("LinkedIn Profile URL", placeholder="linkedin.com/in/alexjohnson")

    with st.container(border=True):
        st.markdown('<p class="section-label">🎓 Education</p>', unsafe_allow_html=True)
        education = st.text_area(
            "Education Details *",
            placeholder="e.g. B.Tech in Computer Science, XYZ University, 2020–2024, GPA: 8.5",
            height=100,
        )

with col2:
    with st.container(border=True):
        st.markdown('<p class="section-label">💼 Professional Details</p>', unsafe_allow_html=True)
        job_role = st.text_input(
            "Target Job Role *",
            placeholder="e.g. Machine Learning Engineer Intern",
        )
        skills = st.text_input(
            "Technical Skills * (comma-separated)",
            placeholder="Python, TensorFlow, PyTorch, SQL, Docker",
        )

    with st.container(border=True):
        st.markdown('<p class="section-label">🚀 Experience & Projects</p>', unsafe_allow_html=True)
        projects = st.text_area(
            "Projects *",
            placeholder="• Sentiment Analysis — Built an LSTM classifier with 91% accuracy...\n• House Price Predictor — Reduced RMSE by 23% using feature engineering...",
            height=110,
        )
        experience = st.text_area(
            "Work Experience",
            placeholder="• Intern @ ABC Corp (June–Aug 2023) — Developed REST APIs using FastAPI...\nLeave blank if no experience yet.",
            height=110,
        )


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION HELPER
# ─────────────────────────────────────────────────────────────────────────────
def validate_inputs() -> list[str]:
    """Return list of validation error messages for required fields."""
    errors = []
    required = {
        "Full Name": full_name,
        "Email": email,
        "Phone": phone,
        "Education": education,
        "Skills": skills,
        "Projects": projects,
        "Target Job Role": job_role,
    }
    for field, value in required.items():
        if not value.strip():
            errors.append(f"**{field}** is required.")
    return errors


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE — Persist generated output across reruns
# ─────────────────────────────────────────────────────────────────────────────
if "generated" not in st.session_state:
    st.session_state.generated = False
if "output_sections" not in st.session_state:
    st.session_state.output_sections = {}
if "evaluation" not in st.session_state:
    st.session_state.evaluation = {}
if "raw_output" not in st.session_state:
    st.session_state.raw_output = ""


# ─────────────────────────────────────────────────────────────────────────────
# GENERATE BUTTON
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
btn_col1, btn_col2, btn_col3 = st.columns([1, 2, 1])

with btn_col2:
    generate_btn = st.button(
        "✨ Generate AI Resume & Cover Letter",
        type="primary",
        use_container_width=True,
    )

if generate_btn:
    errors = validate_inputs()
    if errors:
        st.error("❌ **Please fix the following errors:**\n" + "\n".join(f"  - {e}" for e in errors))
    else:
        with st.spinner("🤖 AI is crafting your personalized resume & cover letter..."):
            try:
                # Step 1: Build structured prompt (Prompt Engineering)
                prompt = generate_prompt(
                    name=full_name, email=email, phone=phone, linkedin=linkedin,
                    education=education, skills=skills, projects=projects,
                    experience=experience, job_role=job_role,
                )

                # Step 2: Call LLM (Transformer-based NLG)
                raw = call_llm(api_key=api_key, user_prompt=prompt, model=model_choice)

                # Step 3: Parse structured output
                sections = format_output(raw)

                # Step 4: Rule-based evaluation (ML heuristics)
                resume_text = sections.get("resume", "") + " " + sections.get("summary", "")
                evaluation  = evaluate_resume(resume_text, skills, job_role)

                # Store in session state
                st.session_state.generated       = True
                st.session_state.output_sections = sections
                st.session_state.evaluation      = evaluation
                st.session_state.raw_output      = raw
                st.session_state.stored_name     = full_name
                st.session_state.stored_email    = email
                st.session_state.stored_phone    = phone
                st.session_state.stored_linkedin = linkedin
                st.session_state.stored_role     = job_role

                st.success("✅ Resume generated successfully!")

            except Exception as e:
                error_msg = str(e)
                if "api_key" in error_msg.lower() or "authentication" in error_msg.lower() or "invalid_api_key" in error_msg.lower():
                    st.error("🔑 **Invalid API Key** — Check your Groq API key at console.groq.com")
                elif "rate limit" in error_msg.lower() or "rate_limit" in error_msg.lower():
                    st.error("⏳ **Rate Limit** — You've hit Groq's free tier limit. Wait a moment and try again.")
                elif "model" in error_msg.lower():
                    st.error(f"🤖 **Model Error** — Try switching to `llama-3.1-8b-instant` in the sidebar.\nDetails: {error_msg}")
                else:
                    st.error(f"❌ **Error:** {error_msg}")


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT SECTION — Displayed when generation is successful
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.generated and st.session_state.output_sections:
    sections   = st.session_state.output_sections
    evaluation = st.session_state.evaluation
    name_for_dl = st.session_state.get("stored_name", "resume").replace(" ", "_")
    role_for_dl = st.session_state.get("stored_role", "role").replace(" ", "_")

    st.markdown("---")
    
    # ── Tabbed Result Navigation ──
    res_tabs = st.tabs(["📊 Evaluation", "📌 Summary", "📄 Resume", "✉️ Cover Letter"])
    
    with res_tabs[0]:
        st.markdown('<p class="section-label">📊 Performance Metrics</p>', unsafe_allow_html=True)
        score = evaluation["total_score"]
        grade = evaluation["grade"]
        color = evaluation["color"]

        # Compact Score Card
        st.markdown(f"""
        <div class="score-card">
            <div style="display:flex; align-items:center; gap:1.5rem;">
                <div class="score-number">{score}</div>
                <div>
                    <div style="font-size: 1.4rem; color: {color}; font-weight: 700;">{grade}</div>
                    <div class="score-label">Overall Resume Strength</div>
                </div>
            </div>
            <div class="metric-row">
                <div class="metric-pill">⚡ Verbs: <span>{evaluation['verb_score']}/30</span></div>
                <div class="metric-pill">🎯 Skills: <span>{evaluation['keyword_score']}/40</span></div>
                <div class="metric-pill">📝 Length: <span>{evaluation['length_score']}/30</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        col_e1, col_e2 = st.columns(2)
        with col_e1:
            st.write("**⚡ Action Verbs Found**")
            if evaluation["found_verbs"]:
                tags = "".join(f'<span class="verb-tag">{v}</span>' for v in evaluation["found_verbs"])
                st.markdown(tags, unsafe_allow_html=True)
            else:
                st.caption("None detected.")

        with col_e2:
            st.write("**✅ Skills Matched**")
            if evaluation["skills_matched"]:
                tags = "".join(f'<span class="skill-tag">{s}</span>' for s in evaluation["skills_matched"])
                st.markdown(tags, unsafe_allow_html=True)
            else:
                st.caption("No direct matches.")

    with res_tabs[1]:
        summary_content = sections.get("summary", "").strip()
        if summary_content:
            with st.container(border=True):
                st.markdown(summary_content)
        else:
            st.info("No summary generated.")

    with res_tabs[2]:
        resume_content = sections.get("resume", "").strip()
        if resume_content:
            with st.container(border=True):
                # We use markdown instead of st.code for better typography in the UI
                st.markdown(resume_content)
        else:
            st.info("No resume generated.")

    with res_tabs[3]:
        cover_content = sections.get("cover_letter", "").strip()
        if cover_content:
            with st.container(border=True):
                st.markdown(cover_content)
        else:
            st.info("No cover letter generated.")

    # ─────────────────────────────────────────────────────────────────────────
    # ACTION BUTTONS
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    dl_col1, dl_col2, dl_col3 = st.columns([1.2, 1.2, 1])

    # Plain text for TXT download
    sep_line = "─" * 40
    full_text = f"""AI-POWERED RESUME & COVER LETTER\nGenerated for: {st.session_state.get('stored_name', '')}\n{'='*60}\n\nSUMMARY\n{sep_line}\n{sections.get('summary', '')}\n\nRESUME\n{sep_line}\n{sections.get('resume', '')}\n\nCOVER LETTER\n{sep_line}\n{sections.get('cover_letter', '')}\n\n{'='*60}\nScore: {evaluation['total_score']}/100 | Grade: {evaluation['grade']}"""

    with dl_col1:
        try:
            pdf_bytes = generate_pdf(
                name=st.session_state.get("stored_name", ""),
                email=st.session_state.get("stored_email", ""),
                phone=st.session_state.get("stored_phone", ""),
                linkedin=st.session_state.get("stored_linkedin", ""),
                job_role=st.session_state.get("stored_role", ""),
                sections=sections,
                evaluation=evaluation,
            )
            st.download_button(
                label="📄 Download PDF",
                data=pdf_bytes,
                file_name=f"{name_for_dl}_{role_for_dl}_Resume.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception as pdf_err:
            st.error(f"PDF Error: {pdf_err}")

    with dl_col2:
        st.download_button(
            label="⬇️ Download .txt",
            data=full_text,
            file_name=f"{name_for_dl}_{role_for_dl}_Resume.txt",
            mime="text/plain",
            use_container_width=True,
        )

    with dl_col3:
        if st.button("🔄 Regenerate", use_container_width=True):
            st.session_state.generated = False
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# ACADEMIC FOOTER — Problem Statement, Conclusion, Future Scope, References
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
Job seekers—especially students and fresh graduates—often lack:
- Writing expertise to articulate their achievements clearly
- Knowledge of ATS (Applicant Tracking System) optimization
- Ability to tailor documents for specific roles

### 2. Proposed Solution
We leverage **Groq's LLaMA (Large Language Model)** to automate:
- Professional summary generation (3–4 sentences)
- ATS-optimized resume creation
- Role-specific cover letter drafting

Using **Prompt Engineering**, we guide the LLM to produce structured, high-quality outputs.

### 3. System Development Approach
| Layer | Technology |
|-------|------------|
| Frontend UI | Streamlit (Python) |
| LLM Integration | Groq Python SDK |
| Prompt Design | Structured System + User Prompts |
| Evaluation | Rule-Based NLP Scoring |
| Deployment | Streamlit Cloud / Local |
""")

    with tab2:
        st.markdown("""
### 4. Algorithm & ML Concepts

**4.1 Transformer Architecture (Vaswani et al., 2017)**
- Self-attention layers capture long-range dependencies across resume content
- GPT uses decoder-only transformer with causal (left-to-right) attention
- Pre-training on 570GB+ of internet text enables zero-shot document generation

**4.2 Prompt Engineering**
- System prompt defines the model's "persona" and output format
- User prompt encodes candidate profile as structured tokens
- Two-shot examples (SUMMARY / RESUME / COVER LETTER headings) guide formatting

**4.3 Natural Language Generation (NLG) Pipeline**
```
User Input → Tokenization → Embedding → Transformer Layers → Sampling → Output Text
```

**4.4 Resume Evaluation (Rule-Based ML Heuristics)**
- **Action Verb Feature:** Regex-based verb frequency extraction
- **Keyword Matching:** Set intersection between skill tokens and resume tokens
- **Content Density:** Word count as a proxy for resume completeness
- **Scoring Formula:** `Score = VerbScore(0–30) + KeywordScore(0–40) + LengthScore(0–30)`

**4.5 Why Pretrained Models?**
- No labeled training data required (zero-shot capability)
- Generalizes across industries, job roles, and writing styles
- Cost-effective: inference only, no GPU training infrastructure needed
""")

    with tab3:
        st.markdown("""
### 6. Conclusion
This capstone demonstrates that **context-aware generative AI** can dramatically improve
the quality and efficiency of professional document creation.

Key findings:
- LLaMA-based generation produces ATS-optimized content superior to templates
- Prompt engineering allows fine-grained control over output structure and tone
- Rule-based scoring provides transparent, interpretable feedback to users
- The modular architecture (utils.py) ensures maintainability and extensibility

The project validates the practical application of **Large Language Models (LLMs)**
in real-world career assistance tools.
""")

    with tab4:
        st.markdown("""
### 7. Future Scope

| Enhancement | Description |
|-------------|-------------|
| 🎯 Fine-tuning | Fine-tune GPT on HR-approved resume datasets for domain-specific quality |
| 📊 PDF Export | Generate formatted PDF resumes with ReportLab or WeasyPrint |
| 🌐 Multi-language | Support resume generation in 10+ languages using multilingual LLMs |
| 🧠 Semantic Scoring | Replace rule-based scoring with BERT-based semantic similarity |
| 🔄 RAG Integration | Retrieve job-role-specific examples using Retrieval-Augmented Generation |
| 📱 Mobile App | React Native wrapper for mobile accessibility |
| 📈 Analytics Dashboard | Track resume score improvements across regenerations |
| 🔒 User Accounts | Save and version-control multiple resume drafts |
""")

    with tab5:
        st.markdown("""
### 8. References

1. Vaswani, A., et al. (2017). *Attention Is All Need*. NeurIPS 2017. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
2. Brown, T., et al. (2020). *Language Models are Few-Shot Learners*. NeurIPS 2020. [arXiv:2005.14165](https://arxiv.org/abs/2005.14165)
3. Meta AI. (2024). *LLaMA 3: Open Foundation and Fine-Tuned Chat Models*. [ai.meta.com](https://ai.meta.com/llama)
4. Jiang, A., et al. (2024). *Mixtral of Experts*. [arXiv:2401.04088](https://arxiv.org/abs/2401.04088)
5. Groq. (2024). *Groq API Documentation*. https://console.groq.com/docs
6. Streamlit Inc. (2024). *Streamlit Documentation*. https://docs.streamlit.io
7. Manning, C., & Schütze, H. (1999). *Foundations of Statistical Natural Language Processing*. MIT Press.
""")

st.markdown("""
<div style="text-align: center; color: rgba(148,163,184,0.5); font-size: 0.8rem; margin-top: 2rem;">
    🤖 AI Resume &amp; Cover Letter Generator | Capstone Project | Powered by Groq LLaMA &amp; Streamlit<br>
    Built with Python 🐍 | For Academic &amp; Professional Use
</div>
""", unsafe_allow_html=True)
