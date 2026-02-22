"""
utils.py — Modular Helper Functions for AI Resume & Cover Letter Generator
==========================================================================
NOTE: Uses Groq API (free tier) — https://console.groq.com
      Models: llama-3.3-70b-versatile, llama-3.1-8b-instant, mixtral-8x7b-32768

AI/ML Justification:
--------------------
This module leverages a Transformer-based Large Language Model (LLM) — OpenAI's GPT —
to perform context-aware Natural Language Generation (NLG).

Key ML Concepts Used:
1. Transformer Architecture:
   - GPT uses self-attention mechanisms to understand contextual relationships
     between words, enabling coherent, role-specific document generation.

2. Prompt Engineering:
   - Structured system + user prompts guide the model's generative behavior,
     enforcing output format (SUMMARY / RESUME / COVER LETTER sections).

3. Natural Language Processing (NLP):
   - The model tokenizes, encodes, and decodes text sequences using subword
     tokenization (BPE), handling diverse vocabulary out of the box.

4. Context-Aware Text Generation:
   - The model considers all input fields (skills, experience, job role) as
     unified context, personalizing output without fine-tuning.

5. Pretrained Generative Models Efficiency:
   - Using a pretrained model eliminates the need for custom training data,
     making the solution scalable and cost-effective for capstone demonstration.

Resume Evaluation (Rule-Based ML Heuristics):
----------------------------------------------
- Action verb counting mimics feature extraction used in NLP classifiers.
- Keyword matching between skills and job role simulates semantic similarity
  scoring, a core concept in Information Retrieval and NLP pipelines.
- The Resume Strength Score is a weighted rule-based model — a precursor to
  learned scoring models (e.g., linear regression on NLP features).
"""

import re
from io import BytesIO
from groq import Groq
from fpdf import FPDF

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# Strong action verbs commonly rewarded by ATS systems
ACTION_VERBS = [
    "achieved", "analyzed", "built", "collaborated", "created", "designed",
    "developed", "enhanced", "engineered", "executed", "facilitated",
    "generated", "implemented", "improved", "integrated", "launched",
    "led", "managed", "optimized", "orchestrated", "produced", "reduced",
    "researched", "resolved", "spearheaded", "streamlined", "trained",
    "transformed", "utilized", "won", "automated", "deployed", "documented",
    "established", "formulated", "initiated", "maintained", "mentored",
    "negotiated", "performed", "planned", "presented", "programmed",
    "published", "reviewed", "solved", "supported", "tested", "validated",
]

SYSTEM_PROMPT = """You are an expert HR professional and senior resume writer.
Generate a high-end, ATS-optimized professional document set.

### GUIDELINES:
1. **Professional Summary**: 3-4 impactful lines using strong action verbs.
2. **Resume**: Use clear sections with markdown-style headers (e.g., ## EXPERIENCE, ## EDUCATION). 
   - Use bullet points (•) for responsibilities.
   - Focus on quantifiable achievements.
3. **Cover Letter**: Professional, tailored, and persuasive.

### IMPORTANT FORMATTING RULES:
- FORCE a single-column layout. 
- ABSOLUTELY NO multi-column formatting using spaces.
- Vertical flow only.
- Use '##' for main section headers in the resume.

Structure your response EXACTLY as follows:

SUMMARY
[Professional summary here]

RESUME
[Full structured resume here starting with ## headers]

COVER LETTER
[Full professional cover letter here]
"""


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION: generate_prompt()
# ─────────────────────────────────────────────────────────────────────────────
def generate_prompt(
    name: str,
    email: str,
    phone: str,
    linkedin: str,
    education: str,
    skills: str,
    projects: str,
    experience: str,
    job_role: str,
) -> str:
    """
    Convert structured user input into a formatted LLM prompt.

    This function performs input serialization — converting heterogeneous
    form fields into a unified textual representation, which is then
    interpreted by the LLM's attention mechanism as contextual tokens.

    Args:
        All individual user profile fields.

    Returns:
        str: A well-structured prompt string for the LLM.
    """
    prompt = f"""
Generate a professional resume, summary, and cover letter for the following candidate applying for the role of **{job_role}**.

--- CANDIDATE PROFILE ---
Full Name       : {name}
Email           : {email}
Phone           : {phone}
LinkedIn        : {linkedin}

Education       : {education}

Technical Skills: {skills}

Projects        :
{projects}

Work Experience :
{experience}

Target Job Role : {job_role}
--- END OF PROFILE ---

Follow the output format strictly:

SUMMARY
[Write 3-4 line professional summary]

RESUME
[Write full ATS-optimized resume]

COVER LETTER
[Write a tailored, professional cover letter for {job_role}]
"""
    return prompt.strip()


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION: call_llm()
# ─────────────────────────────────────────────────────────────────────────────
def call_llm(api_key: str, user_prompt: str, model: str = "llama-3.3-70b-versatile") -> str:
    """
    Invoke a Groq-hosted LLM using prompt engineering (free tier).

    Groq runs open-source models (LLaMA 3, Mixtral) on custom LPU hardware,
    offering the same OpenAI-compatible chat-completions interface at no cost.

    The model uses:
    - Multi-head self-attention to weigh relationships across all input tokens
    - Autoregressive decoding to generate tokens left-to-right
    - Temperature-controlled sampling for creative yet coherent outputs

    Args:
        api_key     (str): Groq API key from https://console.groq.com
        user_prompt (str): Formatted candidate prompt.
        model       (str): Groq model ID (default: llama-3.3-70b-versatile).

    Returns:
        str: Raw generated text from the LLM.
    """
    client = Groq(api_key=api_key)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.7,   # Controls creativity vs. determinism
        max_tokens=2000,   # Sufficient for resume + cover letter
        top_p=0.95,        # Nucleus sampling for diverse vocabulary
    )

    return response.choices[0].message.content.strip()


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION: format_output()
# ─────────────────────────────────────────────────────────────────────────────
def format_output(raw_text: str) -> dict:
    """
    Parse and extract structured sections from LLM-generated text.

    This function applies regex-based pattern matching (a classic NLP
    preprocessing technique) to segment the LLM's response into distinct
    named sections: SUMMARY, RESUME, and COVER LETTER.

    Args:
        raw_text (str): Raw string output from the LLM.

    Returns:
        dict: Keys 'summary', 'resume', 'cover_letter' with extracted content.
    """
    sections = {"summary": "", "resume": "", "cover_letter": ""}

    # Regex pattern to split on section headers
    pattern = r"(?:^|\n)(SUMMARY|RESUME|COVER LETTER)\s*\n"
    parts = re.split(pattern, raw_text, flags=re.IGNORECASE)

    current_key = None
    for part in parts:
        part = part.strip()
        if part.upper() == "SUMMARY":
            current_key = "summary"
        elif part.upper() == "RESUME":
            current_key = "resume"
        elif part.upper() == "COVER LETTER":
            current_key = "cover_letter"
        elif current_key:
            sections[current_key] += part

    # Fallback: if parsing fails, put everything in resume
    if not any(sections.values()):
        sections["resume"] = raw_text

    return {k: v.strip() for k, v in sections.items()}


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION: evaluate_resume()
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_resume(resume_text: str, skills: str, job_role: str) -> dict:
    """
    Compute a rule-based Resume Strength Score.

    This evaluation mimics feature engineering in NLP classification models:
    - Action verb frequency → linguistic quality feature
    - Keyword overlap → TF-based relevance scoring (precursor to TF-IDF)
    - Length check → content density feature

    Scoring Breakdown (out of 100):
    ┌──────────────────────────────────┬────────┐
    │ Component                        │ Weight │
    ├──────────────────────────────────┼────────┤
    │ Action Verb Usage (≥5 verbs=30)  │  30    │
    │ Skill–Job Role Keyword Match     │  40    │
    │ Content Length (≥300 words=30)   │  30    │
    └──────────────────────────────────┴────────┘

    Args:
        resume_text (str): Generated resume text.
        skills      (str): Comma-separated user skills.
        job_role    (str): Target job role string.

    Returns:
        dict: Score details including total score, verb count, keyword matches.
    """
    resume_lower = resume_text.lower()

    # 1. Action Verb Count
    found_verbs = [v for v in ACTION_VERBS if re.search(r"\b" + v + r"\b", resume_lower)]
    verb_count = len(found_verbs)
    verb_score = min(30, verb_count * 5)   # 5 pts per verb, max 30

    # 2. Keyword Matching — Skills vs. Job Role
    skill_list = [s.strip().lower() for s in skills.split(",") if s.strip()]
    job_words = set(re.findall(r"\b\w+\b", job_role.lower()))
    resume_words = set(re.findall(r"\b\w+\b", resume_lower))

    # Skills present in resume
    skills_in_resume = [s for s in skill_list if s in resume_words]
    # Job-role keywords present in resume
    job_kw_in_resume = [w for w in job_words if w in resume_words and len(w) > 3]

    total_keywords = len(skill_list) + len(job_words)
    matched_keywords = len(skills_in_resume) + len(job_kw_in_resume)
    keyword_score = min(40, int((matched_keywords / max(total_keywords, 1)) * 80))

    # 3. Content Length Score
    word_count = len(resume_lower.split())
    length_score = min(30, int((word_count / 300) * 30))

    total_score = verb_score + keyword_score + length_score

    # Determine grade
    if total_score >= 85:
        grade, color = "Excellent ✅", "green"
    elif total_score >= 65:
        grade, color = "Good 👍", "blue"
    elif total_score >= 45:
        grade, color = "Average ⚠️", "orange"
    else:
        grade, color = "Needs Improvement ❌", "red"

    return {
        "total_score": total_score,
        "verb_score": verb_score,
        "keyword_score": keyword_score,
        "length_score": length_score,
        "verb_count": verb_count,
        "found_verbs": found_verbs[:10],       # Top 10 for display
        "skills_matched": skills_in_resume,
        "job_keywords_matched": list(job_kw_in_resume),
        "word_count": word_count,
        "grade": grade,
        "color": color,
    }



# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION: generate_pdf()
# ─────────────────────────────────────────────────────────────────────────────
def generate_pdf(
    name: str,
    email: str,
    phone: str,
    linkedin: str,
    job_role: str,
    sections: dict,
    evaluation: dict,
) -> bytes:
    """
    Build a V2 Premium structured PDF using fpdf2.
    
    Layout:
      - Centered Header: Bold Name + High-contrast subtitle
      - Contact Strip: Icons/Labels in a clean centered line
      - Sections: Each parsed for '##' markers to draw bold headers with accent underlines
      - Footer: Scoring metrics in a sleek bar
    """

    def safe(text: str) -> str:
        """Sanitize text for Latin-1 encoding with expanded replacements."""
        replacements = {
            "\u2013": "-", "\u2014": "--", "\u2018": "'", "\u2019": "'",
            "\u201c": '"', "\u201d": '"', "\u2022": "*", "\u2026": "...",
            "\u00e9": "e", "\u00e8": "e", "\u00ea": "e", "\u00e0": "a",
            "\u2192": "->", "\u2714": "v", "\u2713": "v", "\u00b7": ".",
            "\u00a0": " ", "\u2500": "-", "\u2014": "--",
        }
        for k, v in replacements.items():
            text = text.replace(k, v)
        # Final pass: remove everything non-latin1
        return text.encode("latin-1", errors="ignore").decode("latin-1")

    # ── Color Palette ───────────────────────────────────────────────────────
    PRIMARY     = (43,  45,  66)   # Charcoal
    ACCENT      = (72,  149, 239)  # Soft blue
    MUTED       = (108, 117, 125)  # Grey
    BG_LIGHT    = (248, 249, 250)
    TEXT_MAIN   = (33,  37,  41)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()
    pdf.set_margins(18, 18, 18)
    W = pdf.w - 36

    def draw_header() -> None:
        """Render the modern centered header on current page."""
        pdf.set_y(15)
        pdf.set_font("Helvetica", "B", 24)
        pdf.set_text_color(*PRIMARY)
        pdf.cell(W, 12, safe(name.upper() if name else "CANDIDATE NAME"), ln=True, align="C")

        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(*ACCENT)
        pdf.cell(W, 6, safe(job_role.upper() if job_role else "PROFESSIONAL PROFILE"), ln=True, align="C")

        # Contact Strip
        parts = [p.strip() for p in [email, phone, linkedin] if p and p.strip()]
        if parts:
            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(*MUTED)
            contact_line = "  •  ".join(parts)
            pdf.cell(W, 7, safe(contact_line), ln=True, align="C")

        pdf.ln(8)
        pdf.set_draw_color(*ACCENT)
        pdf.set_line_width(0.5)
        pdf.line(18, pdf.get_y(), 18 + W, pdf.get_y())
        pdf.ln(6)

    # Draw first page header
    draw_header()

    def draw_section_header(label: str) -> None:
        """Render a section divider with accent underline."""
        pdf.ln(4)
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(*PRIMARY)
        pdf.cell(W, 7, safe(label.upper()), ln=True)
        # Accent underline
        curr_y = pdf.get_y() - 1
        pdf.set_draw_color(*ACCENT)
        pdf.set_line_width(0.3)
        pdf.line(18, curr_y, 18 + 25, curr_y) # Short underline
        pdf.ln(2)

    def draw_body_block(content: str) -> None:
        """Smart multi-line rendering with bullet point detection."""
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(*TEXT_MAIN)
        
        # Collapse excessive AI whitespace
        content = re.sub(r' {3,}', ' ', content)
        
        for line in content.splitlines():
            line = line.strip()
            if not line:
                pdf.ln(2)
                continue
            
            # Detect Markdown-style headers inside body
            if line.startswith("##"):
                pdf.ln(3)
                pdf.set_font("Helvetica", "B", 10)
                pdf.set_text_color(*PRIMARY)
                pdf.cell(W, 6, safe(line.replace("##", "").strip().upper()), ln=True)
                pdf.set_font("Helvetica", "", 10)
                pdf.set_text_color(*TEXT_MAIN)
                continue

            # Detect bullets
            if line.startswith("•") or line.startswith("-") or line.startswith("*"):
                # Indented bullet point
                pdf.set_x(23)
                line = "• " + line.lstrip("•-* ")
                pdf.multi_cell(W - 5, 5.5, safe(line))
            else:
                pdf.multi_cell(W, 5.5, safe(line))
        pdf.ln(2)

    # ── Content Loop ────────────────────────────────────────────────────────
    # SUMMARY
    summary = sections.get("summary", "").strip()
    if summary:
        draw_section_header("Professional Summary")
        draw_body_block(summary)

    # RESUME (Usually contains multiple subsections like Experience, Education)
    resume = sections.get("resume", "").strip()
    if resume:
        # Check if it has its own headers
        if "##" not in resume:
            draw_section_header("Resume Details")
        draw_body_block(resume)

    # COVER LETTER (Always start on a new page for professional separation)
    cover = sections.get("cover_letter", "").strip()
    if cover:
        pdf.add_page()
        draw_header()
        draw_section_header("Cover Letter")
        draw_body_block(cover)

    # ── SLEEK FOOTER ────────────────────────────────────────────────────────
    score = evaluation.get("total_score", 0) or 0
    grade = evaluation.get("grade", "N/A")
    words = evaluation.get("word_count", 0)

    pdf.set_y(-25)
    pdf.set_fill_color(*PRIMARY)
    pdf.rect(0, pdf.h - 18, pdf.w, 18, "F")
    
    pdf.set_y(pdf.h - 14)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 9)
    footer_text = f"RESUME STRENGTH SCORE: {score}/100  |  GRADE: {grade}  |  WORD COUNT: {words}"
    pdf.cell(W, 5, safe(footer_text), align="C", ln=True)
    
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(180, 180, 180)
    pdf.cell(W, 5, "Generated via AI Resume Builder - Professional Edition", align="C")

    # ── Export ──────────────────────────────────────────────────────────────
    raw = pdf.output()
    if isinstance(raw, (bytes, bytearray)):
        return bytes(raw)
    buf = BytesIO()
    buf.write(raw)
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# SAMPLE GENERATED OUTPUT (for documentation / PPT reference)
# ─────────────────────────────────────────────────────────────────────────────
# INPUT EXAMPLE:
#   Name: Alex Johnson | Role: Machine Learning Engineer
#   Skills: Python, TensorFlow, PyTorch, Scikit-learn, SQL
#   Education: B.Tech Computer Science, XYZ University (2024)
#
# OUTPUT EXAMPLE:
#
# SUMMARY
# A highly motivated Computer Science graduate with hands-on experience
# in machine learning and deep learning frameworks. Demonstrated ability
# to build, train, and deploy predictive models using TensorFlow and PyTorch.
# Seeking an entry-level ML Engineer role to contribute to innovative AI solutions.
#
# RESUME
# ALEX JOHNSON
# alex.johnson@email.com | +1-555-0100 | linkedin.com/in/alexjohnson
#
# EDUCATION
# B.Tech in Computer Science — XYZ University | 2020–2024 | GPA: 8.7/10
#
# TECHNICAL SKILLS
# Languages: Python, SQL | Frameworks: TensorFlow, PyTorch, Scikit-learn
# Tools: Git, Jupyter, Docker | Cloud: AWS SageMaker (basics)
#
# PROJECTS
# • Sentiment Analysis Engine — Built an LSTM-based sentiment classifier
#   achieving 91% accuracy on the IMDb dataset using PyTorch.
# • House Price Predictor — Developed a regression model with Scikit-learn
#   reducing RMSE by 23% through feature engineering.
#
# COVER LETTER
# Dear Hiring Manager,
# I am excited to apply for the Machine Learning Engineer position. As a
# recent Computer Science graduate with strong expertise in Python, TensorFlow,
# and PyTorch, I have developed and deployed ML models that deliver measurable
# results. I am eager to bring my analytical mindset and technical skills to
# your team. Thank you for your consideration.
# ─────────────────────────────────────────────────────────────────────────────
