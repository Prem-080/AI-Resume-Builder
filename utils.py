"""
utils.py — AI Resume & Cover Letter Generator  |  Helper Functions
===================================================================
Groq API (free tier) — https://console.groq.com
Models: llama-3.3-70b-versatile | llama-3.1-8b-instant | mixtral-8x7b-32768

New additions:
  · generate_pdf()  — 3 distinct PDF templates: Modern, Classic, Minimal
  · analyze_jd()    — Job Description ATS gap analysis & keyword match score
  · get_tips()      — Prioritised AI improvement suggestions for the resume
  · gen_linkedin()  — LinkedIn "About" section generator

AI/ML Concepts:
  1. Transformer Architecture (LLaMA)  — multi-head self-attention for NLG
  2. Prompt Engineering                — structured system+user prompts enforce format
  3. NLP Preprocessing                 — regex section parsing, keyword extraction
  4. Rule-Based Scoring Heuristics     — weighted scoring for resume strength
"""

import re
from io import BytesIO
from groq import Groq
from fpdf import FPDF


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
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

# Plain-text output enforced — prevents markdown bleeding into PDF renderer
SYSTEM_PROMPT = """You are a senior HR professional and expert resume writer with 15+ years of experience.

=== STRICT OUTPUT FORMAT ===
Output EXACTLY these three section markers on their own line (uppercase, nothing else on the line):

SUMMARY
<3-4 sentence professional summary using strong action verbs>

RESUME
<Full ATS-optimized resume>

COVER LETTER
<Full tailored cover letter>

=== MANDATORY FORMATTING RULES ===
- PLAIN TEXT ONLY. No markdown. No *, #, _, ** symbols anywhere.
- Section headers inside the resume (EDUCATION, TECHNICAL SKILLS, WORK EXPERIENCE,
  PROJECTS) must be ALL CAPS on their own line with no punctuation after them.
- Use "  - " (two spaces + dash + space) for every bullet point.
- Single-column layout ONLY. No side-by-side text using spaces or tabs.
- Do NOT repeat contact details inside SUMMARY or COVER LETTER.
- Quantify achievements with numbers/percentages wherever possible.
- Use strong action verbs throughout all sections.

=== RESUME STRUCTURE ===
[CANDIDATE NAME]
[Email] | [Phone] | [LinkedIn]

PROFESSIONAL SUMMARY
[2-3 impact sentences]

EDUCATION
[Degree, Institution, Year, GPA]

TECHNICAL SKILLS
[Languages: ... | Frameworks: ... | Tools: ...]

WORK EXPERIENCE
[Job Title | Company | Dates]
  - [Quantified achievement using action verb]

PROJECTS
[Project Name] | [Tech Stack]
  - [What was built and measurable result]

=== COVER LETTER STRUCTURE ===
[City, Date]
Dear Hiring Manager,
[Opening: enthusiasm for the specific role]
[Body: 2-3 specific achievements]
[Closing: call to action]
Sincerely,
[Candidate Name]
"""


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL: Shared Groq API caller
# ─────────────────────────────────────────────────────────────────────────────
def _groq(api_key: str, system: str, user: str,
          model: str = "llama-3.3-70b-versatile",
          temperature: float = 0.65,
          max_tokens: int = 3500) -> str:
    client = Groq(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user",   "content": user}],
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=0.9,
    )
    return resp.choices[0].message.content.strip()


# ─────────────────────────────────────────────────────────────────────────────
# CORE: Resume generation pipeline
# ─────────────────────────────────────────────────────────────────────────────
def generate_prompt(name: str, email: str, phone: str, linkedin: str,
                    education: str, skills: str, projects: str,
                    experience: str, job_role: str) -> str:
    """Serialise form inputs into a structured LLM prompt string."""
    exp_val = experience.strip() if experience.strip() else "No prior work experience."
    li_val  = linkedin.strip()   if linkedin.strip()   else "N/A"
    return f"""Generate a professional SUMMARY, RESUME, and COVER LETTER for:

TARGET ROLE: {job_role}

--- CANDIDATE PROFILE ---
Name        : {name}
Email       : {email}
Phone       : {phone}
LinkedIn    : {li_val}
Education   : {education}
Skills      : {skills}
Experience  : {exp_val}
Projects    :
{projects.strip()}
--- END PROFILE ---

Plain text only. No markdown symbols. Output SUMMARY, RESUME, and COVER LETTER
as standalone uppercase markers on their own lines. Optimise for ATS keyword
matching for the role: {job_role}.
""".strip()


def call_llm(api_key: str, user_prompt: str,
             model: str = "llama-3.3-70b-versatile") -> str:
    """Call the LLM to generate the full resume package."""
    return _groq(api_key, SYSTEM_PROMPT, user_prompt, model, 0.65, 3500)


def format_output(raw_text: str) -> dict:
    """
    Parse LLM output into named sections: summary, resume, cover_letter.

    Robust to LLM variations — handles '## SUMMARY', '**RESUME**', 'COVER LETTER:'
    and strips any residual markdown that slipped through despite instructions.
    """
    sections: dict[str, str] = {"summary": "", "resume": "", "cover_letter": ""}

    def strip_md(text: str) -> str:
        text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
        text = re.sub(r"\*(.+?)\*",     r"\1", text)
        text = re.sub(r"__(.+?)__",     r"\1", text)
        text = re.sub(r"^#{1,6}\s+",    "",    text, flags=re.MULTILINE)
        text = re.sub(r"^\s*[•]\s",     "  - ", text, flags=re.MULTILINE)
        return text

    raw_text = strip_md(raw_text)

    # Handles: SUMMARY / ## SUMMARY / **SUMMARY** / SUMMARY: / RESUME etc.
    pattern = (r"(?im)^[ \t]*(?:#+\s*)?(?:\*{1,2})?"
               r"(SUMMARY|RESUME|COVER LETTER)"
               r"(?:\*{1,2})?[ \t]*:?\s*$")
    parts = re.split(pattern, raw_text)

    current = None
    for part in parts:
        key = part.strip().upper()
        if key == "SUMMARY":        current = "summary"
        elif key == "RESUME":       current = "resume"
        elif key == "COVER LETTER": current = "cover_letter"
        elif current:               sections[current] += part

    if not any(sections.values()):
        sections["resume"] = raw_text.strip()

    return {k: v.strip() for k, v in sections.items()}


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE: Evaluation (rule-based resume strength scorer)
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_resume(resume_text: str, skills: str, job_role: str) -> dict:
    """
    Rule-based Resume Strength Score (0–100).

    Components:
      Action Verbs   — 5 pts each, max 30
      Keyword Match  — scaled to 40 pts (multi-word skills use substring search)
      Content Length — scaled to 30 pts, target >= 300 words
    """
    rl = resume_text.lower()

    # 1. Action verbs
    found_verbs = [v for v in ACTION_VERBS if re.search(r"\b" + v + r"\b", rl)]
    verb_score  = min(30, len(found_verbs) * 5)

    # 2. Keyword matching — substring search catches multi-word skills
    skill_list = [s.strip().lower() for s in skills.split(",") if s.strip()]
    job_words  = set(re.findall(r"\b\w+\b", job_role.lower()))
    skills_hit = [s for s in skill_list if s in rl]
    jkw_hit    = [w for w in job_words
                  if re.search(r"\b" + re.escape(w) + r"\b", rl) and len(w) > 3]
    total_kw   = max(len(skill_list) + len(job_words), 1)
    kw_score   = min(40, int(((len(skills_hit) + len(jkw_hit)) / total_kw) * 80))

    # 3. Length
    wc        = len(rl.split())
    len_score = min(30, int((wc / 300) * 30))

    total = verb_score + kw_score + len_score

    if   total >= 85: grade, color = "Excellent ✅",        "#10b981"
    elif total >= 65: grade, color = "Good 👍",              "#3b82f6"
    elif total >= 45: grade, color = "Average ⚠️",           "#f59e0b"
    else:             grade, color = "Needs Improvement ❌", "#ef4444"

    return {
        "total_score": total,  "verb_score": verb_score,
        "keyword_score": kw_score, "length_score": len_score,
        "verb_count": len(found_verbs), "found_verbs": found_verbs[:10],
        "skills_matched": skills_hit,  "job_keywords_matched": list(jkw_hit),
        "word_count": wc, "grade": grade, "color": color,
    }


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE: ATS Job Description Analyzer
# ─────────────────────────────────────────────────────────────────────────────
def analyze_jd(api_key: str, job_description: str, resume_text: str,
               model: str = "llama-3.3-70b-versatile") -> dict:
    """
    Compare a job description against the generated resume.
    Returns: match_score (0-100), missing/present keywords, suggestions, quick wins.
    """
    system = """You are an ATS (Applicant Tracking System) expert analyst.
Compare the job description against the candidate resume. Respond in EXACTLY this format:

MATCH_SCORE
[integer 0-100 only]

MISSING_KEYWORDS
[comma-separated list of important JD keywords absent from the resume]

PRESENT_KEYWORDS
[comma-separated list of JD keywords already found in the resume]

SUGGESTIONS
[5 specific actionable improvements, each on its own line starting with "- "]

QUICK_WINS
[3 immediate changes to boost ATS score, each on its own line starting with "- "]
"""
    raw = _groq(api_key, system,
                f"JOB DESCRIPTION:\n{job_description}\n\nRESUME:\n{resume_text}",
                model, temperature=0.25, max_tokens=900)

    def block(marker: str) -> str:
        m = re.search(rf"(?im)^{marker}\s*\n(.*?)(?=\n[A-Z_]{{3,}}\n|\Z)", raw, re.DOTALL)
        return m.group(1).strip() if m else ""

    score = 0
    try:
        score = int(re.search(r"\d+", block("MATCH_SCORE")).group())
    except Exception:
        pass

    csv  = lambda t: [k.strip() for k in t.split(",") if k.strip()]
    buls = lambda t: [s.lstrip("- ").strip() for s in t.splitlines() if s.strip()]

    return {
        "match_score":      score,
        "missing_keywords": csv(block("MISSING_KEYWORDS")),
        "present_keywords": csv(block("PRESENT_KEYWORDS")),
        "suggestions":      buls(block("SUGGESTIONS")),
        "quick_wins":       buls(block("QUICK_WINS")),
    }


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE: AI Resume Improvement Tips
# ─────────────────────────────────────────────────────────────────────────────
def get_tips(api_key: str, resume_text: str, job_role: str,
             model: str = "llama-3.3-70b-versatile") -> list[str]:
    """
    Return 8 prioritised improvement tips for the resume.
    Each tip starts with a priority emoji: 🔴 (critical) 🟡 (important) 🟢 (polish).
    """
    system = """You are a professional resume coach.
Give exactly 8 specific, actionable improvement tips.
Format: one tip per line, starting with a priority emoji:
  🔴 (critical — fix before sending)
  🟡 (important — fix this week)
  🟢 (nice to have — polish later)
No numbering. No preamble. No extra explanation. Exactly 8 lines."""
    raw = _groq(api_key, system,
                f"Target role: {job_role}\n\nResume:\n{resume_text}\n\nGive 8 prioritised tips.",
                model, temperature=0.4, max_tokens=700)
    tips = [l.strip() for l in raw.splitlines()
            if l.strip() and any(l.strip().startswith(e) for e in ("🔴", "🟡", "🟢"))]
    return tips or [l.strip() for l in raw.splitlines() if l.strip()][:8]


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE: LinkedIn Bio Generator
# ─────────────────────────────────────────────────────────────────────────────
def gen_linkedin(api_key: str, name: str, job_role: str, skills: str,
                 experience: str, projects: str,
                 model: str = "llama-3.3-70b-versatile") -> str:
    """
    Generate a LinkedIn 'About' section following personal branding best practices:
    first-person voice, hook-first opening, 3 paragraphs, call to action, ≤2600 chars.
    """
    system = """You are a LinkedIn personal branding expert.
Write a compelling LinkedIn About section following these rules:
- First-person voice ("I build...", "I've spent...")
- Hook in the first sentence — no clichés like "passionate" or "results-driven"
- Three short paragraphs: who you are | what you build/do | what you are looking for
- End with a specific, concrete call to action
- Maximum 2,600 characters total
- No hashtags. No bullet points. No emojis. Professional but warm tone."""
    exp_val = experience.strip() if experience.strip() else "Entry-level / student"
    return _groq(api_key, system,
                 f"Name: {name}\nTarget Role: {job_role}\nSkills: {skills}\n"
                 f"Experience: {exp_val}\nProjects: {projects}\n\nWrite the LinkedIn About section.",
                 model, temperature=0.72, max_tokens=600)


# ─────────────────────────────────────────────────────────────────────────────
# PDF GENERATION — 3 Templates
# ─────────────────────────────────────────────────────────────────────────────

def _s(text: str) -> str:
    """Transliterate Unicode to Latin-1 safe ASCII for fpdf2."""
    m = {"\u2013":"-","\u2014":"--","\u2018":"'","\u2019":"'","\u201c":'"',
         "\u201d":'"',"\u2022":"-","\u2026":"...","\u00e9":"e","\u00e8":"e",
         "\u00ea":"e","\u00e0":"a","\u00e4":"a","\u00f6":"o","\u00fc":"u",
         "\u00b7":".","\u00a0":" ","\u2192":"->","\u2500":"-"}
    for k, v in m.items():
        text = text.replace(k, v)
    return text.encode("latin-1", errors="ignore").decode("latin-1")


def _c(text: str) -> str:
    """Collapse excess whitespace and strip stray markdown."""
    text = re.sub(r" {3,}", " ", text)
    text = re.sub(r"\*{1,2}(.+?)\*{1,2}", r"\1", text)
    text = re.sub(r"_{1,2}(.+?)_{1,2}",   r"\1", text)
    text = re.sub(r"^#{1,6}\s+", "",       text, flags=re.MULTILINE)
    return text.strip()


def _export(pdf: FPDF) -> bytes:
    raw = pdf.output()
    return bytes(raw) if isinstance(raw, (bytes, bytearray)) else BytesIO(raw).getvalue()


def _body(pdf: FPDF, content: str, W: float, left: float,
          col_dark: tuple, col_body: tuple, col_rule: tuple,
          font: str = "Helvetica") -> None:
    """
    Shared body text renderer used by all three templates.
    Three visual modes auto-detected per line:
      ALL-CAPS ≤50 chars   → bold sub-section heading + thin rule
      starts with "  - "   → indented bullet point
      everything else      → normal paragraph

    IMPORTANT: Never uses pdf.cell(..., ln=True) — that param is broken in fpdf2.
    All line-breaks use explicit pdf.ln(h) calls.
    """
    for line in _c(content).splitlines():
        raw = line.strip()
        if not raw:
            pdf.ln(2)
            continue

        is_head = (raw == raw.upper() and re.search(r"[A-Z]", raw)
                   and len(raw) <= 50 and not raw.startswith("-"))

        if is_head:
            pdf.ln(3)
            pdf.set_x(left)
            pdf.set_font(font, "B", 9.5)
            pdf.set_text_color(*col_dark)
            pdf.cell(w=W, h=6, txt=_s(raw))
            pdf.ln(6)
            pdf.set_draw_color(*col_rule)
            pdf.set_line_width(0.2)
            pdf.line(left, pdf.get_y(), left + W, pdf.get_y())
            pdf.ln(3)
            pdf.set_text_color(*col_body)

        elif raw.startswith("- "):
            pdf.set_font(font, "", 9.5)
            pdf.set_text_color(*col_body)
            pdf.set_x(left + 4)
            pdf.multi_cell(w=W - 4, h=5.5, txt=_s(raw))

        else:
            pdf.set_font(font, "", 9.5)
            pdf.set_text_color(*col_body)
            pdf.set_x(left)
            pdf.multi_cell(w=W, h=5.5, txt=_s(raw))

    pdf.ln(3)


def generate_pdf(name: str, email: str, phone: str, linkedin: str,
                 job_role: str, sections: dict, evaluation: dict,
                 template: str = "Modern") -> bytes:
    """
    Render a PDF resume. template must be one of: 'Modern', 'Classic', 'Minimal'.

    Modern  — dark navy header bar, blue accent underlines, Helvetica
    Classic — Times serif, full-width horizontal rules, centered name block
    Minimal — maximum whitespace, teal left-bar accents, restrained palette
    """
    if template == "Classic": return _pdf_classic(name, email, phone, linkedin, job_role, sections, evaluation)
    if template == "Minimal": return _pdf_minimal(name, email, phone, linkedin, job_role, sections, evaluation)
    return _pdf_modern(name, email, phone, linkedin, job_role, sections, evaluation)


# ── Modern ────────────────────────────────────────────────────────────────────
def _pdf_modern(name, email, phone, linkedin, job_role, sections, evaluation) -> bytes:
    DARK  = (22, 28, 54);  ACCENT = (67, 143, 232); BODY = (30, 35, 45)
    MUTED = (105, 115, 130); WHITE = (255, 255, 255); RULE = (210, 215, 228)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=22)
    pdf.add_page()
    pdf.set_margins(left=18, top=10, right=18)
    W = pdf.w - 36

    def header():
        pdf.set_fill_color(*DARK)
        pdf.rect(x=0, y=0, w=pdf.w, h=46, style="F")
        pdf.set_y(9); pdf.set_x(18)
        pdf.set_font("Helvetica", "B", 21); pdf.set_text_color(*WHITE)
        pdf.cell(w=W, h=10, txt=_s(name.upper() if name else "CANDIDATE")); pdf.ln(11)
        pdf.set_x(18); pdf.set_font("Helvetica", "B", 10); pdf.set_text_color(*ACCENT)
        pdf.cell(w=W, h=6, txt=_s(job_role.upper() if job_role else "")); pdf.ln(6)
        parts = [p.strip() for p in [email, phone, linkedin] if p and p.strip()]
        if parts:
            pdf.set_x(18); pdf.set_font("Helvetica", "", 8.5); pdf.set_text_color(180, 195, 220)
            pdf.cell(w=W, h=5, txt=_s("  |  ".join(parts))); pdf.ln(5)
        pdf.ln(7); pdf.set_text_color(*BODY)

    def sh(label: str):
        pdf.ln(4); pdf.set_x(18)
        pdf.set_font("Helvetica", "B", 10.5); pdf.set_text_color(*DARK)
        pdf.cell(w=W, h=7, txt=_s(label.upper())); pdf.ln(7)
        y = pdf.get_y() - 1
        pdf.set_draw_color(*ACCENT); pdf.set_line_width(0.45)
        pdf.line(18, y, 48, y); pdf.ln(3); pdf.set_text_color(*BODY)

    header()
    pdf.set_y(52)  # Adjust content start to below the 46h header bar
    if sections.get("summary", "").strip():
        sh("Professional Summary"); _body(pdf, sections["summary"], W, 18, DARK, BODY, RULE)
    if sections.get("resume", "").strip():
        _body(pdf, sections["resume"], W, 18, DARK, BODY, RULE)
    if sections.get("cover_letter", "").strip():
        pdf.add_page(); header(); pdf.set_y(52); sh("Cover Letter")
        _body(pdf, sections["cover_letter"], W, 18, DARK, BODY, RULE)

    sc = evaluation.get("total_score", 0) or 0; gr = evaluation.get("grade", "N/A")
    wc = evaluation.get("word_count", 0)
    pdf.set_y(pdf.h - 16); pdf.set_fill_color(*DARK)
    pdf.rect(0, pdf.h - 16, pdf.w, 16, "F")
    pdf.set_y(pdf.h - 12); pdf.set_x(18)
    pdf.set_font("Helvetica", "B", 8.5); pdf.set_text_color(*WHITE)
    pdf.cell(W, 5, _s(f"ATS Score: {sc}/100  |  {gr}  |  Words: {wc}"), align="C"); pdf.ln(5)
    pdf.set_x(18); pdf.set_font("Helvetica", "I", 7); pdf.set_text_color(165, 175, 195)
    pdf.cell(W, 5, "Generated by AI Resume & Career Suite  |  Powered by Groq LLaMA", align="C")
    return _export(pdf)


# ── Classic ───────────────────────────────────────────────────────────────────
def _pdf_classic(name, email, phone, linkedin, job_role, sections, evaluation) -> bytes:
    DARK = (15, 15, 15); GRAY = (75, 75, 75); LIGHT = (140, 140, 140)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=22)
    pdf.add_page(); pdf.set_margins(left=22, top=16, right=22); W = pdf.w - 44

    def header():
        pdf.set_y(16); pdf.set_font("Times", "B", 22); pdf.set_text_color(*DARK)
        pdf.set_x(22); pdf.cell(W, 11, _s(name.upper() if name else "CANDIDATE"), align="C"); pdf.ln(11)
        pdf.set_x(22); pdf.set_font("Times", "I", 10); pdf.set_text_color(*GRAY)
        pdf.cell(W, 6, _s(job_role if job_role else ""), align="C"); pdf.ln(6)
        parts = [p.strip() for p in [email, phone, linkedin] if p and p.strip()]
        if parts:
            pdf.set_x(22); pdf.set_font("Times", "", 9); pdf.set_text_color(*LIGHT)
            pdf.cell(W, 5, _s("  ·  ".join(parts)), align="C"); pdf.ln(5)
        pdf.ln(3); pdf.set_draw_color(*DARK); pdf.set_line_width(0.7)
        pdf.line(22, pdf.get_y(), 22 + W, pdf.get_y()); pdf.ln(6); pdf.set_text_color(*DARK)

    def sh(label: str):
        pdf.ln(4); pdf.set_x(22); pdf.set_font("Times", "B", 11); pdf.set_text_color(*DARK)
        pdf.cell(W, 7, _s(label.upper())); pdf.ln(7)
        pdf.set_draw_color(*DARK); pdf.set_line_width(0.3)
        pdf.line(22, pdf.get_y(), 22 + W, pdf.get_y()); pdf.ln(4); pdf.set_text_color(*DARK)

    header()
    if sections.get("summary", "").strip():
        sh("Professional Summary"); _body(pdf, sections["summary"], W, 22, DARK, GRAY, LIGHT, "Times")
    if sections.get("resume", "").strip():
        _body(pdf, sections["resume"], W, 22, DARK, GRAY, LIGHT, "Times")
    if sections.get("cover_letter", "").strip():
        pdf.add_page(); header(); sh("Cover Letter")
        _body(pdf, sections["cover_letter"], W, 22, DARK, GRAY, LIGHT, "Times")

    sc = evaluation.get("total_score", 0) or 0; gr = evaluation.get("grade", "N/A")
    pdf.set_y(pdf.h - 12)
    pdf.set_draw_color(*LIGHT); pdf.set_line_width(0.3)
    pdf.line(22, pdf.get_y(), 22 + W, pdf.get_y()); pdf.ln(3); pdf.set_x(22)
    pdf.set_font("Times", "I", 8); pdf.set_text_color(*LIGHT)
    pdf.cell(W, 5, _s(f"Resume Strength: {sc}/100  |  {gr}  |  AI Resume Suite"), align="C")
    return _export(pdf)


# ── Minimal ───────────────────────────────────────────────────────────────────
def _pdf_minimal(name, email, phone, linkedin, job_role, sections, evaluation) -> bytes:
    DARK  = (12, 12, 12); TEAL = (20, 184, 166)
    BODY  = (55, 65, 75); LIGHT = (150, 160, 170); RULE = (220, 225, 230)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=28)
    pdf.add_page(); pdf.set_margins(left=24, top=20, right=24); W = pdf.w - 48

    def header():
        pdf.set_y(20); pdf.set_x(24); pdf.set_font("Helvetica", "B", 19); pdf.set_text_color(*DARK)
        pdf.cell(W, 10, _s(name if name else "Candidate")); pdf.ln(10)
        pdf.set_x(24); pdf.set_font("Helvetica", "", 10); pdf.set_text_color(*TEAL)
        pdf.cell(W, 6, _s(job_role if job_role else "")); pdf.ln(6)
        parts = [p.strip() for p in [email, phone, linkedin] if p and p.strip()]
        if parts:
            pdf.set_x(24); pdf.set_font("Helvetica", "", 8); pdf.set_text_color(*LIGHT)
            pdf.cell(W, 5, _s("  ·  ".join(parts))); pdf.ln(5)
        pdf.ln(8)
        pdf.set_draw_color(*TEAL); pdf.set_line_width(1.2)
        pdf.line(24, pdf.get_y(), 34, pdf.get_y())
        pdf.set_draw_color(*RULE); pdf.set_line_width(0.25)
        pdf.line(36, pdf.get_y(), 24 + W, pdf.get_y())
        pdf.ln(9); pdf.set_text_color(*BODY)

    def sh(label: str):
        pdf.ln(6); y_now = pdf.get_y()
        pdf.set_fill_color(*TEAL); pdf.rect(24, y_now, 2.5, 6.5, "F")
        pdf.set_x(28.5); pdf.set_font("Helvetica", "B", 9.5); pdf.set_text_color(*DARK)
        pdf.cell(W - 4.5, 6.5, _s(label.upper())); pdf.ln(8); pdf.set_text_color(*BODY)

    header()
    if sections.get("summary", "").strip():
        sh("About"); _body(pdf, sections["summary"], W, 24, DARK, BODY, RULE)
    if sections.get("resume", "").strip():
        _body(pdf, sections["resume"], W, 24, DARK, BODY, RULE)
    if sections.get("cover_letter", "").strip():
        pdf.add_page(); header(); sh("Cover Letter")
        _body(pdf, sections["cover_letter"], W, 24, DARK, BODY, RULE)

    sc = evaluation.get("total_score", 0) or 0; gr = evaluation.get("grade", "N/A")
    pdf.set_y(pdf.h - 11); pdf.set_x(24); pdf.set_font("Helvetica", "", 7.5)
    pdf.set_text_color(*LIGHT)
    pdf.cell(W, 5, _s(f"ATS Score {sc}/100  ·  {gr}  ·  AI Resume Suite"), align="C")
    return _export(pdf)