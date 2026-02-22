# 🤖 AI-Powered Resume & Cover Letter Generator

> **Capstone Project** | AI/ML | Python · Streamlit · OpenAI GPT

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red?logo=streamlit)](https://streamlit.io)
[![Groq](https://img.shields.io/badge/Groq-LLaMA%203-orange?logo=meta)](https://console.groq.com)

---

## 📋 Academic Structure (Capstone)

| # | Section | Coverage |
|---|---------|----------|
| 1 | Problem Statement | Resume writing gap for students & freshers |
| 2 | Proposed Solution | Groq LLaMA-powered generation with Prompt Engineering |
| 3 | System Dev Approach | Modular Python + Streamlit + Groq API |
| 4 | Algorithm & Deployment | Transformer LLM · ATS scoring heuristics |
| 5 | Result | Generated resume, cover letter & strength score |
| 6 | Conclusion | Context-aware AI > template-based tools |
| 7 | Future Scope | Fine-tuning, PDF export, BERT scoring, RAG |
| 8 | References | Vaswani 2017, Brown 2020, OpenAI Docs |

---

## 🚀 Features

- ✅ **ATS-Optimized Resume** — Structured, keyword-rich format
- ✅ **Tailored Cover Letter** — Role-specific, professional tone
- ✅ **Professional Summary** — 3–4 line elevator pitch
- ✅ **Resume Strength Score** — Rule-based NLP evaluation (0–100)
- ✅ **Action Verb Detection** — NLP feature extraction
- ✅ **Skill Keyword Matching** — Precision-style relevance scoring
- ✅ **Download as .txt** — One-click export
- ✅ **Dark-Mode UI** — Professional Streamlit interface
- ✅ **Error Handling** — Missing API key & empty field validation

---

## 🏗️ Project Structure

```
AI Resume Builder/
├── app.py              ← Main Streamlit application
├── utils.py            ← Modular AI/ML helper functions
├── requirements.txt    ← Python dependencies
├── .env.example        ← Environment variable template
├── .env                ← Your actual API key (DO NOT COMMIT)
└── README.md           ← This file
```

---

## ⚙️ Local Setup & Run

### Step 1 — Clone / navigate to the project folder
```bash
cd "c:\ALL Codes\AI Resume Builder"
```

### Step 2 — Create & activate virtual environment
```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

> **No billing required.** This project uses the **Groq API** (free tier) instead of OpenAI.
> Get a free key at [console.groq.com](https://console.groq.com) — sign up with Google/GitHub.

### Step 4 — Configure API key
```bash
copy .env.example .env
# Open .env and replace GROQ_API_KEY with your real key from console.groq.com
```
Or simply enter your key directly in the app's UI the first time you run it.

### Step 5 — Run the app
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## ☁️ Streamlit Cloud Deployment

1. **Push to GitHub** — Upload all files **except `.env`**
2. **Go to** [share.streamlit.io](https://share.streamlit.io)
3. **Connect your repository** and set main file to `app.py`
4. **Add Secret:**
   - Under *Settings → Secrets*, add:
     ```toml
     GROQ_API_KEY = "gsk_your-real-groq-api-key"
     ```
5. **Click Deploy** — Your app will be live in ~2 minutes!

> **Note:** `python-dotenv` is only needed for local `.env` file loading.  
> Streamlit Cloud injects secrets as environment variables automatically.

---

## 🔬 AI/ML Concepts Used

| Concept | Application |
|---------|-------------|
| **Transformer LLM** | LLaMA 3 generates coherent, role-specific documents |
| **Prompt Engineering** | System + user prompts structure model output |
| **NLP Tokenization** | Input fields tokenized for model processing |
| **Context-Aware Generation** | All profile fields influence the generated text |
| **Action Verb Extraction** | Regex-based NLP feature engineering |
| **Keyword Matching** | Set-intersection relevance scoring (TF-inspired) |
| **Zero-Shot Generalization** | No custom training data required |

---

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|--------|
| `streamlit` | 1.32.2 | Web UI framework |
| `groq` | ≥ 0.9.0 | Groq API client — free LLaMA 3 / Mixtral inference |
| `python-dotenv` | 1.0.1 | `.env` file loader |

---

## 📚 References

1. Vaswani, A. et al. (2017). *Attention Is All You Need*. NeurIPS. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
2. Brown, T. et al. (2020). *GPT-3: Language Models are Few-Shot Learners*. [arXiv:2005.14165](https://arxiv.org/abs/2005.14165)
3. OpenAI. (2024). *API Reference*. https://platform.openai.com/docs
4. Streamlit Inc. (2024). *Streamlit Docs*. https://docs.streamlit.io

---

*Built with ❤️ using Python, Streamlit & OpenAI GPT — Academic Capstone Project*
