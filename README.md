# Research Novelty Detector

A modular system that analyzes an uploaded research paper (PDF), extracts key sections, identifies closely related existing research via scoped web search, extracts source content, ranks them locally using semantic embeddings, and provides evidence-based research comparisons and novelty insights using Groq LLM.

---

## 🚀 Features

- **Local PDF Processing**: PyMuPDF-based text extraction with page preservation, scanned-document detection, and OCR-needed alerting.
- **Rule-Based Section Extractor**: Extracts Abstract, Introduction, Problem Statement, Objectives, Literature Review, Methodology, Results, Contributions, and Conclusion with heading hierarchy awareness.
- **KeyBERT Academic Query Generator**: Generates focused, noun-phrase queries locally (Title, Problem, Methodology) without hallucinated keywords.
- **Targeted Tavily Web Search**: Basic search restricted to scholarly domains with disk caching and URL deduplication (normalizes arXiv, DOIs).
- **Multi-Level Content Extractor**: Extracts full-text PDFs/HTML or meta-tag abstracts; gracefully handles paywalls, timeouts, and robots.txt.
- **Local Sentence-Transformers**: Uses `all-MiniLM-L6-v2` locally for document and multi-aspect similarity (Problem, Methodology, Contributions, Objectives).
- **Single Groq LLM Comparison**: Uses 1 call strictly grounded in retrieved evidence with citation validation (`[P1]`, `[P2]`) and explicit warnings when evidence is insufficient.
- **Dual Interface**: Full CLI tool (`main.py`) and Streamlit web dashboard (`app.py`).

---

## 🛠️ Setup & Installation

### 1. Prerequisites
- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (recommended) or standard `pip`

### 2. Install Dependencies
```bash
uv sync
# or using pip:
pip install -r requirements.txt
```

### 3. Configure API Keys
Copy `.env.example` to `.env` and fill in your keys:
```bash
cp .env.example .env
```
In `.env`:
```env
TAVILY_API_KEY=tvly-...
GROQ_API_KEY=gsk_...
GROQ_MODEL=openai/gpt-oss-20b
```

---

## 🏃 Running the Application

### Option A: Streamlit Web UI
```bash
uv run streamlit run app.py
```
Open your browser at `http://localhost:8501`. You can upload any paper or load the included `sample.pdf`.

### Option B: Command Line (CLI)
Full end-to-end pipeline:
```bash
uv run python main.py data/uploaded_papers/sample.pdf
```

Local-only run (extracts text, sections & queries without API usage):
```bash
uv run python main.py data/uploaded_papers/sample.pdf --local-only
```

Run search & local ranking, skipping the Groq comparison:
```bash
uv run python main.py data/uploaded_papers/sample.pdf --no-llm
```

---

## 🧪 Testing

Execute the comprehensive test suite (all 73 unit tests mock external services and run locally):
```bash
uv run pytest
```

## Output
<img width="1862" height="838" alt="image" src="https://github.com/user-attachments/assets/a74ed5b8-6d2f-485c-a32d-9216015decc7" />
<img width="1917" height="912" alt="image" src="https://github.com/user-attachments/assets/9f31b760-5046-42e4-9a4a-26fc7f985447" />
Demo :- https://research-novelty-detector.streamlit.app/
