# 🧠 AI Recruitment Agent (Resume Analyzer & Interview Assistant)

An end-to-end AI-powered recruitment assistant that helps users optimize their resumes, assess ATS (Applicant Tracking System) compatibility, generate interview questions, and tailor resumes based on job descriptions using Retrieval-Augmented Generation (RAG) and Gemini.

---

## 🚀 Features

### ✅ Resume Analysis
- Upload your resume and get:
  - **ATS Score** (based on formatting, keyword relevance, readability)
  - **Skill Gap Identification**
  - **Strengths and Weaknesses**
  - **Improvement Suggestions** to increase selection chances

### 🧠 Intelligent Q&A
- Ask questions **about your own resume**
  - e.g., “What are my top technical skills?” or “Does my resume suit a data analyst role?”
- Ask **general career/resume advice**
  - e.g., “What should I add to improve project descriptions?”

### 💬 Interview Question Generator
- Get **custom interview questions** based on:
  - Your resume
  - Difficulty level (Easy, Medium, Hard)
  - Question tags like `Technical`, `Behavioral`, `Situational`, etc.

### 🎯 Resume Optimization for Job Descriptions
- Upload or paste a **job description**
- The agent will:
  - Match the job with your resume
  - Highlight gaps
  - Suggest changes
  - Provide an **Improved Resume Section** rewritten for better alignment with the job

---

## 🧪 Tech Stack

- **Python 3.11**
- **LLM APIs**: Gemini / GPT (used via RAG)
- **NLP**: SpaCy, Transformers
- **Vector DB**: FAISS (or equivalent for retrieval)
- **Deployment**: AWS (e.g., EC2, S3), Docker
- **Other Tools**: Pandas, Scikit-learn, LangChain, Streamlit, Gemini


---

## 🚀 How to Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/utkarsh9911/ai-recruitment-agent.git
cd ai-recruitment-agent

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run Streamlit app
streamlit run streamlit_app/app.py
