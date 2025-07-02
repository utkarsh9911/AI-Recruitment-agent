import re
import PyPDF2
import io
import google.generativeai as genaiI
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from concurrent.futures import ThreadPoolExecutor
import os
import tempfile
import json

class ResumeAnalysisAgent:
    def __init__(self, api_key, cutoff_score=75):
        self.api_key = api_key
        self.cutoff_score = cutoff_score
        self.resume_texts = None
        self.rag_vectorstore = None
        self.analysis_result= None
        self.jd_text= None
        self.extracted_skills = None
        self.resume_weaknesses = None
        self.resume_strengths = []
        self.imporvement_suggestions = {}
    
    def extract_text_from_pdf(self, pdf_file):
        """Extract text from a PDF file."""
        try:
            if hasattr(pdf_file, 'getvalue'):
                pdf_data = pdf_file.getvalue()
                pdf_file = io.BytesIO(pdf_data)
                reader = PyPDF2.PdfReader(pdf_file)
            else:
                reader = PyPDF2.PdfReader(pdf_file)
            
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            return text
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            return ""
    
    def extract_text_from_text(self, txt_file):
        """Extract text from a text file."""
        try:
            if hasattr(txt_file, 'getvalue'):
                return txt_file.getvalue().decode('utf-8')
            else:
                with open(txt_file, 'r', encoding='utf-8') as file:
                    return file.read()
        except Exception as e:
            print(f"Error extracting text from text file: {e}")
            return ""
        
    def extract_text_from_files(self, file):
        """Extract text from a list of files."""
        if hasattr(file, 'name'):
            file_extension = file.name.split('.')[-1].lower()
        else:
            file_extension = file.split('.')[-1].lower()

        if file_extension == 'pdf':
            return self.extract_text_from_pdf(file)
        elif file_extension == 'txt':
            return self.extract_text_from_text(file)
        else:
            print(f"Unsupported file type: {file_extension}")
            return ""
        
    def create_rag_vector_stor(self, text):
        """create a vecot stor for RAG"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        embeddings = GoogleGenerativeAIEmbeddings(model_name="models/embedding-001", api_key= self.api_key)
        vectorstore = FAISS.from_texts(chunks, embeddings)
        return vectorstore

    def analyze_skills(self, qa_chain, skill):
        
        """Analyze a single skill using the RAG chain and return skill, score, and cleaned reasoning."""
        
        # Ask the chain for evaluation of the skill
        query = f"On a scale of 0-10, how clearly does the candidate mention proficiency in {skill}? Provide a numeric rating first, followed by reasoning"
        response = qa_chain.invoke({"input": query})

        # Extract numeric score from the response
        match = re.search(r'(\d+)', response['answer'])
        score = int(match.group(1)) if match else 0
        final_score = min(score, 10)

        # Clean up the reasoning part
        reasoning_raw = response['answer']
        reasoning_lines = [line.strip() for line in reasoning_raw.split('\n') if line.strip()]
        raw_reasoning = " ".join(reasoning_lines)

        # Remove leading score and optional symbols
        reasoning = re.sub(r'^\d+\s*[-.:]?\s*', '', raw_reasoning)

        return skill, final_score, reasoning
    
    # This is not a final code need some changes
    def analyze_resume_weaknesses(self):
        """Analyze specific weaknesses in the resume based on missing skills"""
        if not self.resume_text or not self.extracted_skills or not self.analysis_result:
            return []
        
        weaknesses  = []
        for skill in analysis_result.get('missing_skill', []):
    prompt = f"""
    Analyze why the resume is weak in demonstrating proficiency in "{skill}".

    For your analysis, consider:
    1. What's missing from the resume regarding this skill?
    2. How could it be improved with specific examples?
    3. What specific action items would make this skill stand out?

    Provide your response in this JSON format:
    {{
        "weakness": "A concise description of what's missing or problematic (1-2 sentences)",
        "improvement_suggestions": [
            "Specific suggestion 1",
            "Specific suggestion 2",
            "Specific suggestion 3"
        ],
        "example_addition": "A specific bullet point that could be added to showcase this skill"
    }}

    Return only valid JSON, no other text.
    """
    
    response = ra_chain.invoke({"input": prompt})
    
    # ✅ Extract string answer
    raw_json = response['answer']

    # ✅ Remove backticks and 'json' label
    cleaned_json = re.sub(r'^```json|```$', '', raw_json.strip(), flags=re.MULTILINE).strip()

    try:
        # ✅ Parse string into dictionary
        weakness_data = json.loads(cleaned_json)

        # ✅ Store in desired format
        weakness_detail = {
            "skill": skill,
            "detail": weakness_data.get("weakness", "No specific details provided."),
            "suggestions": weakness_data.get("improvement_suggestions", []),
            "example": weakness_data.get("example_addition", "")
        }

        weaknesses.append(weakness_detail)
        improvement_suggestions[skill] = {
                    "suggestions": weakness_data.get("improvement_suggestions", []),
                    "example": weakness_data.get("example_addition", "")}

    except json.JSONDecodeError as e:
        print(f"JSON parsing failed for skill {skill}: {e}")
        weaknesses.append({
            "skill": skill,
            "detail": raw_json[:200],  # fallback: first 200 characters
            
            "example": ""
        })


    

    






