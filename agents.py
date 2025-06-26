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
    






