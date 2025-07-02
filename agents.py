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
        """Analyze skills using the RAG chain."""
        query = f"On a scale of 0-10, how clearly does the candidate mention proficiency in {skill}? Provide a numeric rating first, followed by resoning"
        response = qa_chain.invoke({"input": query})
        match = re.search(r'(\d+)', response['answer'])
        score = int(match.group(1) if match else 0)
        reasoning = response['answer'].split('\n', 1)[1].strip() if '.' in response['answer'] and len(response['answer'].split('.')) > 1 else ""
        return score, min(score, 10), reasoning



    






