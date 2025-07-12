import re
import PyPDF2
import io
import google.generativeai as genaiI
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from concurrent.futures import ThreadPoolExecutor
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import os
import tempfile
import json



class ResumeAnalysisAgent:
    def __init__(self, api_key, cutoff_score=75):
        self.api_key = api_key
        self.cutoff_score = cutoff_score
        self.resume_text = None
        self.rag_vectorstore = None
        self.analysis_result= None
        self.jd_text= None
        self.extracted_skills = None
        self.resume_weaknesses = None
        self.resume_strengths = []
        self.improvement_suggestions = {}
    
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
        
    def extract_text_from_file(self, file):
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
        
    def create_rag_vector_store(self, text):
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
    
    def create_vector_store(text):
        """Create a simpler vector store for skill analysis"""
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEYY)
        text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,      # Maximum size of each chunk
        chunk_overlap=200,    # Overlap between chunks to maintain context
        length_function=len ) # Function to measure chunk length
    
        chunks = text_splitter.split_text(text)
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
        # final_score = min(score, 10)

        # Clean up the reasoning part
        reasoning_raw = response['answer']
        reasoning_lines = [line.strip() for line in reasoning_raw.split('\n') if line.strip()]
        raw_reasoning = " ".join(reasoning_lines)

        # Remove leading score and optional symbols
        reasoning = re.sub(r'^\d+\s*[-.:]?\s*', '', raw_reasoning)

        return skill, min(score, 10), reasoning
    
    # This is not a final code need some changes
    def analyze_resume_weaknesses(self):
        """Analyze specific weaknesses in the resume based on missing skills"""
        if not self.resume_text or not self.extracted_skills or not self.analysis_result:
            return []
        
        weaknesses  = []
        for skill in self.analysis_result.get('missing_skill', []):
            prompt = f"""
            Analyze why the resume is weak in demonstrating proficiency in "{skill}".

            For your analysis, consider:
            1. What's missing from the resume regarding this skill?
            2. How could it be improved with specific examples?
            3. What specific action items would make this skill stand out?

            Resume Content:
            {self.resume_text[:3000]}...

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
            

            llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=self.api_key)

            response = llm.invoke(prompt)
    
            #  Extract string answer
            raw_json = response.content

            #  Remove backticks and 'json' label
            cleaned_json = re.sub(r'^```json|```$', '', raw_json.strip(), flags=re.MULTILINE).strip()

            try:
                #  Parse string into dictionary
                weakness_data = json.loads(cleaned_json)

                # Store in desired format
                weakness_detail = {
                "skill": skill,
                "detail": weakness_data.get("weakness", "No specific details provided."),
                "suggestions": weakness_data.get("improvement_suggestions", []),
                "example": weakness_data.get("example_addition", "")
                }

                weaknesses.append(weakness_detail)
                self.improvement_suggestions[skill] = {
                    "suggestions": weakness_data.get("improvement_suggestions", []),
                    "example": weakness_data.get("example_addition", "")}

            except json.JSONDecodeError as e:
            
                weaknesses.append({
                "skill": skill,
                "detail": raw_json[:200],  # fallback: first 200 characters
            
                "example": ""
            })
        self.resume_weaknesses = weaknesses
        return weaknesses
    
    def extract_skills_from_jd(self, jd_text):
        """Extract skills from Job description"""
        try:
            llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=self.api_key)
            prompt = f"""
            Extract a comprehensive list of technical skills, technologies, and competencies required from this job description. 
            Format the output as a Python list of strings. Only include the list, nothing else.
            
            Job Description:
            {jd_text}
            """
            response = llm.invoke(prompt)
            skills_text = response.content

            cleaned = re.sub(r"```python|```", "", skills_text).strip()
            
            skills = eval(cleaned)
            if isinstance(skills, list):
                return skills
        except Exception as e:
            print(f"Error extracting skills from job description: {e}")
            return []

        

    def semantic_skill_analysis(self, resume_text, skills):
        """Analyze skills semantically"""
        vectorstore = self.create_vector_store(resume_text)
        retriever = vectorstore.as_retriever()
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=self.api_key)
        prompt = prompt = ChatPromptTemplate.from_template("""
        Answer the following question based only on the provided context.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        Context:
        {context}

        Question: {input} 
        """)

        document_chain = create_stuff_documents_chain(llm, prompt)
        ra_chain = create_retrieval_chain(retriever, document_chain)

        with ThreadPoolExecutor(max_workers=5) as executor:
            result = executor.map(lambda skill : self.analyze_skills(ra_chain, skill), skills)
        

        skill_scores = {}
        skill_reasoning = {}
        missing_skills = []
        total_score = 0

        for skill, score, reasoning in result:
            skill_scores[skill] = score
            skill_reasoning[skill] = reasoning
            total_score+= score
            if score<=5:
                missing_skills.append(skill)
            
        overall_score =  int((total_score / (10 * len(skills))) * 100)
        selected = overall_score >= self.cutoff_score

        reasoning = "Candidate evaluated based on explicit resume content using semantic similarity and clear numeric scoring."
        strengths = [skill for skill, score in skill_scores.items() if score >= 7]
        improvement_areas = missing_skills if not selected else []  

        self.resume_strengths = strengths
        return {
        "overall_score": overall_score,
            "skill_scores": skill_scores,
            "skill_reasoning": skill_reasoning,
            "selected": selected,
            "reasoning": reasoning,
            "missing_skills": missing_skills,
            "strengths": strengths,
            "improvement_areas": improvement_areas           
        }   
    
    def analyze_resume(self, resume_file, role_requirements=None, custom_jd = None):
        """Analyze a resume against role requirements or a custom JD"""
        self.resume_text = self.extract_text_from_file(resume_file)

        with tempfile.NamedTemporaryFile(delete=False, suffix='.txt', mode='w', encoding='utf-8') as tmp:
            tmp.write(self.resume_text)
            self.resume_file_path = tmp.name
        
        self.rag_vectorstore = self.create_rag_vector_store(self.resume_text)

        if custom_jd:
            self.jd_text = self.extract_text_from_file(custom_jd)
            self.extracted_skills = self.extract_skills_from_jd(self.jd_text)
        
            self.analysis_result = self.semantic_skill_analysis(self.resume_text, self.extracted_skills)

        elif role_requirements:
            self.extracted_skills = role_requirements
            self.analysis_result = self.semantic_skill_analysis(self.resume_text, role_requirements)
        
        if self.analysis_result and "missing_skills" in self.analysis_result and self.analysis_result["missing_skills"]:
            self.analyze_resume_weaknesses()
     
            self.analysis_result["detailed_weaknesses"] = self.resume_weaknesses
        
        return self.analysis_result


    def ask_question(self, question):
        """Ask question regarding to resume"""
        if not self.rag_vectorstore or not self.resume_text:
            return "Please analyze resume first."
        retriever = self.rag_vectorstore.as_retriever(
            search_kwargs={"k": 3}  
        )
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=self.api_key)
        prompt = ChatPromptTemplate.from_template("""
        Answer the following question based only on the provided context.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        Context:
        {context}

        Question: {input} 
        """)

        document_chain = create_stuff_documents_chain(llm, prompt)
        qa_chain = create_retrieval_chain(retriever, document_chain)
        response1 = qa_chain.invoke({"input":question})
        response = response1['answer']
        return response
        

    
    def generate_interview_questions(self, question_types, difficulty, num_questions):
        """Generate interview questions based on the resume"""
        if not self.resume_text or not self.extracted_skills:
            return []
        
        try:
            llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=self.api_key)
            
        
            context = f"""
            Resume Content:
            {self.resume_text[:2000]}...
            
            Skills to focus on: {', '.join(self.extracted_skills)}
            
            Strengths: {', '.join(self.analysis_result.get('strengths', []))}
            
            Areas for improvement: {', '.join(self.analysis_result.get('missing_skills', []))}
            """
            
            prompt = f"""
            Generate {num_questions} personalized {difficulty.lower()} level interview questions for this candidate 
            based on their resume and skills. Include only the following question types: {', '.join(question_types)}.
            
            For each question:
            1. Clearly label the question type
            2. Make the question specific to their background and skills
            3. For coding questions, include a clear problem statement
            
            {context}
            
            Format the response as a list of tuples with the question type and the question itself.
            Each tuple should be in the format: ("Question Type", "Full Question Text")
            """
            
            response = llm.invoke(prompt)
            questions_text = response.content

            questions = []
            questions_text = response.content

            # Step 1: Remove any extra explanation text before the list
            match = re.search(r"\[\s*\(", questions_text, re.DOTALL)
            if match:
                start_index = match.start()
                questions_text = questions_text[start_index:]
 
            # Step 2: Clean up Markdown or Python syntax
            questions_text = re.sub(r"```(?:python)?|```", "", questions_text).strip()

            # Step 3: Safely evaluate the string into a Python list
        
            questions_list = eval(questions_text)
            
            for question_type, question in questions_list:
                for requested_type in question_types:
                    if requested_type.lower() in question_type.lower():
                        questions.append((question_type.strip(), question.strip()))
                        break
            return questions
        except Exception as e:
            print(f"Parsing failed: {e}")
            return []




            







