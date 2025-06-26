import os
from langchain_community.document_loaders import PyPDFLoader # Added for PDF loading
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_core.documents import Document # Explicitly import Document
from dotenv import load_dotenv

# Load environment variables (for API key)
load_dotenv()

# --- Configuration ---
# Replace with your actual Google API Key
# It's highly recommended to set this as an environment variable (e.g., GOOGLE_API_KEY)
# and load it using os.getenv("GOOGLE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "YOUR_GOOGLE_API_KEY") # Replace YOUR_GOOGLE_API_KEY if not using .env file

if GOOGLE_API_KEY == "YOUR_GOOGLE_API_KEY":
    print("WARNING: Please replace 'YOUR_GOOGLE_API_KEY' with your actual Google API key or set the GOOGLE_API_KEY environment variable.")
    print("You can get an API key from Google AI Studio: https://aistudio.google.com/app/apikey")


# --- Step 1: Load Resume Content from PDF ---
# Prompt the user for the PDF file path
pdf_path = input("Please enter the full path to your resume PDF file: ")

try:
    # Initialize PyPDFLoader with the provided PDF path
    loader = PyPDFLoader(pdf_path)
    # Load the pages from the PDF as a list of Document objects
    documents = loader.load()
    print(f"Successfully loaded {len(documents)} pages from '{pdf_path}'.")
except FileNotFoundError:
    print(f"Error: The file '{pdf_path}' was not found. Please check the path and try again.")
    exit() # Exit if file not found
except Exception as e:
    print(f"An error occurred while loading the PDF: {e}")
    exit() # Exit on other PDF loading errors


# --- Step 2: Split the Resume Text into Chunks ---
# We use RecursiveCharacterTextSplitter for robust text splitting
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # Maximum size of each chunk
    chunk_overlap=200,    # Overlap between chunks to maintain context
    length_function=len,  # Function to measure chunk length
    is_separator_regex=False, # Use standard separators
)

# Split the loaded PDF documents into smaller chunks
docs = text_splitter.split_documents(documents)

print(f"Split resume into {len(docs)} chunks.")
# for i, doc in enumerate(docs):
#     print(f"\n--- Chunk {i+1} ---")
#     print(doc.page_content[:200] + "...") # Print first 200 chars of each chunk


# --- Step 3 & 4: Create Embeddings and Build Vector Store (FAISS) ---
# Initialize Google Generative AI Embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)

print("Creating vector store...")
# Create a FAISS vector store from the document chunks and embeddings
# This step might take a moment depending on the size of the resume
vector_store = FAISS.from_documents(docs, embeddings)
print("Vector store created successfully.")

# Define the retriever from the vector store
retriever = vector_store.as_retriever()

# --- Step 5: Set up the RAG Chain ---
# Initialize the Google Generative AI Chat Model (gemini-2.0-flash)
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY)

# Define the prompt template for the RAG chain
# This prompt guides the LLM on how to use the retrieved context
prompt = ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {input}
""")

# Create a chain that combines the retrieved documents with the user's question
document_chain = create_stuff_documents_chain(llm, prompt)

# Create the full retrieval chain that first retrieves relevant documents
# and then passes them to the document_chain for answering
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# --- Step 6: Ask Questions and Get Answers ---
print("\n--- RAG Project Ready ---")
print("You can now ask questions about the resume. Type 'exit' to quit.")

while True:
    user_question = input("\nYour question: ")
    if user_question.lower() == 'exit':
        break

    try:
        # Invoke the retrieval chain with the user's question
        print("Thinking...")
        response = retrieval_chain.invoke({"input": user_question})

        # The response structure contains 'answer' and 'context' (retrieved documents)
        print("\nAnswer:")
        print(response["answer"])

        # Optionally, print the sources used to generate the answer
        # print("\nSources (chunks used):")
        # for i, doc in enumerate(response["context"]):
        #     print(f"--- Document {i+1} ---")
        #     print(doc.page_content[:150] + "...") # Print first 150 chars of context
        #     print("-" * 20)

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure your Google API Key is valid and has access to Gemini models.")

print("\nThank you for using the resume RAG assistant!")
