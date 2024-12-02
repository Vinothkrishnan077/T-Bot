import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import pandas as pd
from docx import Document
from datetime import datetime
import re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from summa import summarizer as textrank_summarizer
from gtts import gTTS
import streamlit_webrtc as webrtc
import tempfile
import speech_recognition as sr



# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Load spaCy NER model
nlp = spacy.load("en_core_web_sm")

# Function to extract text from PDF
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to extract text from Excel
def get_excel_text(excel_docs):
    text = ""
    for excel in excel_docs:
        xlsx = pd.ExcelFile(excel)
        for sheet_name in xlsx.sheet_names:
            df = pd.read_excel(excel, sheet_name=sheet_name)
            text += f"Sheet: {sheet_name}\n{df.to_json(index=False)}\n"
    return text

# Function to extract text from Word
def get_word_text(word_docs):
    text = ""
    for word in word_docs:
        doc = Document(word)
        for paragraph in doc.paragraphs:
            text += paragraph.text
    return text

# Function to clean and preprocess text
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\x0c', ' ', text)  # Handle form feed character in PDFs
    return text.strip()

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create and save vector store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to get conversational chain
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in the provided context, say, "answer is not available in the context". Do not provide an incorrect answer.

    Context:
    {context}

    Question: 
    {question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to save chat response to a file
def save_response(question, response):
    filename = "chat_responses.txt"
    with open(filename, "a", encoding="utf-8") as file:  # Specify 'utf-8' encoding
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        file.write(f"Timestamp: {timestamp}\n")
        file.write(f"Question: {question}\n")
        file.write(f"Response: {response}\n")
        file.write("\n" + "=" * 50 + "\n\n")

# Function to handle user input and generate response
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply: ", response["output_text"])
    save_response(user_question, response["output_text"])
    play_text(response["output_text"])

# NER function using spaCy
def get_ner(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

# Function to play text using gTTS
def play_text(text):
    tts = gTTS(text)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
        tts.save(temp_audio.name)
        st.audio(temp_audio.name, format="audio/mp3")

# Function to recognize speech using microphone
def recognize_speech():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    with mic as source:
        st.info("Listening...")
        audio = recognizer.listen(source)
    try:
        st.success("Recognized!")
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        st.error("Could not understand the audio")
    except sr.RequestError:
        st.error("Request Error from Google Speech Recognition")
    return ""

# Main function
def main():
    st.set_page_config(page_title="Chat with Documents")
    st.header("T-1 Bot🔎")

    # User input through text or voice
    user_question = st.text_area("Ask a Question from the Documents")
    if st.button("Submit Question"):
        if user_question:
            user_input(user_question)

    st.write("OR")

    if st.button("Use Voice Input"):
        user_question = recognize_speech()
        if user_question:
            st.write(f"You asked: {user_question}")
            user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        doc_files = st.file_uploader("Upload your PDF, Excel, and Word Files", accept_multiple_files=True, type=['pdf', 'xlsx', 'xls', 'docx'])

        if st.button("Submit & Process Documents"):
            if not doc_files:
                st.error("Please upload at least one PDF, Excel, or Word file before submitting.")
            else:
                with st.spinner("Processing..."):
                    raw_text = ""
                    pdf_docs = [doc for doc in doc_files if doc.name.endswith('.pdf')]
                    excel_docs = [doc for doc in doc_files if doc.name.endswith(('.xlsx', '.xls'))]
                    word_docs = [doc for doc in doc_files if doc.name.endswith('.docx')]

                    # Extract text from uploaded documents
                    if pdf_docs:
                        raw_text += get_pdf_text(pdf_docs)
                    if excel_docs:
                        raw_text += get_excel_text(excel_docs)
                    if word_docs:
                        raw_text += get_word_text(word_docs)

                    # Clean and preprocess text
                    raw_text = clean_text(raw_text)

                    # Split text into chunks
                    text_chunks = get_text_chunks(raw_text)

                    # Create and save vector store
                    get_vector_store(text_chunks)
                    st.success("Documents Uploaded...")

if __name__ == "__main__":
    main()
