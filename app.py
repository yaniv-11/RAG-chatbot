# app.py
import os
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from operator import itemgetter

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpointEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnableParallel

# Load Hugging Face API key
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_API_KEY")

# -------------------
# Backend functions
# -------------------
def pdf_to_text(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def build_chain(text):
    # 1. Split
    splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=15)
    docs = splitter.create_documents([text])

    # 2. Embeddings + Vectorstore
    embeddings = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        huggingfacehub_api_token=HF_TOKEN,
    )
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 3. Prompt
    prompt = ChatPromptTemplate.from_template(
        """
        You are a helpful assistant. 
        Use the following context to answer the question:

        Context: {context}

        Question: {question}
        Answer:
        """
    )

    # 4. Model
    llm = HuggingFaceEndpoint(
        repo_id="deepseek-ai/DeepSeek-V3.1-Terminus",
        task="text-generation",
        huggingfacehub_api_token=HF_TOKEN,
    )
    
    model=ChatHuggingFace(llm=llm)

    # 5. Parser
    parser = StrOutputParser()

    # 6. Parallel chain (retriever + question)
    parallel_chain = RunnableParallel(
        {"context": itemgetter("question") | retriever,
            "question": itemgetter("question")}
    )

    # 7. Main chain
    main_chain = parallel_chain | prompt | model | parser
    return main_chain

# -------------------
# Streamlit UI
# -------------------
st.set_page_config(page_title="Chat with PDF", layout="wide")
st.title("📄 Chat with your PDF")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    if "main_chain" not in st.session_state:
        with st.spinner("Processing PDF..."):
            raw_text = pdf_to_text(uploaded_file)
            st.session_state.main_chain = build_chain(raw_text)
            st.session_state.chat_history = []

    # Chat input
    user_question = st.chat_input("Ask something about the PDF")
    if user_question:
        with st.spinner("Thinking..."):
            answer = st.session_state.main_chain.invoke({"question": user_question})
        st.session_state.chat_history.append((user_question, answer))

    # Display chat history
    for q, a in st.session_state.chat_history:
        st.chat_message("user").write(q)
        st.chat_message("assistant").write(a)
footer = """
<style>
.footer {
    position: fixed;
    bottom: 0;
    width: 100%;
    background-color: #000000;
    color: white;
    text-align: center;
    padding: 10px;
    font-size: 14px;
}
</style>

<div class="footer">
    built by Yaniv 
</div>
"""


st.markdown(footer, unsafe_allow_html=True)
