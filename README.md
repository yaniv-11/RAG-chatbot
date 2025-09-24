Chat with Your PDF: A RAG-Powered Document Q&A Application
This project is a web application that allows you to have an interactive conversation with any PDF document. You can upload a PDF, and the application uses a powerful Large Language Model (LLM) combined with a Retrieval-Augmented Generation (RAG) pipeline to answer your questions based on the document's content.

The development process and experimentation are detailed in the rag.ipynb notebook, while the final, polished web application is implemented in app.py.

## ✨ Features

Interactive Chat Interface: A user-friendly interface built with Streamlit to ask questions and receive answers.

Dynamic PDF Upload: Upload any PDF document on the fly.

Context-Aware Responses: The model is instructed to answer questions only from the context provided by the uploaded document, reducing hallucinations.

Session-Based Memory: The application maintains a chat history for the duration of the session.

Built with LangChain: Leverages the LangChain framework for building the entire RAG pipeline, from data loading to generation.

⚙️ How It Works: The RAG Pipeline
The application is built around a Retrieval-Augmented Generation (RAG) architecture to ensure that the LLM's responses are grounded in the content of the uploaded PDF.

PDF Parsing: When a user uploads a PDF, the text content is extracted from each page.

Text Chunking: The extracted text is split into smaller, manageable chunks using a RecursiveCharacterTextSplitter. This helps in creating focused and relevant embeddings.

Embedding Generation: Each text chunk is converted into a numerical vector (embedding) using the sentence-transformers/all-MiniLM-L6-v2 model from Hugging Face.

Vector Storage & Retrieval: The embeddings are stored in a FAISS vector store, which allows for efficient similarity searches. When a user asks a question, a retriever fetches the most relevant chunks from the store.

LLM Generation: The user's question and the retrieved text chunks are passed to a powerful LLM (deepseek-ai/DeepSeek-V3.1-Terminus) through a carefully crafted prompt. The LLM then generates a final answer based on the provided context.

LCEL Implementation: The final chain is elegantly constructed using the LangChain Expression Language (LCEL), which pipes together the retriever, the prompt, the model, and an output parser for a clean and efficient workflow.

🛠️ Tech Stack
Frameworks: LangChain, Streamlit

LLM: deepseek-ai/DeepSeek-V3.1-Terminus (via Hugging Face Endpoints)

Embedding Model: sentence-transformers/all-MiniLM-L6-v2

Vector Store: FAISS

Core Libraries: PyPDF2, python-dotenv, langchain-huggingface
