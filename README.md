# RAG-Document-Question-Answering-System
Overview

This project implements a Retrieval-Augmented Generation (RAG) based Document Question Answering System that allows users to upload PDF documents and ask natural-language questions about their content.

The system processes the uploaded document, divides it into meaningful text chunks, generates semantic embeddings using Google Gemini, stores the embeddings in a FAISS vector database, retrieves the most relevant document sections, and uses a Gemini LLM to generate answers based only on the retrieved context.

The application provides an interactive Streamlit web interface for document upload and question answering.
