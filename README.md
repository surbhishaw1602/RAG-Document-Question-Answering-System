# RAG-Document-Question-Answering-System

**Overview**

This project implements a Retrieval-Augmented Generation (RAG) based Document Question Answering System that allows users to upload PDF documents and ask natural-language questions about their content.

The system processes the uploaded document, divides it into meaningful text chunks, generates semantic embeddings using Google Gemini, stores the embeddings in a FAISS vector database, retrieves the most relevant document sections, and uses a Gemini LLM to generate answers based only on the retrieved context.

The application provides an interactive Streamlit web interface for document upload and question answering.

**Features**

*  Upload PDF documents through a web interface
*  Semantic search over document content
*  Intelligent text chunking for improved retrieval
*  Gemini-based text embeddings
*  Fast similarity search using FAISS
*  Natural-language question answering
*  Gemini-powered answer generation
*  Interactive Streamlit UI
*  API key management using environment variables
*  Supports multi-page PDF documents
*  Prevents the model from answering using information outside the uploaded document
*  Configurable chunk size and chunk overlap

**Architecture**

```text
             PDF Document
                  │
                  ▼
           PDF Document Loader
                  │
                  ▼
            Text Extraction
                  │
                  ▼
            Text Chunking
                  │
                  ▼
          Gemini Embeddings
                  │
                  ▼
          FAISS Vector Store
                  │
                  ▼
             Retriever
                  │
                  ▼
           Relevant Chunks
                  │
                  ▼
            Gemini LLM
                  │
                  ▼
          Generated Answer
                  │
                  ▼
         Streamlit Interface
```

**Tech Stack**

| Technology                     | Purpose                                |
| ------------------------------ | -------------------------------------- |
| Python                         | Core programming language              |
| Streamlit                      | Web application interface              |
| LangChain                      | RAG pipeline and component integration |
| Google Gemini API              | Embeddings and answer generation       |
| FAISS                          | Vector database and similarity search  |
| PyPDFLoader                    | PDF document loading                   |
| RecursiveCharacterTextSplitter | Text chunking                          |
| python-dotenv                  | Environment variable management        |

**Project Structure**

```text
Rag_Pdf/
│
├── app.py
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

**File Description**

**`app.py`**

Main Streamlit application containing:
** PDF upload
* PDF processing
* Text chunking
* Gemini embeddings
* FAISS vector database
* Document retrieval
* Gemini question answering
* Streamlit UI*

**RAG Pipeline**

1. Document Upload
2. Document Loading
3. Text Chunking
4. Embedding Generation
5. FAISS Vector Database
6. Retrieval
7. Answer Generation

**Example Usage**

Upload a document

For example:

```text
DS1.pdf
```

The system processes the document:

```text
PDF loaded successfully — 19 pages.
Created 36 text chunks.
Vector database created.
```

**Ask a question**

**Question:**

```text
data preprocessing steps ?
```

**Answer:**

> Based on the provided documents, common steps in data preprocessing include:
>
> * Remove duplicates.
> * Handle missing values.
