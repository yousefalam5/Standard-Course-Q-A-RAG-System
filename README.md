# Multi-Course Retrieval-Augmented Generation (RAG) System 📚🤖

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Framework-Streamlit](https://img.shields.io/badge/Framework-Streamlit-red.svg)](https://streamlit.io/)
[![VectorDB-ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-purple.svg)](https://www.trychroma.com/)
[![HuggingFace-Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)
[![License-Academic](https://img.shields.io/badge/License-Academic%20Use-green.svg)](#license)

An end-to-end, production-grade **Retrieval-Augmented Generation (RAG)** application designed to answer student and researcher questions grounded strictly in university lecture materials. The system indexes course documents across multiple formats (PDF, PowerPoint, Word, CSV, TXT), performs multilingual dense vector retrieval with cosine similarity, and produces verifiable answers with exact page and slide citations.

---

## 🌟 Key Features

- **Multilingual Support (Arabic & English):** Leverages multilingual sentence transformers (`paraphrase-multilingual-MiniLM-L12-v2`) to accurately embed and retrieve content across languages.
- **Multi-Format Ingestion Engine:** Extracts clean text, slide contents, and table data from:
  - **PDF** (page-level tracking)
  - **PowerPoint PPTX** (slide-level & table cell extraction)
  - **Word DOCX** (paragraph & table extraction)
  - **Structured CSV & Plain TXT**
- **Persistent Vector Store:** High-performance vector search powered by **ChromaDB** with persistent cosine distance indexing.
- **Granular Course Filtering:** Allows querying individual courses (`Deep Learning`, `Data Analysis`, `Parallel and Distributed Computing`) or searching globally across all enrolled subjects.
- **Source Attribution & Grounded Generation:** Every generated answer is backed by direct text passages and accompanied by an interactive **Source Relevance & Similarity Chart** to prevent hallucination.
- **Interactive Web Interface:** Modern, responsive UI built with **Streamlit** featuring real-time indexing status, coverage analytics, and parameter tuning (Top-K selection).
- **Automated Retrieval Evaluation:** Includes a dedicated evaluation module measuring **Hit@K** accuracy against ground-truth test cases (`evaluation_questions.csv`).

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    subgraph Ingestion ["1. Document Ingestion & Indexing"]
        A[Lecture Files<br>PDF, PPTX, DOCX, CSV] --> B[Text Extractor & Cleaner]
        B --> C[Recursive Character Splitter<br>Chunk Size: 800, Overlap: 120]
        C --> D[Multilingual Embedder<br>MiniLM-L12-v2]
        D --> E[(ChromaDB Vector Store<br>Cosine Metric)]
    end

    subgraph QueryFlow ["2. Retrieval & Generation Pipeline"]
        F[User Question<br>Arabic or English] --> G[Query Embedding]
        G --> H{Course Filter?}
        H -->|Specific Course| I[Course-Constrained Search]
        H -->|All Courses| J[Global Vector Search]
        E -.-> I
        E -.-> J
        I --> K[Top-K Ranked Passages + Metadata]
        J --> K
        K --> L[Augmented Context Prompt]
        L --> M[Language Model<br>Qwen2.5-1.5B-Instruct]
        M --> N[Synthesized Grounded Answer]
    end

    subgraph Presentation ["3. User Experience & Evaluation"]
        N --> O[Streamlit Web App]
        K --> P[Source Attribution & Similarity Chart]
        O --- P
        Q[Evaluation Suite<br>evaluation_questions.csv] --> R[Hit@K Benchmarking Chart]
    end
```

---

## 📁 Repository Structure

```text
Course_QA_RAG/
│
├── app.py                          # Streamlit interactive web application
├── course_rag.py                   # Core RAG engine (Ingestion, ChromaDB, Retrieval, LLM)
├── standard_course_qa_rag.ipynb    # Research & evaluation Jupyter notebook
├── evaluation_questions.csv        # Ground-truth test dataset for retrieval evaluation
├── requirements.txt                # Production dependencies
├── .gitignore                      # Git ignore rules for environments, cache, and DB
│
├── .streamlit/
│   └── config.toml                 # Streamlit UI custom theme configuration
│
└── data/
    └── courses/                    # University course materials organized by folder
        ├── Data_Analysis/          # Data Analysis lecture slides & notes (PDFs)
        ├── Deep_Learning/          # Deep Learning lectures & decks (PDF, PPTX)
        └── Parallel_Distributed_Computing/ # Parallel Computing lecture notes (PDFs)
```

---

## ⚙️ Technology Stack

| Component | Technology | Description |
| :--- | :--- | :--- |
| **User Interface** | [Streamlit](https://streamlit.io/) | Interactive dashboard with real-time analytics |
| **Embeddings** | `sentence-transformers` | `paraphrase-multilingual-MiniLM-L12-v2` |
| **Vector Database** | [ChromaDB](https://www.trychroma.com/) | Persistent HNSW index with cosine similarity metric |
| **Text Chunking** | `langchain-text-splitters` | Recursive character text splitting (800 chars / 120 overlap) |
| **Generative LLM** | `transformers` | `Qwen/Qwen2.5-1.5B-Instruct` |
| **Document Parsers** | `pypdf`, `python-pptx`, `python-docx` | Native structural extraction for multi-format decks |
| **Evaluation** | `pandas`, `matplotlib` | Hit@K retrieval evaluation and coverage visualization |

---

## 🚀 Getting Started

### Prerequisites
- **Python 3.11** (Recommended for optimal PyTorch and ChromaDB compatibility)
- Windows / Linux / macOS
- (Optional) NVIDIA GPU with CUDA support for accelerated model inference

---

### Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/course-qa-rag.git
   cd course-qa-rag
   ```

2. **Create and activate a virtual environment:**
   * **Windows (PowerShell):**
     ```powershell
     py -3.11 -m venv .venv311
     .\.venv311\Scripts\Activate.ps1
     ```
   * **Linux / macOS:**
     ```bash
     python3.11 -m venv .venv
     source .venv/bin/activate
     ```

3. **Install dependencies:**
   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

### Running the Application

1. **Start the Streamlit Web Server:**
   ```bash
   streamlit run app.py
   ```
2. **Access the application:** Open `http://localhost:8501` in your browser.
3. **Index the Course Materials:**
   - On the left sidebar, click **"Build or refresh index"**.
   - The system will ingest all files in `data/courses/`, extract text, compute multilingual embeddings, and build the persistent vector database.
4. **Ask Questions:** Select a course or search across all courses, enter your query in Arabic or English, and inspect the grounded answer with exact slide/page references.

---

## 📊 Evaluation & Research Notebook

The repository includes a comprehensive Jupyter Notebook (`standard_course_qa_rag.ipynb`) designed for academic review and empirical evaluation:

- **Ingestion Analysis:** Visualizes the distribution of indexed chunks per course.
- **Hit@K Evaluation:** Evaluates retrieval precision against curated benchmark questions (`evaluation_questions.csv`):
  $$\text{Hit@K} = \frac{1}{|Q|} \sum_{q \in Q} \mathbb{I}(\text{Expected Course} \in \text{Top-K Retrieved})$$
- **Semantic Sanity Testing:** Tests cross-lingual query retrieval and validates cosine relevance thresholds.

To launch the notebook:
```bash
jupyter notebook standard_course_qa_rag.ipynb
```

---

## 📋 Course Syllabus Coverage

The included dataset comprises materials across three core computing disciplines:
1. **Deep Learning:** Neural network fundamentals, activation functions, backpropagation, CNN architectures, optimization algorithms, and regularizations.
2. **Data Analysis:** Data collection & cleaning pipelines, exploratory data analysis (EDA), statistical inference, hypothesis testing, and feature scaling.
3. **Parallel & Distributed Computing:** Parallel hardware architectures, shared vs. distributed memory models, Amdahl's Law, thread synchronization, and message-passing paradigms (MPI).

---

## 🛡️ License & Academic Integrity

This project is developed for educational and academic purposes. All lecture materials belong to their respective university departments and instructors.
