# 🎓 College Rules RAG Assistant (Lab 1 Pipeline)

A complete, self-contained **Retrieval-Augmented Generation (RAG)** pipeline designed to answer college institutional policy questions (examinations, hostel regulations, library rules, and academic services) with strict source citation and hallucination resistance.

Built with **Sentence-Transformers**, **FAISS**, and **Hugging Face Transformers** using open-source, local models.

---

## 📌 Architecture & Pipeline Overview

```
               [ College Policy Documents (.txt) ]
                               │
                               ▼
                    [ Chunking & Cleaning ]
                               │
                               ▼
        [ all-MiniLM-L6-v2 Dense Embeddings (384-dim) ]
                               │
                               ▼
                   [ FAISS Vector Index (IP) ]
                               │
[ User Query ] ──► [ Query Embedding ] ──► [ Top-k Retrieval ]
                                                   │
                                                   ▼
                                       [ Grounded Context Assembly ]
                                       (with [1], [2] source tags)
                                                   │
                                                   ▼
                                      [ Qwen2.5-1.5B-Instruct ]
                                                   │
                                                   ▼
                                        [ Cited Response / Refusal ]
```

### Key Components:
1. **Document Ingestion**: Cleans and chunks plain-text policy documents from `data/` while preserving metadata tags (document source, chunk IDs).
2. **Dense Vector Embeddings**: Encodes document chunks into 384-dimensional dense vectors using `sentence-transformers/all-MiniLM-L6-v2` with $L_2$ normalization.
3. **Vector Indexing**: Fast approximate nearest-neighbor retrieval using **FAISS** with an Inner Product index (`IndexFlatIP`), mathematically equivalent to cosine similarity after unit normalization.
4. **Answer Generation**: Generates grounded, factual answers using **`Qwen/Qwen2.5-1.5B-Instruct`** (Hugging Face Transformers). Automatically adapts to GPU (`float16`) or CPU (`float32`).
5. **Hallucination Prevention**: Strict prompt grounding prompts the model to answer *only* from retrieved context and output `"I don't know based on the provided documents."` if the answer is missing.

---

## 📂 Project Structure

```
college_rag_lab1_project/
├── data/
│   ├── academic_services.txt     # Academic transcripts, ID cards, bonafide certs
│   ├── exam_rules.txt            # Attendance requirements, grading, re-evaluation
│   ├── hostel_rules.txt          # Curfew hours, quiet hours, visitor policy
│   ├── library_rules.txt         # Book loan limits, late fines, membership rules
│   └── README.txt                # Dataset notes & attribution
├── lab1_rag_pipeline.ipynb       # End-to-end RAG implementation & evaluation notebook
├── requirements.txt              # Production/lab Python dependencies
├── .env.example                  # Environment variable template
├── .gitignore                    # Git ignore file (excludes .venv, caches, models)
└── README.md                     # Documentation
```

---

## 🚀 Getting Started

### Prerequisites
- **Python 3.10+**
- Minimum 6GB free disk space for PyTorch & model weights (or use **Google Colab** with free T4 GPU).

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/college_rag_lab1_project.git
cd college_rag_lab1_project
```

### 2. Set Up a Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment (Optional)
Copy the example environment configuration:
```bash
cp .env.example .env
```
*(Optional: add your `HF_TOKEN` if you encounter Hugging Face download rate limits).*

### 5. Launch the Notebook
```bash
jupyter lab
# or
jupyter notebook lab1_rag_pipeline.ipynb
```
Follow the cells sequentially from **0. Install dependencies** to **7. Test set and failure log**.

---

## 🧪 Evaluation Test Set

The pipeline is benchmarked against 8 rigorous queries designed to evaluate both retrieval precision and refusal mechanisms:

| ID | Test Query | Expected Type | Target Source |
|:---|:---|:---:|:---|
| **Q1** | What minimum attendance is required for the semester-end examination? | Answerable | `exam_rules.txt` (75%) |
| **Q2** | When does the hostel gate close on weekends? | Answerable | `hostel_rules.txt` (10:30 PM) |
| **Q3** | What are the library's weekday opening hours? | Answerable | `library_rules.txt` (8:00 AM – 9:00 PM) |
| **Q4** | How much is the fine per book per day for a late return? | Answerable | `library_rules.txt` (₹5/day) |
| **Q5** | How many days after result declaration can a student apply for re-evaluation? | Answerable | `exam_rules.txt` (10 working days) |
| **Q6** | What is the MBA tuition fee? | **Unanswerable** | Out-of-domain *(Must Refuse)* |
| **Q7** | What is today's lunch menu in the hostel mess? | **Unanswerable** | Out-of-domain *(Must Refuse)* |
| **Q8** | What is the student help desk's phone number? | **Unanswerable** | Out-of-domain *(Must Refuse)* |

---

## 📦 Pushing to GitHub & Reclaiming Disk Space

If you are backing up this repository to GitHub before removing it from your local machine to save storage:

### 1. Stage and Commit
```bash
git add .
git commit -m "feat: complete college rules RAG pipeline with FAISS, MiniLM, and Qwen2.5"
```

### 2. Push to GitHub
```bash
# Rename branch to main if not already
git branch -M main

# Link to your GitHub remote repository
git remote add origin https://github.com/<your-username>/<your-repo-name>.git

# Push code
git push -u origin main
```

### 3. Reclaim Disk Space
Once your repository is safely pushed to GitHub:
- The virtual environment (`.venv/`) occupies ~6 GB due to PyTorch and CUDA libraries.
- You can safely remove the virtual environment:
  ```bash
  rm -rf .venv/
  ```
- Or delete the entire project directory from your Desktop:
  ```bash
  cd ..
  rm -rf college_rag_lab1_project/
  ```
You can clone it back at any time and re-run `pip install -r requirements.txt`.

---

## ⚠️ Disclaimer
The text files in `data/` contain synthetic sample documents created specifically for academic demonstration and laboratory exercises. They are **not** official institutional policies.
