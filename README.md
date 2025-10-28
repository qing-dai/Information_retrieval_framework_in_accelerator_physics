# From Dataset to Optimization: A Benchmarking Framework for Information Retrieval in the Particle Accelerator Domain

This repository publishes a comprehensive benchmarking framework and dataset for information retrieval (IR) in the accelerator physics domain. It includes annotated training and test datasets, tools for LLM-based annotation, model fine-tuning scripts, evaluation frameworks, and metadata analysis utilities.

## 📊 Repository Overview

The repository is organized into several key directories, each serving a specific purpose in the information retrieval pipeline:

---

## 📁 Folder Structure

### `Data/` - **Benchmark Datasets** ⭐

**This is the most valuable folder in the repository**, containing high-quality benchmark datasets for information retrieval in the accelerator domain.

- **`Test_data/`**
  - `Classification_test_dataset.xlsx`: Test dataset for classification tasks
  - `Filtered_Positive_Questions.xlsx`: Curated test questions for IR evaluation
  - **Use case**: Evaluate and benchmark IR systems against standardized test data
  
- **`Training_data/`**
  - `total_training_data.xlsx`: Annotated training data for model fine-tuning
  - **Use case**: Fine-tune embedding models and rerankers to improve performance on accelerator physics domain

---

### `LLM_annotation/` - **LLM-Based Annotation Pipeline**

Tools for generating training data using Large Language Models (LLM) with embedding-based retrieval.

**Key Scripts:**

1. **`annotate_against_IPAC_top_k.py`**
   - Annotates IPAC papers using Gemini-2.5-Pro
   - Uses embedding models to retrieve top-K relevant chunks before LLM annotation
   - Implements async processing with caching and retry logic
   
   **Example usage:**
   ```bash
   python LLM_annotation/annotate_against_IPAC_top_k.py
   ```
   Note: Configure `LLM_MODEL`, `EMBED_MODEL`, `TOP_K` variables in the script. Set `GEMINI_API_KEY` in your environment.

2. **`annotate_against_expert_paper.py`**
   - Annotates expert papers (comprehensive annotation of all chunks)
   - Unlike IPAC annotation, processes all chunks without top-K filtering
   
   **Example usage:**
   ```bash
   python LLM_annotation/annotate_against_expert_paper.py
   ```

3. **`build_vector_store.py`**
   - Builds FAISS vector stores for fast similarity search
   - Supports GPU acceleration for large-scale retrieval
   
   **Example usage:**
   ```bash
   python LLM_annotation/build_vector_store.py
   ```

4. **`check_training_data_count.py` / `count_json.py`**
   - Utility scripts to analyze and verify annotation data counts

**Key Differences:**
- **Expert papers**: All chunks are annotated by the LLM
- **IPAC papers**: Only top-K chunks (retrieved via embedding models) are annotated by the LLM for efficiency

---

### `finetune/` - **Model Fine-Tuning**

Scripts to fine-tune embedding models using the annotated training data.

**Key Scripts:**

1. **`finetune_model_exc_qwen.py`**
   - Fine-tunes SentenceTransformer models (excluding Qwen models)
   - Supports contrastive learning objectives
   
   **Example usage:**
   ```bash
   python finetune/finetune_model_exc_qwen.py \
       --model-name sentence-transformers/all-mpnet-base-v2 \
       --data-path Data/Training_data/total_training_data.xlsx \
       --output-dir ./fine_tuned_models/mpnet_ft \
       --num-epochs 3 \
       --batch-size 16
   ```

2. **`finetune_qwen_0.6B_emb.py`**
   - Fine-tunes Qwen embedding models (0.6B parameters)
   
   **Example usage:**
   ```bash
   python finetune/finetune_qwen_0.6B_emb.py \
       --model-name Qwen/Qwen-0.6B \
       --data-path Data/Training_data/total_training_data.xlsx \
       --output-dir ./fine_tuned_models/qwen_ft
   ```

**Note**: Adjust `CUDA_VISIBLE_DEVICES` environment variable in scripts to select GPU devices.

---

### `overall_eval/` - **Evaluation Framework**

Comprehensive evaluation tools for both classification and information retrieval tasks.

#### `overall_eval/IR/` - Information Retrieval Evaluation

**Main Script: `cli.py`**

Evaluates embedding models and rerankers on the benchmark test dataset.

**Example usage for embedding models:**
```bash
python overall_eval/IR/cli.py \
    --input Data/Test_data/Filtered_Positive_Questions.xlsx \
    --model sentence-transformers/all-mpnet-base-v2 \
    --kind st \
    --device cuda:0 \
    --output mpnet_base_results
```

**Example usage for Qwen reranker:**
```bash
python overall_eval/IR/cli.py \
    --input Data/Test_data/Filtered_Positive_Questions.xlsx \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --kind qwen-reranker \
    --device cuda:0 \
    --output qwen_reranker_results
```

**Parameters:**
- `--input`: Path to test data Excel file
- `--model`: Model name or path (HuggingFace model ID or local path)
- `--kind`: Model type (`st`, `qwen`, `st-last-token`, `qwen-reranker`)
- `--device`: CUDA device (e.g., `cuda:0`, `cuda:5`)
- `--add-instruction`: Add instruction prompt for BGE models (auto-enabled for BGE)
- `--ks`: Comma-separated k values for evaluation metrics (default: `3,5,10,15,20`)
- `--output`: Output prefix for result files

**Outputs:**
- `{output}_allowlist_with_scores.xlsx`: Scores for allowlist evaluation
- `{output}_full_corpus_with_scores.csv.gz`: Scores for full corpus evaluation
- Metrics: NDCG@k, MAP@k for specified k values

**Additional IR scripts:**
- `save_allowlist_scores.py`: Saves allowlist evaluation scores
- `save_full_corpus_scores.py`: Saves full corpus evaluation scores
- `data.py`: Data loading and preprocessing utilities
- `models.py`: Model loading utilities
- `scorer.py`: Scoring and ranking functions

#### `overall_eval/classification/` - Classification Evaluation

**Key Scripts:**

1. **`classification_eval_allowlist.py`**
   - Evaluates classification performance on allowlist
   
   **Example usage:**
   ```bash
   python overall_eval/classification/classification_eval_allowlist.py \
       --input Data/Test_data/Classification_test_dataset.xlsx \
       --model base_model=sentence-transformers/all-mpnet-base-v2 \
       --kind st \
       --device cuda:0
   ```

2. **`classification_eval_full_corpus.py`**
   - Evaluates classification performance on full corpus

**Analysis Tools** (`classification_result_analysis/`):
- `F1_by_threshold.py`: F1 score analysis across different thresholds
- `ROC_analysis.py`: ROC curve and AUC computation
- `PR_AUC_analysis.py`: Precision-Recall AUC analysis

---

### `meta_data_analysis/` - **Result Analysis Tools**

Scripts for analyzing evaluation results and generating insights.

#### `meta_data_analysis/IR/allowlist/`
- `question_type_allowlist_IR_analysis.py`: Analyzes IR performance by question type (allowlist)
- `specific_to_paper_allowlist_IR_analysis.py`: Analyzes paper-specific IR results (allowlist)

#### `meta_data_analysis/IR/full-corpus/`
- `question_type_full_corpus_IR_analysis.py`: Analyzes IR performance by question type (full corpus)
- `specific_to_paper_full_corpus_IR_analysis.py`: Analyzes paper-specific IR results (full corpus)

**Example usage:**
```bash
python meta_data_analysis/IR/allowlist/question_type_allowlist_IR_analysis.py
```

**Note**: These scripts provide model-specific result analysis for the test dataset. Update file paths in scripts to point to your evaluation output files.

---

## 🚀 Quick Start

### 1. **Install Dependencies**

```bash
pip install sentence-transformers torch pandas numpy faiss-cpu openpyxl ranx litellm google-generativeai python-dotenv tqdm
```

For GPU support with FAISS:
```bash
pip install faiss-gpu
```

### 2. **Evaluate a Model**

```bash
# Evaluate a base embedding model
python overall_eval/IR/cli.py \
    --input Data/Test_data/Filtered_Positive_Questions.xlsx \
    --model BAAI/bge-base-en-v1.5 \
    --kind st \
    --device cuda:0 \
    --output bge_base_results
```

### 3. **Fine-tune a Model**

```bash
# Fine-tune on training data
python finetune/finetune_model_exc_qwen.py \
    --model-name sentence-transformers/all-mpnet-base-v2 \
    --data-path Data/Training_data/total_training_data.xlsx \
    --output-dir ./fine_tuned_models/mpnet_ft
```

### 4. **Evaluate Fine-tuned Model**

```bash
# Evaluate the fine-tuned model
python overall_eval/IR/cli.py \
    --input Data/Test_data/Filtered_Positive_Questions.xlsx \
    --model ./fine_tuned_models/mpnet_ft \
    --kind st \
    --device cuda:0 \
    --output mpnet_ft_results
```

---

## ⚙️ Path Configuration

**Important**: Most scripts contain hardcoded paths that need to be adapted to your environment.

**Common paths to update:**
- Input data paths in evaluation scripts
- Model save directories in fine-tuning scripts
- Output directories for results
- GPU device IDs (`CUDA_VISIBLE_DEVICES`, `--device`)

**Example locations to check:**
- `overall_eval/classification/classification_eval_allowlist.py`: Line ~28 (default input path)
- `finetune/finetune_model_exc_qwen.py`: Line ~11 (CUDA device)
- `LLM_annotation/*.py`: API keys and model configurations

---

## 📈 Evaluation Metrics

The framework supports multiple IR evaluation metrics:
- **NDCG@k** (Normalized Discounted Cumulative Gain)
- **MAP@k** (Mean Average Precision)

Default k values: 3, 5, 10, 15, 20 (configurable via `--ks` parameter)

---

## 🔬 Dataset Details

The benchmark datasets cover:
- **Domain**: Particle accelerator physics
- **Sources**: IPAC conference papers + expert-annotated papers
- **Annotation**: LLM-assisted (Gemini-2.5-Pro) with human verification
- **Task types**: 
  - Information Retrieval (question → relevant document chunks)
  - Classification (question-chunk relevance classification)

---

## 📝 Citation

If you use this benchmark dataset or framework, please cite:

```
From Dataset to Optimization: A Benchmarking Framework for Information 
Retrieval in the Particle Accelerator Domain
```

---

## 🤝 Contributing

Contributions are welcome! Please ensure:
- Code follows existing style conventions
- Paths are parameterized (not hardcoded)
- Documentation is updated for new features

---

## 📧 Contact

For questions or issues, please open a GitHub issue in this repository.

---

## License

[Add appropriate license information]