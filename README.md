# Victorian AI: Fine-Tuning GPT-2 on Sherlock Holmes

An end-to-end Natural Language Processing (NLP) pipeline exploring domain adaptation and text generation using Sir Arthur Conan Doyle's *The Adventures of Sherlock Holmes*. The project benchmarks training language models from scratch versus transfer learning via pre-trained foundation models, culminating in an interactive investigative console built natively inside Google Colab.

---

## 📑 Project Overview

Generative AI models excel at capturing style, vocabulary, and cadence when adapted properly. This repository documents a four-phase transition from raw data ingestion to interactive inference:

1. **Automated Corpus Ingestion & Hygiene:** Extracts raw text from Project Gutenberg, programmatically strips licensing headers and boilerplate, and normalizes the narrative.
2. **Phase 1 Benchmark (Character-Level from Scratch):** Demonstrates the limitations of training an uninitialized model on a small dataset without pre-existing language representations.
3. **Phase 2 Domain Adaptation (Fine-Tuned GPT-2):** Adapts a 124M-parameter foundation model using subword Byte-Pair Encoding (BPE), causal language modeling, and mixed-precision acceleration.
4. **Interactive Deployment ("221B Baker Street Inquiry Console"):** A user-facing interface powered by `ipywidgets` providing prompt scaffolding, persona selection, temperature controls, and formatted case dossiers.

---

## 🏛️ Architecture & Comparative Analysis

| Dimension | Phase 1: From Scratch (Character Baseline) | Phase 2: Transfer Learning (Fine-Tuned GPT-2) |
| :--- | :--- | :--- |
| **Token Representation** | Character-level (65 unique tokens) | Byte-Pair Encoding (BPE) (50,257 subwords) |
| **Context Window** | 64 characters (~10–12 words) | 128 subword tokens (~100+ words) |
| **Prior Knowledge** | Zero (random weight initialization) | Pre-trained web-scale English grammar & semantics |
| **Training Duration** | 5 epochs (~2.5 min CPU / ~15s GPU) | 3 epochs (~90s on Nvidia T4 GPU) |
| **Final Loss** | ~1.68 (Character Categorical Cross-Entropy) | 2.82 (Subword Causal Language Modeling Loss) |
| **Output Evaluation** | **Phonetic Gibberish:** Learned whitespace and word length patterns, but invented non-words (`mereath`, `contrain`, `prost beck`). | **Syntactic & Stylistic Fluency:** Valid Victorian vocabulary, authentic character dynamics, and dramatic gothic narrative flow. |

---

## 📂 Repository Structure

```text
├── sherlock_holmes_generative_ai.ipynb   # Complete, runnable Google Colab notebook
├── sherlock_holmes.txt                   # Cleaned Project Gutenberg text corpus
├── README.md                             # Project documentation and reproduction guide
└── requirements.txt                      # Core dependencies for local/cloud environments
```

---

## ⚙️ Requirements & Environment

The notebook is optimized for **Google Colab** with an active **Nvidia Tesla T4 GPU**.

### Core Libraries
- `torch >= 2.0.0`
- `transformers >= 4.35.0`
- `datasets >= 2.14.0`
- `accelerate >= 0.24.0`
- `ipywidgets >= 8.0.0`

Install via pip:

```bash
pip install -q transformers datasets accelerate ipywidgets torch
```

---

## 🚀 Quickstart Guide

### 1. Open in Google Colab
Upload `sherlock_holmes_generative_ai.ipynb` to Google Colab.

### 2. Enable Hardware Acceleration
1. Navigate to **Runtime** > **Change runtime type**.
2. Under **Hardware accelerator**, select **T4 GPU**.
3. Click **Save**.

### 3. Run the Pipeline
Execute cells sequentially (`Runtime` > `Run all` or `Ctrl + F9` / `Cmd + F9`).

---

## 🕵️‍♂️ The 221B Baker Street Inquiry Console

The final stage deploys an interactive console built directly inside the notebook environment:

```text
+-------------------------------------------------------------------------+
| 🕵️‍♂️ 221B Baker Street Interactive Inquiry Console                        |
|                                                                         |
| Case:           [ The Case of the Muddy Sovereign                  v ]  |
| Inquiry:        [ how could the gold sovereign be dropped in mud?   ]  |
| Speaker:        (o) Sherlock Holmes        ( ) Dr. John Watson          |
| Thinking Style: [ Strict Logic ---- Standard Deduction ---- Creative ]  |
| Length (Words): [ ------------------o------------------------------ ] 90|
|                                                                         |
| [                         Deduce Case Solution                        ] |
+-------------------------------------------------------------------------+
```

### Key Functional Features
- **Persona Framing:** Dynamically structures raw input into Victorian dialogue conventions before tokenization:
  - *Holmes:* `"Watson," said Holmes, his sharp eyes glinting as he considered [Inquiry]...`
  - *Watson:* `I recorded in my journal the strange incident regarding [Inquiry]...`
- **User-Friendly Thinking Styles:** Maps qualitative terms to model temperature parameters:
  - *Strict Logic:* Temperature `0.40` (low variance, conservative deductions)
  - *Standard Deduction:* Temperature `0.65` (balanced narrative flow)
  - *Creative Theory:* Temperature `0.85` (high variance, unexpected plot developments)
- **Top-p & Top-k Nucleus Sampling:** Applies `top_k=50` and `top_p=0.92` to eliminate improbable token tails while preventing repetitive generation loops.

---

## 🧠 Key Findings & Limitations

1. **The 50k Vocabulary Bottleneck:** Training a 50,257-token subword vocabulary from scratch on a small corpus (~155,000 tokens) produces extreme underfitting. In 216 steps, an uninitialized model collapses into high-frequency punctuation and newline loops.
2. **Transfer Learning Efficiency:** Fine-tuning an existing foundation model requires only domain adaptation—learning Arthur Conan Doyle's tone, syntax, and pacing—reducing training time to ~90 seconds on a T4 GPU.
3. **Local Coherence vs. Global Logic:** While fine-tuned models generate grammatically convincing Victorian prose, they operate on statistical token probabilities rather than narrative memory. Characters can abruptly contradict established premises or introduce ungrounded plot artifacts. Language models serve as effective brainstorming and ideation assistants, but require human editorial oversight to maintain global plot continuity.

---

## 👤 Author

- **Name:** Cesar Juarez
- **Date:** September 20, 2026
- **Program:** Data Analytics & Applied Artificial Intelligence
