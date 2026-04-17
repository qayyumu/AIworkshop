# AI Workshop

Welcome to the **AI Practical Workshop**: *From Foundations to Agentic AI*.

This repository holds lecture-aligned practical materials for a **5-day** workshop (lecture plus hands-on labs). Official topics, learning objectives, and day-by-day themes are summarized in **[Course Contents.pdf](Course%20Contents.pdf)** in the repo root—use that document as the syllabus; this README lists **where each lab lives** and **Open in Colab** links.

## How the repo maps to the course

| Course day (see PDF) | Themes from *Course Contents.pdf* | Materials in this repo |
|----------------------|-----------------------------------|-------------------------|
| **Day 1** — AI & deep learning foundations | Python for AI, supervised learning basics, pipelines, train/test, evaluation | `Session-01` — Python refresher, OOP, NumPy / Pandas / Matplotlib |
| **Day 2** — Classical ML & optimization | Regression, classification, SVM, trees, ensembles, scikit-learn labs | `Session-02` — `classical.ipynb`, `linear_classification.py`, and `Session-02/res/` datasets |
| **Day 3** — Neural networks & computer vision | MLP / CNN / ViT concepts, Keras, digit-style classification | `Session-03` — `Tensorflow.ipynb`, `Gradio_ex1.py`, `Gradio_ex2_chatbot.py` |
| **Day 4** — Generative foundations & LLMs | LLMs, prompting, OpenAI, Hugging Face transformers, local / generative workflows | `Session-04` — OpenAI notebook, DSPy scripts, Gradio + Hugging Face, vision transformers notebook, Gradio MCP, OCR and LLM training/inference scripts |
| **Day 5** — Agentic AI | Agents, planning, memory, tools; CrewAI project | `Session-05` — `Multi-agent.ipynb` |
| **Supplementary** | Extra CV experiment | `session_computervision/depth_Estimation.py` |

**Note:** Day 1 in the PDF also references scikit-learn at a foundational level; the main **comparative classical ML lab** is in `Session-02`, which aligns with **Day 2** of the course document.

## What you will practice

- **Python & OOP** — Core syntax and structuring code for ML and AI scripts.
- **NumPy, Pandas, Matplotlib** — Data handling and visualization.
- **Classical ML** — Regression, classification, and scikit-learn workflows.
- **Deep learning** — Keras / TensorFlow for neural networks (e.g. image tasks).
- **Transformers & demos** — Hugging Face pipelines, Gradio UIs, MCP-oriented examples, vision-focused transformer notebooks.
- **LLM APIs & DSPy** — OpenAI API usage and small DSPy programs.
- **Agentic workflows** — Multi-agent patterns with CrewAI (keys or local models via `.env`).

## Repository layout

| Folder | Contents |
|--------|----------|
| `Session-01` | `Python_basics.ipynb`, `Python_OOP_and_external_packages.ipynb` |
| `Session-02` | `classical.ipynb`, `linear_classification.py`, sample data in `res/` |
| `Session-03` | `Tensorflow.ipynb`, `Gradio_ex1.py`, `Gradio_ex2_chatbot.py` (Keras / TensorFlow and Gradio labs) |
| `Session-04` | `OpenAI_API.ipynb`, `Gradio_and_Huggingface.ipynb`, `Gradio_MCP.ipynb`, `HuggingFace_Computervision.ipynb`, `Dspy_ex1.py`, `Dspy_ex2_docSummary.py`, `Gradio_ocr.py`, `llm_training.py`, `llm_inference.py` |
| `Session-05` | `Multi-agent.ipynb` |
| `session_computervision` | `depth_Estimation.py` |
| *(root)* | `Course Contents.pdf`, `requirements.txt`, `README.md` |

Colab URLs below assume the GitHub repo **`qayyumu/GradioExp`** on branch **`main`**. After a fork or rename, change the path segment after `https://colab.research.google.com/github/`.

## Notebooks and scripts — Open in Colab

### Day 1 — `Session-01`

- **Python basics**: <a href="https://colab.research.google.com/github/qayyumu/GradioExp/blob/main/Session-01/Python_basics.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
- **OOP and external packages** (NumPy, Pandas, Matplotlib): <a href="https://colab.research.google.com/github/qayyumu/GradioExp/blob/main/Session-01/Python_OOP_and_external_packages.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

### Day 2 — `Session-02`

- **Classical ML (scikit-learn)**: <a href="https://colab.research.google.com/github/qayyumu/GradioExp/blob/main/Session-02/classical.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>  
  Uses CSV files in `Session-02/res/` — upload or mount that folder in Colab if paths are relative to the notebook.
- **Linear classification script**: <a href="https://colab.research.google.com/github/qayyumu/GradioExp/blob/main/Session-02/linear_classification.py" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> `linear_classification.py`

### Day 3 — `Session-03`

- **TensorFlow / Keras**: <a href="https://colab.research.google.com/github/qayyumu/GradioExp/blob/main/Session-03/Tensorflow.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
- **Gradio example 1**: <a href="https://colab.research.google.com/github/qayyumu/GradioExp/blob/main/Session-03/Gradio_ex1.py" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> `Gradio_ex1.py`
- **Gradio example 2 (chatbot)**: <a href="https://colab.research.google.com/github/qayyumu/GradioExp/blob/main/Session-03/Gradio_ex2_chatbot.py" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> `Gradio_ex2_chatbot.py`

### Day 4 — `Session-04`

- **OpenAI API**: <a href="https://colab.research.google.com/github/qayyumu/GradioExp/blob/main/Session-04/OpenAI_API.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
- **Gradio and Hugging Face**: <a href="https://colab.research.google.com/github/qayyumu/GradioExp/blob/main/Session-04/Gradio_and_Huggingface.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
- **Gradio MCP**: <a href="https://colab.research.google.com/github/qayyumu/GradioExp/blob/main/Session-04/Gradio_MCP.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
- **Hugging Face computer vision (Transformers)**: <a href="https://colab.research.google.com/github/qayyumu/GradioExp/blob/main/Session-04/HuggingFace_Computervision.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
- **DSPy** (Python scripts; open from GitHub in Colab or run locally):  
  - <a href="https://colab.research.google.com/github/qayyumu/GradioExp/blob/main/Session-04/Dspy_ex1.py" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> `Dspy_ex1.py`  
  - <a href="https://colab.research.google.com/github/qayyumu/GradioExp/blob/main/Session-04/Dspy_ex2_docSummary.py" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> `Dspy_ex2_docSummary.py`
- **Gradio OCR app**: <a href="https://colab.research.google.com/github/qayyumu/GradioExp/blob/main/Session-04/Gradio_ocr.py" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> `Gradio_ocr.py`
- **LLM scripts** (local training/inference demos):  
  - <a href="https://colab.research.google.com/github/qayyumu/GradioExp/blob/main/Session-04/llm_training.py" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> `llm_training.py`  
  - <a href="https://colab.research.google.com/github/qayyumu/GradioExp/blob/main/Session-04/llm_inference.py" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> `llm_inference.py`

### Day 5 — `Session-05`

- **Multi-agent (CrewAI)**: <a href="https://colab.research.google.com/github/qayyumu/GradioExp/blob/main/Session-05/Multi-agent.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

### Supplementary

- **Depth estimation**: <a href="https://colab.research.google.com/github/qayyumu/GradioExp/blob/main/session_computervision/depth_Estimation.py" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> `session_computervision/depth_Estimation.py`

## Getting started

### Prerequisites

- Basic programming comfort; ML concepts help from Day 2 onward.
- Read **Day objectives** in [Course Contents.pdf](Course%20Contents.pdf) before each session.

### Local setup

```bash
git clone https://github.com/qayyumu/GradioExp.git
cd GradioExp
pip install -r requirements.txt
```

For **Colab**, use the links above. Mount Google Drive or upload small assets (for example `Session-02/res/`) when notebooks expect local paths.

For **APIs and local LLMs**, create a **`.env`** file in the project root (ignored by git) with the variables each notebook expects—for example `OPENAI_API_KEY`, optional GLM-related keys, or settings for local Ollama as described in `Session-05/Multi-agent.ipynb` and related labs.
