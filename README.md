# AI Workshop

Welcome to the **AI Practical Workshop**.

This repository contains materials and code examples for a hands-on workshop. Content is organized by session folders (`Session-01` … `Session-05`), each mapping to a day of topics.

## Workshop overview

You will work with:

- **Python**: Syntax, data types, control flow, and structuring code with OOP.
- **Data stack**: NumPy, Pandas, and Matplotlib for analysis and plotting.
- **Classical ML**: scikit-learn for regression, classification, and related workflows.
- **Gradio**: Interactive demos and UIs for models.
- **Hugging Face**: Transformers, pipelines, and vision examples.
- **TensorFlow / Keras**: Neural networks (e.g. image classification).
- **OpenAI & DSPy**: API usage and small programmatic LLM workflows.
- **Multi-agent systems**: CrewAI-style agents, tasks, and tools (local or cloud LLMs).

## Repository layout

| Folder | Focus |
|--------|--------|
| `Session-01` | Python basics and OOP / external packages |
| `Session-02` | Classical machine learning (scikit-learn) |
| `Session-03` | Gradio, Hugging Face, TensorFlow, computer vision |
| `Session-04` | OpenAI API and DSPy examples |
| `Session-05` | Multi-agent patterns with CrewAI |
| `session_computervision` | Extra computer-vision script (`depth_Estimation.py`) |

Colab links below use the GitHub path `qayyumu/GradioExp` on the `main` branch. If you fork or rename the repo, update the URL segment after `/github/`.

## Workshop schedule

### Day 1 — `Session-01`: Python basics and OOP

- **Python basics**: <a href="https://colab.research.google.com/github/qayyumu/GradioExp/blob/main/Session-01/Python_basics.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>  
  Introduction to Python: types, structures, and core syntax.

- **OOP and external packages**: <a href="https://colab.research.google.com/github/qayyumu/GradioExp/blob/main/Session-01/Python_OOP_and_external_packages.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>  
  Classes, objects, and practical use of NumPy, Pandas, and Matplotlib.

### Day 2 — `Session-02`: Classical ML

- **Classical ML with scikit-learn**: <a href="https://colab.research.google.com/github/qayyumu/GradioExp/blob/main/Session-02/classical.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>  
  Linear regression, logistic regression, train/test splits, and related examples (datasets under `Session-02/res/`).

### Day 3 — `Session-03`: Gradio, Hugging Face, TensorFlow, vision

- **Gradio and Hugging Face**: <a href="https://colab.research.google.com/github/qayyumu/GradioExp/blob/main/Session-03/Gradio_and_Huggingface.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>  
  Building demos with Gradio and Hugging Face pipelines.

- **Gradio MCP example**: <a href="https://colab.research.google.com/github/qayyumu/GradioExp/blob/main/Session-03/Gradio_MCP.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>  
  Connecting Gradio with MCP-style tooling patterns.

- **Hugging Face computer vision**: <a href="https://colab.research.google.com/github/qayyumu/GradioExp/blob/main/Session-03/HuggingFace_Computervision.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>  
  Vision models and tasks with Transformers (large notebook).

- **TensorFlow / Keras**: <a href="https://colab.research.google.com/github/qayyumu/GradioExp/blob/main/Session-03/Tensorflow.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>  
  Deep learning with Keras (TensorFlow backend).

### Day 4 — `Session-04`: OpenAI and DSPy

- **OpenAI API**: <a href="https://colab.research.google.com/github/qayyumu/GradioExp/blob/main/Session-04/OpenAI_API.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>  
  Using the OpenAI API for text and related tasks.

- **DSPy (Python scripts)** — run locally or open in Colab from GitHub:  
  - <a href="https://colab.research.google.com/github/qayyumu/GradioExp/blob/main/Session-04/Dspy_ex1.py" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> `Dspy_ex1.py`  
  - <a href="https://colab.research.google.com/github/qayyumu/GradioExp/blob/main/Session-04/Dspy_ex2_docSummary.py" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> `Dspy_ex2_docSummary.py`

### Day 5 — `Session-05`: Multi-agent systems

- **Multi-agent with CrewAI**: <a href="https://colab.research.google.com/github/qayyumu/GradioExp/blob/main/Session-05/Multi-agent.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>  
  Agents, tasks, crews, and optional web tools; configure API keys or local models via environment variables (see notebook and `.env`).

### Supplementary — computer vision script

- **Depth estimation**: <a href="https://colab.research.google.com/github/qayyumu/GradioExp/blob/main/session_computervision/depth_Estimation.py" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> `session_computervision/depth_Estimation.py`

## Getting started

### Prerequisites

- Comfortable with basic programming concepts.
- Familiarity with machine learning ideas is helpful for later sessions.

### Local setup

Clone the repository and install dependencies:

```bash
git clone https://github.com/qayyumu/GradioExp.git
cd GradioExp
pip install -r requirements.txt
```

You can also run most notebooks in [Google Colab](https://colab.research.google.com/) using the links above (upload or mount files such as `Session-02/res/` when a notebook expects them).

For API-based sessions, create a `.env` file (not committed; see `.gitignore`) and set keys as required by each notebook—for example `OPENAI_API_KEY`, optional GLM keys, or local Ollama endpoints.
