# 🌾 Kisaan AI — Crop Doctor & Yield Optimizer

An AI-powered multimodal agricultural assistant that analyzes crop images, retrieves organic farming knowledge, and generates hyper-local farming recommendations using Vision-Language Models, RAG, weather intelligence, and soil analysis.

Built using Groq LLMs, LangChain, FAISS/ChromaDB, Hugging Face embeddings, and Streamlit.

**Deployed Link** https://kisaan-ai.streamlit.app/

---

# 🚀 Features

- 📷 Crop image disease detection using Vision LLM
- 🌱 Organic and chemical-free farming recommendations
- 🔍 Retrieval-Augmented Generation (RAG) pipeline
- 🌦️ Real-time weather-aware crop suggestions
- 🌍 Location-based soil condition analysis
- 🧠 Context-aware agricultural advisory system
- ⚡ Fast inference using Groq LLM APIs
- 🎨 Interactive Streamlit-based user interface

---

# 🧠 Tech Stack

## Languages
- Python

## AI / Machine Learning
- Groq LLM
- LangChain
- Hugging Face Embeddings
- Transformers
- RAG (Retrieval-Augmented Generation)

## Vector Database
- FAISS

## Frontend
- Streamlit

## APIs
- OpenWeatherMap API

---

# 🏗️ System Architecture

```text
User Input
   │
   ├── Crop Image
   ├── Farmer Query
   └── Location
        │
        ▼
Vision Model Analysis
(qwen3.6-27b)
        │
        ▼
Weather + Soil Retrieval
        │
        ▼
RAG Semantic Search
(Hugging Face + FAISS)
        │
        ▼
Context Synthesis
(gpt-oss-120b)
        │
        ▼
Organic Farming Recommendations
```

---

# 📂 Project Structure

```text
Kisaan-AI/
│
├── app.py                 # Main Streamlit application
├── setup_rag.py           # Builds vector database
├── requirements.txt       # Dependencies
├── faiss_index             # Local vector database
├── .env                   # API keys
└── README.md
```

---

# ⚙️ Installation

## 1️⃣ Clone Repository

```bash
git clone https://github.com/piyushjat/Kisaan-AI.git
cd Kisaan-AI
```

---

## 2️⃣ Create Virtual Environment

### Windows
```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / Mac
```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🔑 Environment Variables

Create a `.env` file in the root directory.

```env
GROQ_API_KEY=your_groq_api_key
OPENWEATHER_API_KEY=your_openweather_api_key
```

---

# 📚 Setup RAG Database

Run this once before starting the application.

```bash
python setup_rag.py
```

This script:
- Downloads the KisanVaani agriculture QA dataset
- Generates embeddings using Hugging Face
- Stores vectors in local ChromaDB

---

# ▶️ Run the Application

```bash
streamlit run app.py
```

---

# 📸 How It Works

1. Upload crop image
2. Enter location and farming query
3. Vision model analyzes crop health
4. Weather and soil conditions are fetched
5. RAG retrieves relevant agricultural knowledge
6. LLM generates personalized organic recommendations

---

# 🧩 AI Models Used

## Vision Model
- `qwen/qwen3.6-27b`

Used for:
- Crop identification
- Disease detection
- Nutrient deficiency analysis

---

## Text Generation Model
- `openai/gpt-oss-120b`

Used for:
- Agricultural reasoning
- Organic recommendation generation
- Context synthesis

---

# 📊 Dataset

Dataset used:
- **KisanVaani/agriculture-qa-english-only**

Source:
- Hugging Face Datasets

Contains:
- Agricultural questions
- Organic farming solutions
- Crop disease discussions

---

# 🌱 Example Use Cases

- Detect crop diseases from leaf images
- Organic pesticide recommendations
- Soil-aware farming guidance
- Weather-adaptive crop management
- Sustainable farming assistance

---

# 🔥 Future Improvements

- Multi-language farmer support
- Voice-based interaction
- Live satellite crop monitoring
- Mobile application deployment
- Fine-tuned agriculture-specific LLM

---

# 👨‍💻 Author

Piyush Choudhary

---

# ⭐ Acknowledgements

- Groq
- Hugging Face
- LangChain
- Streamlit
- OpenWeatherMap
- KisanVaani Dataset

---
