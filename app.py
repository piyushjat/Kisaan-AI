import streamlit as st
import os
import requests
import base64
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from sentence_transformers import CrossEncoder
 
# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
weather_api_key = os.getenv("OPENWEATHER_API_KEY")
FAISS_INDEX_DIR = "./faiss_index"

# --- INITIALIZATION ---
@st.cache_resource
def load_vector_db():
    """Load the vector DB using a local HuggingFace embedding model (no API key needed)."""
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    if not os.path.exists(FAISS_INDEX_DIR):
        raise FileNotFoundError(
            f"FAISS index not found at '{FAISS_INDEX_DIR}'. "
            "Run `python setup_rag.py` first to build it."
        )
 
    return FAISS.load_local(
        FAISS_INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True,
    )

vector_db = load_vector_db()

#retrives top 10 and reranks top 5
CANDIDATE_K = 10
RERANK_TOP_N = 5
retriever = vector_db.as_retriever(search_kwargs={"k": CANDIDATE_K})
 
@st.cache_resource
def load_reranker():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
 
reranker = load_reranker()
 

# initializing Groq LLM (text/reasoning model)
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.3,
    api_key=groq_api_key,
)

# Vision model 
vision_llm = ChatGroq(
    model="qwen/qwen3.6-27b",
    temperature=0.3,
    api_key=groq_api_key,
)

# --- HELPER FUNCTIONS (to get weather and soil data)---
def get_weather(location):
    """Fetches real-time weather. Falls back to mock data if no API key is provided."""
    if not weather_api_key:
        return "Temperature: 32°C, Humidity: 65% (Mock Data)"

    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={weather_api_key}&units=metric"
        res = requests.get(url, timeout=10).json()
        temp = res['main']['temp']
        humidity = res['main']['humidity']
        desc = res['weather'][0]['description']
        return f"Temperature: {temp}°C, Humidity: {humidity}%, Condition: {desc.capitalize()}"
    except Exception:
        return "Weather data unavailable."


def get_soil_condition(location):
    """Mock lookup for soil conditions based on location."""
    soil_database = {
        "lucknow": "Alluvial soil, rich in potash but poor in phosphorus",
        "punjab": "Loamy to sandy loam, good for wheat and rice",
        "maharashtra": "Black cotton soil, retains moisture well",
        "kerala": "Laterite soil, acidic in nature",
        "gujarat": "Sandy loam to clay loam, moderate fertility",
        "rajasthan": "Sandy and arid soil, low organic matter",
        "karnataka": "Red laterite soil, slightly acidic",
        "andhra pradesh": "Black and red loamy soil",
        "tamil nadu": "Red loam and alluvial, varies by region",
        "west bengal": "Alluvial soil, good for paddy cultivation",
    }
    city = location.lower().split(",")[0].strip()
    return soil_database.get(city, "Standard Loamy Soil (Default)")


def pil_to_base64(img: Image.Image) -> str:
    """Converts a PIL Image to a base64 JPEG string."""
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


# --- LCEL CHAINS ---
 
# 1. Vision chain: prompt (text + N images)|vision_llm|output parser
def build_vision_messages(inputs: dict) -> list:
    content = [
        {
            "type": "text",
            "text": (
                "You are an agricultural expert AI. You have been given "
                f"{len(inputs['img_base64_list'])} photo(s) of the same crop/plant, "
                "possibly from different angles or showing different affected areas. "
                "Analyze them together. Identify: (1) the crop type, (2) any visible "
                "diseases, pests, or nutrient deficiencies, (3) overall health status. "
                "Be concise and specific. Also address the farmer's specific question: "
                f"{inputs['user_query']}"
            ),
        }
    ]
    for img_base64 in inputs["img_base64_list"]:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
        })
    return [HumanMessage(content=content)]


vision_chain = RunnableLambda(build_vision_messages) | vision_llm | StrOutputParser()
 
# 2. Retrieval chain:
#    rerank -> select top 5 -> construct a single context string for the LLM
def rerank_top_n(inputs: dict) -> list:
   
    query = inputs["query"]
    docs = inputs["docs"]
    if not docs:
        return []
    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda pair: pair[1], reverse=True)
    return [doc for doc, _ in ranked[:RERANK_TOP_N]]
 
 
def format_docs(docs) -> str:
    """context construction"""
    if not docs:
        return "No specific documents found. Rely on general organic agricultural knowledge."
    return "\n\n".join(doc.page_content for doc in docs)
 
 
retrieval_chain = (
    RunnableParallel(docs=retriever, query=RunnablePassthrough())
    | RunnableLambda(rerank_top_n)
    | RunnableLambda(format_docs)
)
 
# 3. Synthesis chain: prompt (all collected context) | llm | output parser
SYSTEM_PROMPT = (
    "You are an expert Agricultural AI assistant specializing in organic and sustainable farming. "
    "You always recommend chemical-free, eco-friendly solutions. "
    "Format your response clearly using markdown with headings and bullet points."
)
 
SYNTHESIS_TEMPLATE = """
A farmer has asked: "{user_query}"
 
**Data Collected:**
- **Image Analysis:** {image_analysis}
- **Location:** {location}
- **Current Weather:** {weather}
- **Soil Condition:** {soil}
 
**Reference Knowledge (from organic farming database):**
{rag_context}
 
**Your Task:**
1. Confirm the crop and any disease/issue identified from the image.
2. Provide practical, **chemical-free/organic** treatment or care advice based on the reference knowledge.
3. Adapt your advice specifically to the current weather and soil conditions mentioned above.
4. Add preventive tips for future crop health.
 
Use clear markdown headings (##) and bullet points. Be practical and farmer-friendly.
"""
 
synthesis_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", SYNTHESIS_TEMPLATE),
])
synthesis_chain = synthesis_prompt | llm | StrOutputParser()
 
 
# --- CORE LOGIC ---
def process_farmer_query(images: list, user_query: str, location: str) -> str:
    """Full pipeline: vision chain (multi-image) → weather/soil → retrieval chain → synthesis chain."""
 
    # 1. Image Analysis via the vision LCEL chain
    st.info("👁️ Analyzing image with Vision Model (Groq)...")
    img_base64_list = [pil_to_base64(img) for img in images]
    image_analysis = vision_chain.invoke({
        "user_query": user_query,
        "img_base64_list": img_base64_list,
    })
 
    # 2. Fetch Environmental Data 
    st.info("🌍 Fetching Weather and Soil Data...")
    weather = get_weather(location)
    soil = get_soil_condition(location)
 
    # 3. RAG Retrieval via the retrieval LCEL chain (similarity search -> rerank -> top 5 -> context)
    st.info("📚 Searching & Reranking Agricultural Knowledge Base...")
    search_query = f"Organic chemical-free solutions for: {image_analysis}. Farmer question: {user_query}"
    rag_context = retrieval_chain.invoke(search_query)
 
    # 4. Final Synthesis via the synthesis LCEL chain
    st.info("🧠 Generating customized organic farming advice (Groq)...")
    return synthesis_chain.invoke({
        "user_query": user_query,
        "image_analysis": image_analysis,
        "location": location,
        "weather": weather,
        "soil": soil,
        "rag_context": rag_context,
    })
 
 
# --- STREAMLIT UI ---
st.set_page_config(page_title="AI Crop Doctor", page_icon="🌾", layout="wide")
 
# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Source+Sans+3:wght@400;600&display=swap');
 
    html, body, [class*="css"] {
        font-family: 'Source Sans 3', sans-serif;
    }
    h1, h2, h3 {
        font-family: 'Playfair Display', serif;
    }
    .stButton > button {
        background: linear-gradient(135deg, #2d6a4f, #40916c);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #1b4332, #2d6a4f);
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(45,106,79,0.4);
    }
    .stAlert {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)
 
st.title("🌾 AI Crop Doctor & Yield Optimizer")
st.caption("Developer@Piyush")
st.write(
    "Upload a photo of your crop, describe your concern, and the AI will analyze it using "
    "computer vision, check your local weather and soil data, and recommend organic solutions."
)
 
st.divider()
 
col1, col2 = st.columns(2)
with col1:
    location = st.text_input("📍 Your Location:", value="Lucknow, India")
with col2:
    user_query = st.text_input(
        "❓ Your Question:",
        placeholder="How do I treat this disease organically?",
    )
 
uploaded_files = st.file_uploader(
    "📷 Upload Crop Image", type=["jpg", "jpeg", "png"], accept_multiple_files=True
)
 
 
MAX_DISPLAY_WIDTH = 450 

 
images = []
if uploaded_files:
    images = [Image.open(f) for f in uploaded_files]
 
    cols = st.columns(min(4, len(images)))
    for i, img in enumerate(images):
        display_image = img.copy()
        display_image.thumbnail((MAX_DISPLAY_WIDTH, MAX_DISPLAY_WIDTH))
        with cols[i % len(cols)]:
            st.image(display_image, caption=f"Image {i + 1}")
st.divider()
 
if st.button("🔍 Analyze Crop", use_container_width=False):
    if not images:
        st.error("Please upload a crop image first.")
    elif not location or not user_query:
        st.error("Please fill in both your location and your question.")
    elif not groq_api_key:
        st.error("GROQ_API_KEY is missing. Please set it in your .env file.")
    else:
        with st.spinner("Processing your request — this takes ~15 seconds..."):
            try:
                answer = process_farmer_query(images, user_query, location)
                st.success("✅ Analysis Complete!")
                st.markdown("---")
                st.markdown("### 🌱 Diagnosis & Organic Recommendations")
                st.markdown(answer)
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.info(
                    "💡 Tip: Make sure your GROQ_API_KEY is valid and the faiss_index folder "
                    "exists (run setup_rag.py first)."
                )