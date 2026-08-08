
import streamlit as st
import os
import asyncio
import sys
import time
import base64
import logging
from typing import TypedDict,Optional
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
from setup_rag import build_vector_db

from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from sentence_transformers import CrossEncoder
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph,START,END

MCP_SERVER_SCRIPT = os.path.join(os.path.dirname(__file__), "McpServer.py")

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
weather_api_key = os.getenv("OPENWEATHER_API_KEY")
FAISS_INDEX_DIR = "./faiss_index"

# --- LOGGING SETUP ---
logger=logging.getLogger("Kissan_ai")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _formatter= logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s |%(message)s"
    )
    _file_handler = logging.FileHandler("Kissan_ai.log")
    _file_handler.setFormatter(_formatter)
    _console_handler = logging.StreamHandler()
    _console_handler.setFormatter(_formatter)
    logger.addHandler(_file_handler)
    logger.addHandler(_console_handler)

# --- RETRY CONFIG (env-configurable) ---
GROQ_RETRY_MAX_ATTEMPTS = int(os.getenv("GROQ_RETRY_MAX_ATTEMPTS", "3"))
GROQ_RETRY_BACKOFF_BASE = int(os.getenv("GROQ_RETRY_BACKOFF_BASE", "2"))

def with_retries(func, *args, **kwargs):
    """Calls func with exponential backoff retries. Raises the last exception if all attempts fail."""
    last_exception = None
    for attempt in range(1, GROQ_RETRY_MAX_ATTEMPTS + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            logger.warning(f"Attempt {attempt}/{GROQ_RETRY_MAX_ATTEMPTS} failed: {e}")
            if attempt < GROQ_RETRY_MAX_ATTEMPTS:
                time.sleep(GROQ_RETRY_BACKOFF_BASE ** attempt)
    raise last_exception
 
# --- INITIALIZATION ---
@st.cache_resource
def load_vector_db():
    """Load the vector DB using a local HuggingFace embedding model (no API key needed)."""
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    if not os.path.exists(FAISS_INDEX_DIR):
        st.info("Building FAISS index for the first time. This may take a minute...")
        build_vector_db()
 
    return FAISS.load_local(
        FAISS_INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True,
    )

vector_db = load_vector_db()

#retrives top 10 and reranks the retrieved top 5
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

# vision model 
vision_llm = ChatGroq(
    model="qwen/qwen3.6-27b",
    temperature=0.3,
    api_key=groq_api_key,
)

mcp_client = MultiServerMCPClient(
    {
        "crop_doctor_tools": {
            "command": sys.executable,
            "args": [MCP_SERVER_SCRIPT],
            "transport": "stdio",
        }
    }
)

@st.cache_resource
def get_mcp_tools_by_name():
    
    async def _load():
        tools = await mcp_client.get_tools()
        return {tool.name: tool for tool in tools}
 
    return asyncio.run(_load())
 
 
async def fetch_weather_and_soil(location: str,tools_by_name: dict) -> tuple[str, str]:
    """Calls the MCP-backed get_weather / get_soil_condition LangChain tools."""
    
    weather = await tools_by_name["get_weather"].ainvoke({"location": location})
    soil = await tools_by_name["get_soil_condition"].ainvoke({"location": location})
    return weather, soil
 
 
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
 
class KissanState(TypedDict):
    # inputs
    images:list
    user_query:str
    location:str


    # populated by vision_node
    img_base64_list:list
    image_analysis:str

    # populated by weather_soil_node
    weather:str
    soil:str
    weather_soil_failed:bool

    # populated by retrieval_node
    rag_context:str
    retrieval_failed:bool

    # populated by synthesis_node
    final_answer:str

    # error tracking (hard-fail signal, set by vision_node or synthesis_node)
    error:Optional[str]


def vision_node(state:KissanState)->dict:
    try:
        img_base64_list=[pil_to_base64(img) for img in state["images"]]
        image_analysis=with_retries(
            vision_chain.invoke,{
                "user_query":state["user_query"],
                "img_base64_list":img_base64_list

            }
        )
        return{
            "img_base64_list":img_base64_list,
            "image_analysis":image_analysis
        }
    except Exception as e:
        logger.error(f"vision_node hard-failed:{e}")
        return{"error":f"Image analysis failed: {e}"}

def weather_soil_node(state:KissanState)->dict:
    """Non-critical node — degrades gracefully instead of crashing the graph."""
    if state.get("error"):
        return{}  # upstream already hard-failed, skip work

    try:
        tools_by_name=get_mcp_tools_by_name()
        weather,soil=with_retries(
            lambda:asyncio.run(fetch_weather_and_soil(state["location"],tools_by_name))
        )

        return {"weather":weather,"soil":soil,"weather_soil_failed":False}

    except Exception as e:
        logger.warning(f"weather_soil_node degraded: {e}")
        return{
            "weather":"unavailable",
            "soil":"unavailable",
            "weather_soil_failed":True
        }


def retrieval_node(state:KissanState)->dict:
    """Non-critical node — degrades gracefully instead of crashing the graph."""
    if state.get("error"):
        return{}

    try:
        search_query=(
            f"organic-chemical-free solutions for: {state['image_analysis']}."
            f"farmer question: {state['user_query']}"
        )
        rag_context=retrieval_chain.invoke(search_query)
        return{"rag_context":rag_context,"retrieval_failed":False}
    
    except Exception as e:
        logger.warning(f"retrieval node degraded:{e}")
        return{
            "rag_context":"No refrence data available- rely on generic farming Knowledge.",
            "retrieval_failed":True
        }

def synthesis_node(state:KissanState)->dict:
    """Critical node — hard-fails into state['error'] if retries are exhausted."""
    if state.get("error"):
        return {}

    try:
        final_answer=with_retries(
            synthesis_chain.invoke,{
                "user_query": state["user_query"],
                "image_analysis": state["image_analysis"],
                "location": state["location"],
                "weather": state["weather"],
                "soil": state["soil"],
                "rag_context": state["rag_context"],
            },
        )
        return {"final_answer":final_answer}
    except Exception as e:
        logger.error(f"synthesis_node hard-failed: {e}")
        return{"error":f"could not generate advice: {e}"}

builder=StateGraph(KissanState)

builder.add_node("vision_node", vision_node)
builder.add_node("weather_soil_node", weather_soil_node)
builder.add_node("retrieval_node", retrieval_node)
builder.add_node("synthesis_node", synthesis_node)

builder.add_edge(START,"vision_node")
builder.add_edge("vision_node","weather_soil_node")
builder.add_edge("weather_soil_node","retrieval_node")
builder.add_edge("retrieval_node","synthesis_node")
builder.add_edge("synthesis_node",END)

kisaan_graph = builder.compile()
 
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
                result = kisaan_graph.invoke({
                    "images": images,
                    "user_query": user_query,
                    "location": location,
                })
 
                if result.get("error"):
                    st.error(f"❌ {result['error']}")
                    st.info(
                        "💡 Tip: Make sure your GROQ_API_KEY is valid and the faiss_index folder "
                        "exists (run setup_rag.py first)."
                    )
                else:
                    answer = result["final_answer"]
                    if result.get("weather_soil_failed"):
                        st.warning(
                            "⚠️ Weather/soil data was unavailable — advice below is based on "
                            "image and general knowledge only."
                        )
                    if result.get("retrieval_failed"):
                        st.warning(
                            "⚠️ Knowledge base lookup failed — advice below is based on general "
                            "organic farming knowledge."
                        )
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
