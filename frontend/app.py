
import streamlit as st
import os
import base64
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv
import requests
 
load_dotenv()
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
 
 
def pil_to_base64(img: Image.Image) -> str:
    """Converts a PIL Image to a base64 JPEG string."""
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")
 
 
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
    else:
        with st.spinner("Processing your request — this takes ~15 seconds..."):
            try:
                images_base64 = [pil_to_base64(img) for img in images]
                response = requests.post(
                    f"{BACKEND_URL}/diagnose",
                    json={
                        "images_base64": images_base64,
                        "user_query": user_query,
                        "location": location,
                    },
                    timeout=120,
                )
                response.raise_for_status()
                result = response.json()
 
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
            except requests.exceptions.RequestException as e:
                st.error(f"An error occurred: {e}")
                st.info(
                    "💡 Tip: Make sure your GROQ_API_KEY is valid and the faiss_index folder "
                    "exists (run setup_rag.py first)."
                )
