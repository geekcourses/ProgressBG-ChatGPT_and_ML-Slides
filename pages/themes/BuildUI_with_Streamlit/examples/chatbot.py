import streamlit as st
import ollama
from transformers import pipeline

# --- 1. SETTINGS & MODEL LOADING ---
st.set_page_config(page_title="Hybrid Sentiment Bot", layout="wide")

LLM_MODEL = "phi3:latest"
EMO_MODEL = "j-hartmann/emotion-english-distilroberta-base"

# Cache the model so it doesn't reload on every interaction
@st.cache_resource
def load_models():
    st.write(f"🔄 Loading {EMO_MODEL} into memory...")
    return pipeline("text-classification", model=EMO_MODEL)

sentiment_pipe = load_models()

# --- 2. SESSION STATE (Memory) ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_emotion" not in st.session_state:
    st.session_state.last_emotion = "None"

# --- 3. UI LAYOUT ---
st.title("🤖 Hybrid Sentiment-Aware Assistant")

# Sidebar for status and metrics
with st.sidebar:
    st.header("Analytics")
    st.metric("Detected Emotion", st.session_state.last_emotion.upper())
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 4. CHAT LOGIC ---
if prompt := st.chat_input("How are you feeling today?"):
    # Step 1: Detect Emotion
    emotion_result = sentiment_pipe(prompt)[0]
    label = emotion_result['label']
    st.session_state.last_emotion = label

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Step 2: Set dynamic system prompt based on your logic
    tone_map = {
        "anger": "The user is ANGRY. Be calm, empathetic, and try to de-escalate.",
        "joy": "The user is HAPPY. Be energetic and share their excitement!",
        "sadness": "The user is SAD. Be supportive, gentle, and very comforting."
    }
    system_prompt = tone_map.get(label, "Be a helpful and professional AI assistant.")

    # Step 3: Generate Ollama Response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        # Stream the output from Ollama
        stream = ollama.chat(
            model=LLM_MODEL,
            messages=[{"role": "system", "content": system_prompt}] + st.session_state.messages,
            stream=True,
        )

        for chunk in stream:
            content = chunk['message']['content']
            full_response += content
            response_placeholder.markdown(full_response + "▌")
        
        response_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
    st.rerun() # Refresh to update the sidebar metric