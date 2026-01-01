import streamlit as st
import ollama
from transformers import pipeline

# --- 1. CONFIGURATION & STYLING ---
st.set_page_config(page_title="EASA - Emotionally Adaptive Service Assistant", layout="wide")

LLM_MODEL = "phi3:latest"
EMO_MODEL = "j-hartmann/emotion-english-distilroberta-base"

# Load the Sentiment Model
@st.cache_resource
def load_sentiment_pipeline():
    return pipeline("text-classification", model=EMO_MODEL)

sentiment_pipe = load_sentiment_pipeline()

# --- 2. SESSION STATE MANAGEMENT ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "token_count" not in st.session_state:
    st.session_state.token_count = 0

# --- 3. SIDEBAR (Usage Stats) ---
with st.sidebar:
    st.header("📊 Usage Stats")
    
    # Display the token count from the last interaction
    st.metric("Tokens Used (Last Turn)", st.session_state.token_count)
    
    st.markdown("---")
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.session_state.token_count = 0
        st.rerun()

# --- 4. MAIN INTERFACE ---
st.title("🤖 Emotionally Adaptive Service Assistant")
st.markdown("---")

# Display History
for message in st.session_state.messages:
    # We use distinct avatars to match your image (Red for user, Robot for AI)
    avatar = "🔴" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# --- 5. CHAT LOGIC ---
if prompt := st.chat_input("Type your message here..."):
    
    # --- Step A: Analyze Sentiment ---
    # We detect the emotion BEFORE generating the response
    emotion_result = sentiment_pipe(prompt)[0]
    label = emotion_result['label']
    
    # --- Step B: Display User Message ---
    with st.chat_message("user", avatar="🔴"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # --- Step C: Select System Personality ---
    tone_map = {
        "anger": "The user is ANGRY. Be extremely polite, apologetic, and professional.",
        "joy": "The user is HAPPY. Be enthusiastic, use an emoji, and match their energy!",
        "sadness": "The user is SAD. Be empathetic, gentle, and supportive.",
        "fear": "The user is AFRAID. Be reassuring and clear."
    }
    system_instruction = tone_map.get(label, "Be a helpful and concise AI assistant.")

    # --- Step D: Generate Response ---
    with st.chat_message("assistant", avatar="🤖"):
        response_placeholder = st.empty()
        
        # We start the response with the emotional tag as seen in your image
        # e.g., "[Neutral] • "
        prefix = f"**[{label.capitalize()}]** • "
        full_response = prefix
        raw_ai_text = ""

        # Stream from Ollama
        stream = ollama.chat(
            model=LLM_MODEL,
            messages=[{"role": "system", "content": system_instruction}] + st.session_state.messages,
            stream=True,
        )

        for chunk in stream:
            content = chunk['message']['content']
            raw_ai_text += content
            full_response += content
            response_placeholder.markdown(full_response + "▌")
        
        response_placeholder.markdown(full_response)

    # --- Step E: Update State & Stats ---
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    
    # Simple token estimation (Words / 0.75 is a common rule of thumb)
    # We count input + output tokens
    est_tokens = int((len(prompt) + len(raw_ai_text)) / 4)
    st.session_state.token_count = est_tokens
    
    # Force a rerun to update the Sidebar metric immediately
    st.rerun()