import ollama
from transformers import pipeline

# 1. Configuration
# Make sure you have run 'ollama pull phi3' or your preferred model
LLM_MODEL = "phi3:latest"
EMO_MODEL = "j-hartmann/emotion-english-distilroberta-base"

# 2. Setup Local Perception (HuggingFace)
print(f"Loading {EMO_MODEL}...")
sentiment_pipe = pipeline("text-classification", model=EMO_MODEL)

# 3. Message History
messages = []

print(f"\n--- Hybrid Local Bot Active (LLM: {LLM_MODEL}) ---")

while True:
    user_input = input("\nYou: ")
    if user_input.lower() in ["exit", "quit"]: break

    # --- STEP 1: Local Emotion Detection ---
    # The pipeline returns a list like: [{'label': 'joy', 'score': 0.98}]
    emotion_result = sentiment_pipe(user_input)[0]
    detected_emotion = emotion_result['label']
    
    # --- STEP 2: Dynamic System Instruction ---
    # We change the bot's "personality" based on the detected emotion
    if detected_emotion == "anger":
        system_prompt = "The user is ANGRY. Be calm, empathetic, and try to de-escalate."
    elif detected_emotion == "joy":
        system_prompt = "The user is HAPPY. Be energetic and share their excitement!"
    elif detected_emotion == "sadness":
        system_prompt = "The user is SAD. Be supportive, gentle, and very comforting."
    else:
        system_prompt = "Be a helpful and professional AI assistant."

    # --- STEP 3: Local LLM Generation (Ollama) ---
    # We inject the prompt for this specific turn
    messages.append({"role": "user", "content": user_input})
    
    print(f"[{detected_emotion.upper()}] AI: ", end="", flush=True)

    # Use streaming for a better UI experience
    stream = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "system", "content": system_prompt}] + messages,
        stream=True,
    )

    full_response = ""
    for chunk in stream:
        content = chunk['message']['content']
        print(content, end="", flush=True)
        full_response += content

    messages.append({"role": "assistant", "content": full_response})
    print()