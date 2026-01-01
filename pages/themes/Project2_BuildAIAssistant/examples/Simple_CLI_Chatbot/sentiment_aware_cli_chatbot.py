import os
import sys
from dotenv import load_dotenv, find_dotenv
from google import genai
from google.genai import types
from dotenv import load_dotenv, find_dotenv

# --- HELPER FUNCTIONS ---
def get_chatbot_client():
    """Initializes and returns the GenAI client."""
    load_dotenv(find_dotenv())
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        print("ERROR: GOOGLE_API_KEY not found.")
        sys.exit(1)

    try:
        return genai.Client(api_key=api_key)
    except Exception as e:
        print(f"Failed to initialize Gemini Client: {e}")
        sys.exit(1)

def analyze_sentiment(text):
    """
    Returns: 'POSITIVE', 'NEGATIVE', or 'NEUTRAL'
    """
    text = text.lower()
    if any(w in text for w in ["bad", "hate", "terrible", "angry", "slow"]):
        return "NEGATIVE"
    elif any(w in text for w in ["good", "love", "great", "happy", "fast"]):
        return "POSITIVE"
    return "NEUTRAL"

def get_dynamic_instruction(sentiment):
    if sentiment == "NEGATIVE":
        return "You are a Senior Support Agent. The user is frustrated. Apologize and be concise."
    elif sentiment == "POSITIVE":
        return "You are an energetic Brand Ambassador! The user is happy. Use emojis!"
    else:
        return "You are a helpful AI assistant. Be clear and direct."



# ---------------------------------------------------------------------------- #
#                               Main chatbot loop                              #
# ---------------------------------------------------------------------------- #
client = get_chatbot_client()
MODEL = "gemini-2.5-flash"

print(f"   (Auto-selected Model: {MODEL})")

chat_history = []

print("--- Sentiment-Aware Gemini Chat (Type 'quit' to exit) ---")

while True:
    # 1. Get User Input
    user_input = input("\nYou: ")

    # 2. Check for exit condition
    if user_input.lower() in ["quit", "exit"]:
        print("Bot: Goodbye!")
        break

    # 3. Update Memory (User)    
    chat_history.append(types.Content(role="user", parts=[types.Part(text=user_input)]))

    # 4. Analyze Sentiment & Prepare Context    
    current_mood = analyze_sentiment(user_input)
    system_instruction = get_dynamic_instruction(current_mood)
    print(f"   (System: Detected mood '{current_mood}' -> Adjusting Personality...)")

    # 5. Call AI API (Gemini)
    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=chat_history,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.7
            )
        )
        bot_text = response.text
    except Exception as e:
        bot_text = f"Error: {e}"

    # 6. Update Memory (Bot)
    chat_history.append(types.Content(role="model", parts=[types.Part(text=bot_text)]))

    # 7. Print Response
    print(f"Bot: {bot_text}")

    # 8. (Optional) Print Chat History Debug
    # print(chat_history)