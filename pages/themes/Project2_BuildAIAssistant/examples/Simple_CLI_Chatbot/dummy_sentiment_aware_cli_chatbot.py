import random # Just for the dummy response simulation

# --- The Sentiment Logic ---
def analyze_sentiment(text):
    """
    In a real app, this would use NLTK, TextBlob, or a Hugging Face model.
    For this specific logic demo, we simulate it with keywords.
    """
    text = text.lower()
    if any(word in text for word in ["bad", "hate", "terrible", "angry"]):
        return "NEGATIVE"
    elif any(word in text for word in ["good", "love", "great", "happy"]):
        return "POSITIVE"
    else:
        return "NEUTRAL"

# --- The Context Manager ---
def get_system_instruction(sentiment):
    if sentiment == "NEGATIVE":
        return "System: [ALERT] User is upset. Be apologetic and professional."
    elif sentiment == "POSITIVE":
        return "System: [NOTE] User is happy. Be energetic and fun!"
    else:
        return "System: [Standard] Be helpful and concise."

# --- The Main Execution Loop ---
print("--- Sentiment-Aware CLI Chat (Type 'quit' to exit) ---")
chat_history = []

while True:
    user_input = input("\nYou: ")
    if user_input.lower() in ["quit", "exit"]:
        break
    
    # [STEP 1] Analyze Sentiment
    # We do this BEFORE calling the AI
    current_mood = analyze_sentiment(user_input)
    print(f"   (Debug: Detected mood is {current_mood})")
    
    # [STEP 2] Build the "Engineered Prompt"
    system_instruction = get_system_instruction(current_mood)
    
    # [STEP 3] Simulate the AI Response
    print(f"   (Debug: Sending instruction to AI -> '{system_instruction}')")
    bot_response = f"Echo: {current_mood}-{user_input}" 

    # Mock response to show flow completion
    print(f"Bot: [Responds based on {current_mood} context...]")