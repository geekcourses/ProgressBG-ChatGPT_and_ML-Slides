from google import genai
from google.genai import types
from model_manager import get_chatbot_client, get_best_model


# ---------------------------------------------------------------------------- #
#                               Main chatbot loop                              #
# ---------------------------------------------------------------------------- #
client = get_chatbot_client()
MODEL = get_best_model(client, preferred_model="gemini-2.5-flash-lite")

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

    # 4. Call AI API (Gemini)
    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=chat_history,
            config=types.GenerateContentConfig(
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