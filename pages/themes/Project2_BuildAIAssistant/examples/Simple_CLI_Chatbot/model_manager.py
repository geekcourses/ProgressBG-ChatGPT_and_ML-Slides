from google import genai
import os
import sys
from dotenv import load_dotenv, find_dotenv

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

def list_available_chat_models(client):
    """
    Returns a list of models authorized for your API key/tier
    that are suitable for a chatbot (Gemini & Gemma).
    """
    chat_models = []
    
    # client.models.list() only returns models your API Key is authorized to use.
    for model in client.models.list():
        name_lower = model.name.lower()
        
        # 1. Must support content generation
        can_gen = model.supported_actions and "generateContent" in model.supported_actions
        
        # 2. Must be a chatbot family (Gemini or Gemma)
        is_chatbot_family = "gemini" in name_lower or "gemma" in name_lower
        
        # 3. Exclude Image-only models (Imagen)
        is_not_image = "imagen" not in name_lower

        if can_gen and is_chatbot_family and is_not_image:
            chat_models.append({
                "id": model.name,
                "display_name": model.display_name,
                "input_limit": model.input_token_limit
            })
            
    return chat_models

def get_best_model(client, preferred_model="gemini-1.5-flash"):
    """
    Tries to find your preferred model. If not available in your tier,
    it falls back to the first available Gemini model.
    """
    available = list_available_chat_models(client)
    
    if not available:
        raise RuntimeError("No chat-compatible models found for this API Key/Tier.")

    # Check for preferred model (handling the 'models/' prefix)
    for m in available:
        if preferred_model in m['id']:
            return m['id']
            
    # Fallback to the first available model in your authorized list
    return available[0]['id']

# --- Example Usage ---
if __name__ == "__main__":
    ai_client = get_chatbot_client()
    
    try:
        available = list_available_chat_models(ai_client)
        print("Available models:")
        for model in available:
            print(f"- {model['display_name']} ({model['id']})")
    except Exception as e:
        print(f"Error: {e}")