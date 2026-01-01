import os
import sys
from dotenv import load_dotenv, find_dotenv
from google import genai
from google.genai import types
from transformers import pipeline



# ---------------------------------------------------------
# STEP 1: API Key Validation & Client Initialization
# ---------------------------------------------------------
# find_dotenv() returns a string of the absolute path to your .env file
# load_dotenv() then uses that specific path to load the variables
load_dotenv(find_dotenv())

API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    print("ERROR: GOOGLE_API_KEY not found in .env file or environment variables.")
    print("Please create a .env file with: GOOGLE_API_KEY=your_key_here")
    sys.exit(1) # Exit the script safely

try:
    # Initialize the modern 2026 Client
    client = genai.Client(api_key=API_KEY)
except Exception as e:
    print(f"Failed to initialize Gemini Client: {e}")
    sys.exit(1)




# ---------------------------------------------------------
# STEP 2: Local LLM Model (Sentiment Analysis)
# ---------------------------------------------------------
# Using DistilBERT locally for a near-instant "vibe check"
print("Loading local sentiment model...")
vibe_check = pipeline(
    "sentiment-analysis", 
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

def get_vibe(text):
    result = vibe_check(text)[0]
    return result['label']

# ---------------------------------------------------------
# STEP 3: Prompt Engineering (Dynamic Context)
# ---------------------------------------------------------
def generate_system_instruction(vibe):
    # Dynamic context based on local model output
    if vibe == "POSITIVE":
        return "You are a highly enthusiastic and supportive assistant. Use emojis and encouraging language."
    else:
        return "You are a calm, empathetic, and professional assistant. Focus on being helpful and steady."

# ---------------------------------------------------------
# STEP 4: LLM Cloud API (Gemini 3 Flash)
# ---------------------------------------------------------
def get_ai_response(user_text, system_prompt):
    # Initialize the latest GenAI client    
    client = genai.Client(api_key=API_KEY) 

    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=user_text,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.7
        )
    )
    return response.text

# ---------------------------------------------------------
# EXECUTION FLOW
# ---------------------------------------------------------
def main():
    # Step 1: User Input
    raw_input = input("Enter your message: ")

    # Step 2 & 3: Local Vibe Check & Prompt Construction
    vibe = get_vibe(raw_input)
    system_prompt = generate_system_instruction(vibe)
    
    print(f"\n--- Internal Logic ---")
    print(f"Detected Vibe: {vibe}")
    print(f"Engineered Prompt: {system_prompt}\n")

    # Step 4 & 5: Cloud Generation & Final Response
    final_answer = get_ai_response(raw_input, system_prompt)
    
    print(f"--- AI Response ---")
    print(final_answer)

if __name__ == "__main__":
    main()