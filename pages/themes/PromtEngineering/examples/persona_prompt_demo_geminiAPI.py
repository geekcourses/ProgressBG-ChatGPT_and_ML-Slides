import os
from google import genai
from dotenv import load_dotenv

### AUTHENTICATION
load_dotenv()
my_api_key = os.getenv("GOOGLE_API_KEY")

if not my_api_key:
    print("Error: GOOGLE_API_KEY not found in .env")
    exit()

# 1. Initialize the Client
client = genai.Client(api_key=my_api_key)

# 2. Define the System Instruction in the Config
try:
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="Tell me about apple.",
        config={
            "system_instruction": "You are a witty nutritionist. Give technical facts but keep it very sarcastic."
        },
    )
    print(response.text)
except Exception as e:
    print(f"An error occurred: {e}")
