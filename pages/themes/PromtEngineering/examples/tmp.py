import os
from google import genai
from google.genai import types
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
        contents="""
In my world, the number 2 is actually 5, and the number 3 is actually 10. What is (2 + 2) + 3? Give me the final number only.
""",
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(include_thoughts=False, thinking_budget=0)
        ),
    )
    print(response.text)
except Exception as e:
    print(f"An error occurred: {e}")
