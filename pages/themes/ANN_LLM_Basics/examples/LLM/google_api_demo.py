import os
from google import genai
from dotenv import load_dotenv

### AUTHENTICATION
# Load environment variables from a .env file located in the same directory.
# Your .env file should contain: GOOGLE_API_KEY="AIzaSy..."
load_dotenv()

my_api_key = os.getenv("GOOGLE_API_KEY")

if not my_api_key:
    print("Error: GOOGLE_API_KEY not found in .env")
    exit()

client = genai.Client(api_key=my_api_key)

### Query the model:
MODEL_NAME = "gemini-2.5-flash"
prompt = "Tell me a witty joke about Python programmers."

print("Sending request to Google...")
response = client.models.generate_content(model=MODEL_NAME, contents=prompt)

# print response
print("Response received:")
print(response.text)
