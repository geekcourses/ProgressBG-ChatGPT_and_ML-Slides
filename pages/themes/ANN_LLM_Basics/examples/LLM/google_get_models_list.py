import os
from google import genai
from dotenv import load_dotenv

# 1. Load the .env file
load_dotenv()
my_api_key = os.getenv("GOOGLE_API_KEY")

# Check if key exists to prevent vague errors
if not my_api_key:
    print("Error: GOOGLE_API_KEY not found. Check your .env file.")
    exit()

# 2. Initialize the Client with the key
client = genai.Client(api_key=my_api_key)

print("List of models that support generateContent:\n")
for m in client.models.list():
    for action in m.supported_actions:
        if action == "generateContent":
            print(m.name)
