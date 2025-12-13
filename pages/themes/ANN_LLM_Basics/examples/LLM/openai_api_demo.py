from openai import OpenAI

# 1. AUTHENTICATION
# The computer looks for a "secret password" saved in your system
# called OPENAI_API_KEY.
# In your terminal/command prompt, you would run:
# export OPENAI_API_KEY="sk-proj-..."

client = OpenAI()

print("Sending request to OpenAI...")

# 2. THE REQUEST
# We ask the "waiter" to get us a joke.
completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a one-sentence joke about programming."},
    ],
)

# 3. THE RESPONSE
# We print the text the model sent back.
print("Response received:")
print(completion.choices[0].message.content)
