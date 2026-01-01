# Initialize Memory (The "Context Window")
chat_history = []

print("--- CLI Chatbot (Type 'quit' to exit) ---")

# The Conversation Loop
while True:
    # 1. Get User Input
    user_input = input("You: ")

    # 2. Check for exit condition
    if user_input.lower() in ["quit", "exit"]:
        print("Bot: Goodbye!")
        break

    # 3. Update Memory (User)
    chat_history.append({"role": "user", "content": user_input})

    # 4. Simulate the AI Response
    # (This is where we will later call the Sentiment Model + LLM)
    bot_response = f"Echo: {user_input}" 

    # 5. Update Memory (Bot)
    chat_history.append({"role": "assistant", "content": bot_response})

    # 6. Print Response
    print(f"Bot: {bot_response}")

    # 7. Print Chat History
    print("\n--- Chat History ---")
    for message in chat_history:
        print(f"{message['role']}: {message['content']}")
    print("-------------------")