def chatbot_response(user_input: str) -> str:
    user_input = user_input.lower()

    if "hello" in user_input or "hi" in user_input:
        return "Hello! How can I help you today?"

    elif "how are you" in user_input:
        return "I'm doing great! Thanks for asking 😊"

    elif "your name" in user_input:
        return "I am a chatbot."

    elif "help" in user_input:
        return "Sure! You can ask me about greetings, my name, or say goodbye."

    elif "bye" in user_input or "goodbye" in user_input:
        return "Goodbye! Have a great day 👋"

    else:
        return "Sorry, I didn't understand that. Can you rephrase?"
    
print("Chatbot: Hi! Type 'bye' to exit.")

while True:
    user_message = input("You: ")
    if user_message.lower() == "bye":
        print("Chatbot: Goodbye! 👋")
        break
    response = chatbot_response(user_message)
    print("Chatbot:", response)
