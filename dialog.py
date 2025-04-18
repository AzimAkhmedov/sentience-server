from load_model import get_response

print("Welcome to AI Therapist. Type 'exit' to quit.\n")

while True:
    user_input = input("\nYou: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    reply = get_response(f"{user_input}\n")
    print("Therapist:", reply)