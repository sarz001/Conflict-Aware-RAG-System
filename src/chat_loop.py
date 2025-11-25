from src.rag_pipeline import answer_query

def start_chat():
    print("\n=== NebulaGears Policy Assistant ===")
    print("Ask any question about company policy.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        query = input("You: ").strip()

        if query.lower() in ["exit", "quit", "q"]:
            print("Goodbye!")
            break

        print("\nThinking...\n")

        try:
            answer = answer_query(query)
            print("Assistant:\n")
            print(answer)
            print("\n" + "-"*60 + "\n")

        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    start_chat()
