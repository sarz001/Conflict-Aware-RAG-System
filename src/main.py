from src.rag_pipeline import answer_query

if __name__ == "__main__":
    query = input("Enter your question: ")
    answer = answer_query(query)
    print("\n=== FINAL ANSWER ===\n")
    print(answer)
