from healthcare_rag.rag.retrieval.vector_retriever import VectorRetriever


def main():
    retriever = VectorRetriever()

    results = retriever.retrieve(
        "What are the symptoms of type 2 diabetes?",
        top_k=3,
    )

    print("=" * 80)
    print(f"Retrieved documents: {len(results)}")
    print("=" * 80)

    for i, result in enumerate(results, 1):
        print(f"\nRESULT {i}")
        print("-" * 80)
        print("ID:", result.get("id"))
        print("Score:", result.get("score"))
        print("Source:", result.get("source"))
        print("Metadata:", result.get("metadata"))
        print("Content:")
        print(result.get("content", "")[:500])


if __name__ == "__main__":
    main()