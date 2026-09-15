
from healthcare_rag.rag.pipeline import RAGPipeline
def test_rag_pipeline():
    pipeline=RAGPipeline(retriever_type="vector")

    question="what are the symptoms of type2 diabetes?"

    print(f"question: {question}\n")
    print("="*60)
    result=pipeline.query(question,top_k=3)
    print(f"Answer:\n{result['answer']}\n")
    print("="*60)
    print(f"sources:{result['sources']}")
    print(f"Documents retrieved: {result['retrieved_count']}")
if __name__=="__main__":
    test_rag_pipeline()    