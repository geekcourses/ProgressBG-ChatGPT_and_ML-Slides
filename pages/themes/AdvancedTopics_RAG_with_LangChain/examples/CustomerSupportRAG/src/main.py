import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate

# Load environment variables (like GOOGLE_API_KEY)
load_dotenv()

def get_llm_client():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite", 
        temperature=0,
    )

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_retriever(embeddings):
    return Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    ).as_retriever(search_kwargs={"k": 3})

def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def get_vector_store(embeddings):
    if not os.path.exists("./chroma_db"):
        print("Error: 'chroma_db' directory not found. Please run 'vector_store.py' first.")
        return
    
    return Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )

def get_prompt():
    template = """You are a helpful customer support assistant. 
Use the following pieces of retrieved context to answer the user's question.
If you don't know the answer based on the context, just say that you don't know. 
Do not try to make up an answer. 

Context:
{context}

Question: {question}

Helpful Answer:"""
    return PromptTemplate.from_template(template)

def get_rag_chain(llm, retriever):
    prompt = get_prompt()
    
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain

def main():
    embeddings = get_embeddings()
    retriever = get_retriever(embeddings)
    llm = get_llm_client()
    rag_chain = get_rag_chain(llm, retriever)
    
    print("\n--- Customer Support RAG Bot (Gemini & LCEL) ---")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        query = input("Ask a question: ")
        
        if query.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        
        if not query.strip():
            continue

        try:
            response = rag_chain.invoke(query)
            
            print(f"\nAI: {response}\n")
            
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
