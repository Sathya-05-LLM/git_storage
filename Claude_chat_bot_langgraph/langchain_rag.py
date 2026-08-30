from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

loader = TextLoader("sample_doc.txt")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(documents)

embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings)

llm = ChatOllama(model="gemma3:4b")

prompt_template = ChatPromptTemplate.from_template("""Answer the question using ONLY the context below.
If multiple relevant details exist in the context (e.g. different employee types, categories, or conditions), include ALL of them in your answer, not just the first one.
If the answer isn't in the context, say "I don't have that information."

Context:
{context}

Question: {question}

Answer:""")

chain = prompt_template | llm

fallback_template = ChatPromptTemplate.from_template("""You are a friendly, natural-sounding assistant. The user's message doesn't require looking up specific document information (it's a greeting, small talk, or casual conversation).

Reply naturally and conversationally, the way a person would in a normal chat.
Only mention that you can help with the employee handbook (leave, remote work, reimbursement, performance reviews, probation, code of conduct) if the user seems to be asking what you can do, or if this is clearly their first message. Otherwise, just chat normally without repeating that list.

User said: {question}

Reply:""")

fallback_chain = fallback_template | llm

SIMILARITY_THRESHOLD = 1.0


def ask(question: str) -> str:
    results_with_scores = vector_store.similarity_search_with_score(question, k=4)

    relevant_chunks = [doc.page_content for doc, score in results_with_scores if score < SIMILARITY_THRESHOLD]

    if not relevant_chunks:
        response = fallback_chain.invoke({"question": question})
        return response.content

    context = "\n\n".join(relevant_chunks)
    response = chain.invoke({"context": context, "question": question})
    return response.content


if __name__ == "__main__":
    print("Ready! Ask a question (type 'exit' to quit).")
    while True:
        question = input("You: ")
        if question.lower() == "exit":
            break
        answer = ask(question)
        print(f"Bot: {answer}\n")