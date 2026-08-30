from typing import TypedDict

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from ddgs import DDGS

# ---- Same setup as langchain_rag.py ----
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

web_prompt_template = ChatPromptTemplate.from_template("""Answer the question using the web search results below.
Be concise and factual. If the results don't clearly answer the question, say so.

Web search results:
{context}

Question: {question}

Answer:""")

web_chain = web_prompt_template | llm

classify_template = ChatPromptTemplate.from_template("""Classify the user's message as exactly one word: "smalltalk" or "question".
"smalltalk" = greetings, thanks, casual chat, no real information being asked for.
"question" = the user is asking for a real fact, even if unrelated to any specific document.

Message: {question}

Classification (one word only):""")

classify_chain = classify_template | llm

SIMILARITY_THRESHOLD = 1.0


# ---- Web search tool ----

def web_search(query: str, max_results: int = 3) -> list[str]:
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=max_results)
        return [r["body"] for r in results]


# ---- LangGraph state ----

class ChatState(TypedDict):
    question: str
    relevant_chunks: list[str]
    intent: str
    answer: str


# ---- LangGraph nodes ----

def retrieve(state: ChatState) -> dict:
    print("[NODE] retrieve")
    results_with_scores = vector_store.similarity_search_with_score(state["question"], k=4)
    relevant = [doc.page_content for doc, score in results_with_scores if score < SIMILARITY_THRESHOLD]
    return {"relevant_chunks": relevant}


def route_after_retrieve(state: ChatState) -> str:
    if not state["relevant_chunks"]:
        return "classify_intent"
    else:
        return "generate_answer"


def classify_intent(state: ChatState) -> dict:
    print("[NODE] classify_intent")
    response = classify_chain.invoke({"question": state["question"]})
    label = response.content.strip().lower()
    return {"intent": label}


def route_after_classify(state: ChatState) -> str:
    if "question" in state["intent"]:
        return "web_search_node"
    else:
        return "generate_fallback"


def generate_answer(state: ChatState) -> dict:
    print("[NODE] generate_answer")
    context = "\n\n".join(state["relevant_chunks"])
    response = chain.invoke({"context": context, "question": state["question"]})
    return {"answer": response.content}


def generate_fallback(state: ChatState) -> dict:
    print("[NODE] generate_fallback")
    response = fallback_chain.invoke({"question": state["question"]})
    return {"answer": response.content}


def web_search_node(state: ChatState) -> dict:
    print("[NODE] web_search_node")
    results = web_search(state["question"])
    context = "\n\n".join(results)
    response = web_chain.invoke({"context": context, "question": state["question"]})
    return {"answer": response.content}


# ---- Build the graph ----

graph_builder = StateGraph(ChatState)

graph_builder.add_node("retrieve", retrieve)
graph_builder.add_node("classify_intent", classify_intent)
graph_builder.add_node("generate_answer", generate_answer)
graph_builder.add_node("generate_fallback", generate_fallback)
graph_builder.add_node("web_search_node", web_search_node)

graph_builder.set_entry_point("retrieve")

graph_builder.add_conditional_edges(
    "retrieve",
    route_after_retrieve,
    {
        "generate_answer": "generate_answer",
        "classify_intent": "classify_intent",
    },
)

graph_builder.add_conditional_edges(
    "classify_intent",
    route_after_classify,
    {
        "web_search_node": "web_search_node",
        "generate_fallback": "generate_fallback",
    },
)

graph_builder.add_edge("generate_answer", END)
graph_builder.add_edge("generate_fallback", END)
graph_builder.add_edge("web_search_node", END)

graph = graph_builder.compile()


def ask(question: str) -> str:
    result = graph.invoke({"question": question})
    return result["answer"]


if __name__ == "__main__":
    print("Ready! Ask a question (type 'exit' to quit).")
    while True:
        question = input("You: ")
        if question.lower() == "exit":
            break
        answer = ask(question)
        print(f"Bot: {answer}\n")