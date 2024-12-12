from langchain.agents import Tool
import requests
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama.llms import OllamaLLM
def initialize_llm():
    path = "Your_model_path"
    return OllamaLLM(model=path, host="localhost", port=11434)
def load_vectorstore(path='./faiss_index'):
    return FAISS.load_local(path, HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"), allow_dangerous_deserialization=True)

vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
llm = initialize_llm()

class SummarizerTool(Tool):
    name: str = "Summarizer"
    description: str = "Summarize drug information based on the JSON vector index."

    def _call(self, query: str) -> str:
        docs = retriever.get_relevant_documents(query)
        combined_docs = " ".join([doc.page_content for doc in docs])
        return f"Here is a summary of the drug information: {combined_docs[:100]}..."  # Truncate for demo

    def __init__(self, name: str = name, func: str = _call, description: str = description, **kwargs):
        """Initialize the SummarizerTool."""
        # If func is None, use the _call method as the default function
        func = self._call
        super().__init__(name=name, func=func, description=description, **kwargs)

class RecommenderTool(Tool):
    name: str = "Recommender"
    description: str = "Recommend drugs based on symptoms using the vector index. If no recommendations are found, return a message indicating that no recommendations were found. Only use the drugs that are in the vector index."

    def _call(self, query: str) -> str:
        docs = retriever.get_relevant_documents(query)
        recommendations = []
        for doc in docs:
            content = doc.page_content.lower()
            if "contraindication" in content or "recommended" in content:
                recommendations.append(content[:100])
        if not recommendations:
            return "I couldn't find any specific drug recommendations based on your symptoms."
        return f"Based on your symptoms, here are some drugs to consider: {'; '.join(recommendations)}"

    def __init__(self, name: str = name, func: str = _call, description: str = description, **kwargs):
        """Initialize the SummarizerTool."""
        # If func is None, use the _call method as the default function
        func = self._call
        super().__init__(name=name, func=func, description=description, **kwargs)


class AlternativesGeneratorTool(Tool):
    name: str = "Alternatives_Generator"
    description:str = "Suggest alternative drugs based on symptoms or specific conditions. Use the JSON vector index to find relevant information. If no alternatives are found, return a message indicating that no alternatives were found."

    def _call(self, query: str) -> str:
        docs = retriever.get_relevant_documents(query)
        alternatives = []
        for doc in docs:
            if "alternative" in doc.page_content.lower():
                alternatives.append(doc.page_content[:100])
        if not alternatives:
            return "I couldn't find any alternative drugs for your symptoms."
        return f"Here are some alternatives you could consider: {'; '.join(alternatives)}"
    def __init__(self, name: str = name, func: str = _call, description: str = description, **kwargs):
        """Initialize the SummarizerTool."""
        # If func is None, use the _call method as the default function
        func = self._call
        super().__init__(name=name, func=func, description=description, **kwargs)


class LLMDecisionTool(Tool):
    name:str = "LLMDecision"
    description:str = "Decide the type of user query and route it to the appropriate tool. Use the LLM model to make the decision. The choices are: Summarize drug information, Recommend a drug based on symptoms, Suggest alternative drugs based on symptoms or conditions, Perform a web search if no drug information is found."
    global llm
    def _call(self, query: str) -> str:
        prompt = f"""
        You are a pharmaceutical assistant agent. Analyze the following user query and decide if the user is asking to:
        - Summarize drug information.
        - Recommend a drug based on symptoms.
        - Suggest alternative drugs based on symptoms or conditions.
        - Perform a web search if no drug information is found.

        User query: "{query}"
        """
        decision = llm(prompt)
        return decision.strip()
    def __init__(self, name: str = name, func: str = _call, description: str = description, **kwargs):
        """Initialize the SummarizerTool."""
        # If func is None, use the _call method as the default function
        func = self._call
        super().__init__(name=name, func=func, description=description, **kwargs)


class WebRetrievalTool(Tool):
    def __init__(self):
        super().__init__(
            name="Web_Retrieval_Tool",
            func=self._call,
            description="Retrieve information from the web."
        )

    def _call(self, query: str) -> str:
        # Simple web retrieval implementation
        try:
            response = requests.get(f"https://api.duckduckgo.com/?q={query}&format=json")
            data = response.json()
            abstract = data.get("Abstract", "")
            if abstract:
                return abstract
            else:
                return "No information found on the web."
        except Exception as e:
            return f"Error during web retrieval: {e}"
