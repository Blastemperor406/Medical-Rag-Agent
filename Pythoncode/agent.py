from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import AgentAction, AgentFinish
from langchain.agents.agent import AgentExecutor
from langchain.agents import ZeroShotAgent, Tool
from tools import SummarizerTool, RecommenderTool, AlternativesGeneratorTool, WebRetrievalTool, LLMDecisionTool

def load_vectorstore(path='./faiss_index'):
    return FAISS.load_local(path, HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"), allow_dangerous_deserialization=True)

def initialize_llm():
    path = "hostllama"
    return OllamaLLM(model=path, host="localhost", port=11434)

def initialize_agent():
    llm = initialize_llm()

    Summarizer = SummarizerTool()
    Recommender = RecommenderTool()
    Alternatives_Generator = AlternativesGeneratorTool()
    web_retrieval_tool = WebRetrievalTool()
    LLMDecision = LLMDecisionTool()

    tools = [LLMDecision, Summarizer, Recommender, Alternatives_Generator, web_retrieval_tool]
    agent = ZeroShotAgent.from_llm_and_tools(llm=llm, tools=tools,verbose=True)

    return agent, tools

def process_query(agent, tools, query):
    

    agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True, max_iterations=20)
    response = agent_executor.run(query)
    #chain_of_thought = agent_executor.get_chain_of_thought()
    return response