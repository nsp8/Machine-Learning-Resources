from langchain.agents import load_tools, initialize_agent, AgentType
from text_generator import TextGenerator


def agent():
    m = TextGenerator()
    tools = load_tools(["wikipedia", "llm-math"], llm=m.model)

