from langchain.agents import create_react_agent
from langchain_community.agent_toolkits.load_tools import load_tools


def invoke_agent(llm, prompt):
    tools = load_tools(["llm-math"], llm=llm)
    agent = create_react_agent(tools=tools, llm=llm, prompt=prompt)
    return agent.invoke(prompt)
