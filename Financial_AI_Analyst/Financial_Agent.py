# Importing necessary libraries and modules
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import openai

import os
from dotenv import load_dotenv

# Loading environment variables from .env file
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")  # Setting the OpenAI API key

# Defining a web search agent using DuckDuckGo tool
web_search_agent = Agent(
    name="Web Search Agent",  # Name of the agent
    role="Search the web for the information",  # Role description
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),  # Model used for the agent
    tools=[DuckDuckGo()],  # Adding DuckDuckGo tool for web search
    instructions=["Always include sources"],  # Instructions to the agent
    show_tools_calls=True,  # Enable display of tool calls during execution
    markdown=True,  # Format responses in markdown
)

# Defining a financial data agent using YFinanceTools
finance_agent = Agent(
    name="Finance AI Agent",  # Name of the agent
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),  # Model used for the agent
    tools=[
        YFinanceTools(
            stock_price=True,  # Enable stock price retrieval
            analyst_recommendations=True,  # Enable analyst recommendations retrieval
            stock_fundamentals=True,  # Enable stock fundamentals retrieval
            company_news=True,  # Enable company news retrieval
        ),
    ],
    instructions=["Use tables to display the data"],  # Instruction to present data in tables
    show_tool_calls=True,  # Enable display of tool calls during execution
    markdown=True,  # Format responses in markdown
)

# Creating a multi-agent system to leverage both web search and finance agents
multi_ai_agent = Agent(model=Groq(id="llama3-groq-70b-8192-tool-use-preview")),
    team=[web_search_agent, finance_agent],  # Team of agents working together
    instructions=["Always include sources", "Use table to display the data"],  # Shared instructions for the team
    show_tool_calls=True,  # Enable display of tool calls during execution
    markdown=True,  # Format responses in markdown
)

# Sending a query to the multi-agent system
multi_ai_agent.print_response(
    "Summarize analyst recommendation and share the latest news for NVDA",  # Query for the agents
    stream=True,  # Enable streaming of the response
)
