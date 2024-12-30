# Import required libraries
import openai  # OpenAI library for language model integration
from phi.agent import Agent  # Agent class for creating custom agents
import phi.api  # API support for Phi
from phi.model.groq import Groq
from phi.model.openai import OpenAIChat  # OpenAI chat model for natural language tasks
from phi.tools.yfinance import YFinanceTools  # Tools for financial data handling
from phi.tools.duckduckgo import DuckDuckGo  # DuckDuckGo tool for web search
from dotenv import load_dotenv  # To load environment variables from a .env file
import os  # OS module for environment variables handling
import phi  # Phi framework for creating AI playgrounds
from phi.playground import Playground, serve_playground_app  # For creating and serving the Playground app

# Load environment variables from a .env file
load_dotenv()  # Ensures the .env file is loaded to set environment variables

# Setting the Phi API key from the environment variable
phi.api = os.getenv("PHI_API_KEY")  # Fetch the API key for Phi services from the .env file

# Define a web search agent
web_search_agent = Agent(
    name="Web Search Agent",  # Name of the agent
    role="Search the web for the information",  # Role or description of the agent's functionality
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),  # Groq model for processing search queries
    tools=[DuckDuckGo()],  # Tool used for web search
    instructions=["Always include sources"],  # Specific instructions for the agent
    show_tools_calls=True,  # Show tool calls in the output
    markdown=True,  # Format output in Markdown for readability
)

# Define a finance agent
finance_agent = Agent(
    name="Finance AI Agent",  # Name of the agent
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),  # Groq model for financial data processing
    tools=[
        YFinanceTools(
            stock_price=True,  # Enables stock price retrieval
            analyst_recommendations=True,  # Fetches analyst recommendations
            stock_fundamentals=True,  # Retrieves stock fundamentals
            company_news=True  # Fetches recent news about the company
        ),
    ],
    instructions=["Use tables to display the data"],  # Instruction for output format (tables)
    show_tool_calls=True,  # Display tool calls in the output
    markdown=True,  # Format output in Markdown
)

# Create a Playground app and register the agents
app = Playground(agents=[finance_agent, web_search_agent]).get_app()

# Serve the Playground app when the script is executed
if __name__ == "__main__":
    serve_playground_app("playground:app", reload=True)  # Serve the app with live reload for development
