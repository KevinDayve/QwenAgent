from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.models.google import Gemini
import requests
from io import BytesIO
from module import QwenModelTool
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.mcp import MCPTools
from tool import ModelHTTP
from misc import ImageDescriptionTool
from agno.media import Image
from dotenv import load_dotenv
import os
# qwenTool = QwenGRPOTool()
load_dotenv()
apiKey = os.getenv("GEMINI_API_KEY")
oaiApiKey = os.getenv("OPENAI_API_KEY")
mcpTool = MCPTools(
    url='http://localhost:8000/mcp',
    session='sse',
    timeout_seconds=30
)

agent = Agent(
    model=Gemini(
        id="gemini-2.0-flash-exp",
        api_key=apiKey,
        show_tool_calls=True,
        max_output_tokens=256,
    ),
    description="You are an expert visual analyser. Please you the tools at your disposal any how.",
    tools=[ImageDescriptionTool()],
    instructions=[
        "You MUST use the provided tool to generate your response. Unless explicitly asked not to.",
        "Do not answer the user's query directly using your own knowledge, unless explicitly asked to.",
        "Invoke the tool for every relevant user request, unless explicitly asked not to.",
    ],
    show_tool_calls=True,
    markdown=True,
)

URL = "https://buffer.com/resources/content/images/2024/11/free-stock-image-sites.png"
agent.print_response(
    f"Describe this image, here is the URL: {URL}.",
    stream=True
)