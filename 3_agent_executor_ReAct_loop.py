from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field
from typing import List


load_dotenv()

class Source(BaseModel):
    """
    Schema for a source used by the agent
    """
    url: str = Field(description="The URL of the source")

class AgentResponse(BaseModel):
    """
    Schema for the response returned by the agent
    """
    answer: str = Field(description="The answer to the user's query")
    sources: List[Source] = Field(default_factory=list, description="The sources used to generate the answer")


if __name__ == "__main__":
    print("Hello from langchain-course!")
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.9, api_key=GEMINI_API_KEY)
    tools = [TavilySearch(api_key=os.environ.get("TAVILY_API_KEY"))]
    agent = create_agent(model=llm, tools=tools)
    result = agent.invoke({
                "messages": [
                    {
                        "role": "user",
                        "content": "search for 3 job openings for generative AI engineers using langchain in San Francisco on linkedin and list their details. the jobs should have been posted in the last month only"
                    }
                ]
            })
    # print(result)

    final_result = llm.with_structured_output(AgentResponse).invoke(result["messages"][-1].content)
    print(final_result)
