from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from tavily import TavilyClient

from pydantic import BaseModel, Field
from typing import List

load_dotenv()

tavily_client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))

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

@tool
def search(query: str) -> str:
    """
    Tool that searches over internet
    Args:
        query (str): The search query
    Returns:
        str: The search results
    """
    print(f"Searching for '{query}'...")
    return tavily_client.search(query=query)


def main():
    print("Hello from langchain-course!")
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
    # print(GEMINI_API_KEY)
    # information = """
    #             Elon Reeve Musk (/ˈiːlɒn/ EE-lon; born June 28, 1971) is a businessman and entrepreneur known for his leadership of Tesla, SpaceX, Twitter, and xAI. Musk has been the wealthiest person in the world since 2025; as of February 2026, Forbes estimates his net worth to be around US$852 billion.

    #         Born into a wealthy family in Pretoria, South Africa, Musk emigrated in 1989 to Canada; he has Canadian citizenship since his mother was born there. He received bachelor's degrees in 1997 from the University of Pennsylvania before moving to California to pursue business ventures. In 1995, Musk co-founded the software company Zip2. Following its sale in 1999, he co-founded X.com, an online payment company that later merged to form PayPal, which was acquired by eBay in 2002. Musk also became an American citizen in 2002.

    #         In 2002, Musk founded the space technology company SpaceX, becoming its CEO and chief engineer; the company has since led innovations in reusable rockets and commercial spaceflight. Musk joined the automaker Tesla as an early investor in 2004 and became its CEO and product architect in 2008; it has since become a leader in electric vehicles. In 2015, he co-founded OpenAI to advance artificial intelligence (AI) research, but later left; growing discontent with the organization's direction and their leadership in the AI boom in the 2020s led him to establish xAI, which became a subsidiary of SpaceX in 2026. In 2022, he acquired the social network Twitter, implementing significant changes, and rebranding it as X in 2023. His other businesses include the neurotechnology company Neuralink, which he co-founded in 2016, and the tunneling company the Boring Company, which he founded in 2017. In November 2025, a Tesla pay package worth $1 trillion for Musk was approved, which he is to receive over 10 years if he meets specific goals.

    #         Musk was the largest donor in the 2024 U.S. presidential election, where he supported Donald Trump. After Trump was inaugurated as president in early 2025, Musk served as Senior Advisor to the President and as the de facto head of the Department of Government Efficiency (DOGE). After a public feud with Trump, Musk left the Trump administration and returned to managing his companies. Musk is a supporter of global far-right figures, causes, and political parties. His political activities, views, and statements have made him a polarizing figure. Musk has been criticized for COVID-19 misinformation, promoting conspiracy theories, and affirming antisemitic, racist, and transphobic comments. His acquisition of Twitter was controversial due to a subsequent increase in hate speech and the spread of misinformation on the service, following his pledge to decrease censorship. His role in the second Trump administration attracted public backlash, particularly in response to DOGE. The emails he sent to Jeffrey Epstein are included in the Epstein files, which were published between 2025–26 and became a topic of worldwide debate.
    #         """
    # summary_template = """
    #             given the information {information} about a person, I want you to create:
    #             1. a short summary
    #             2. two interesting facts about the person
    #             """
    # summary_prompt_template = PromptTemplate(
    #     template=summary_template,
    #     input_variables=["information"]
    # )
    # llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.9, api_key=GEMINI_API_KEY)
    # chain = summary_prompt_template | llm
    # response = chain.invoke(input={"information": information})
    # print(response.content)


    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.9, api_key=GEMINI_API_KEY)
    # llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.9, api_key=GEMINI_API_KEY)
    tools = [search]
    agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)
    response = agent.invoke({
        "messages": [
            #  HumanMessage(content="What is the weather in Tokyo?")
             HumanMessage(content="search for 3 job openings for generative AI engineers using langchain in San Francisco on linkedin and list their details. the jobs should have been posted in the last 1 month only")
        ]   
    })
    print(response)
    print("===============================================================\n")
    print(response["structured_response"].answer)
    print("===============================================================\n")
    print(response["structured_response"].sources)

if __name__ == "__main__":
    main()

