from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from tavily import TavilyClient

load_dotenv()

MAX_ITERATIONS = 10

@tool
def get_product_price(product: str) -> float:
    """
    Tool that gets the price of a product
    Args:
        product (str): The name of the product
    Returns:
        float: The price of the product
    """
    print(f"Getting price for '{product}'...")
    prices = {
        "laptop": 1299.99,
        "keyboard": 89.50,
        "headphones": 149.95
    }
    return prices.get(product, 0.0)

@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount to a price based on the discount tier."""
    discount_rates = {
        "bronze": 0.05,
        "silver": 0.12,
        "gold": 0.23
    }
    discount = discount_rates.get(discount_tier, 0)
    return round(price - (price * discount), 2)


def run_agent(question: str):
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
    # llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.9, api_key=GEMINI_API_KEY)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.9, api_key=GEMINI_API_KEY)
    # llm = init_chat_model(model="gemini-2.5-flash-lite", temperature=0.9)
    tools = [get_product_price, apply_discount]
    llm_with_tools = llm.bind_tools(tools)
    tools_dict = {t.name: t for t in tools}

    messages = [
        SystemMessage(
            content=(
                "You are a helpful shopping assistant. "
                "You have access to a product catalog tool "
                "and a discount tool.\n\n"
                "STRICT RULES — you must follow these exactly:\n"
                "1. NEVER guess or assume any product price. "
                "You MUST call get_product_price first to get the real price.\n"
                "2. Only call apply_discount AFTER you have received "
                "a price from get_product_price. Pass the exact price "
                "returned by get_product_price — do NOT pass a made-up number.\n"
                "3. NEVER calculate discounts yourself using math. "
                "Always use the apply_discount tool.\n"
                "4. If the user does not specify a discount tier, "
                "ask them which tier to use — do NOT assume one."
                "5. Product names MUST be normalized to lowercase singular form "
                '(e.g., "a laptop", "the laptop", "laptops" → "laptop") '
                "before calling tools."
            )
        ),
        HumanMessage(content=question),
    ]

    for i in range(1, MAX_ITERATIONS+1):
        print(f"\n--- Iteration {i} ---")
        ai_message = llm_with_tools.invoke(messages)
        tool_calls = ai_message.tool_calls
        if not tool_calls:
            print(f"\nAnswer found by AI...\n")
            return ai_message.content
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args")
        tool_id = tool_call.get("id")
        print(f"AI called tool: {tool_name} with args: {tool_args}")
        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            return f"Error: Tool '{tool_name}' not found."
        observation = tool_to_use.invoke(tool_args)
        print(f"Tool Result: {observation}")
        messages.append(ai_message)
        messages.append(ToolMessage(content=str(observation), tool_name=tool_name, tool_call_id=tool_id))
    return "\nLLM Error: Reached max iterations without a final answer."


if __name__ == "__main__":
    print("Hello from langchain-course!")
    response = run_agent("What is the price of a laptop after applying a gold discount?")
    print(response)
