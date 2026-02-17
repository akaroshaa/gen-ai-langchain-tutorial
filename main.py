from dotenv import load_dotenv
import os

load_dotenv()


def main():
    print("Hello from langchain-course!")
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
    print(GEMINI_API_KEY)


if __name__ == "__main__":
    main()
