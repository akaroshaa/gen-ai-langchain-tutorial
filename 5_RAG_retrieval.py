from operator import itemgetter
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import TextLoader

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", api_key=os.environ.get("GEMINI_API_KEY"))
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.9, api_key=os.environ.get("GEMINI_API_KEY"))
vector_store = PineconeVectorStore(
    embedding=embeddings,
    index_name=os.environ.get("PINECONE_INDEX_NAME")
)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})       # retrieving top 3 relevant chunks from the vector store
prompt_template = ChatPromptTemplate.from_template(
    """
    Answer the question based only on the following context. If you don't know the answer, say you don't know.
    {context}
    Question: {question}
    Provide a detailed answer:
    """
)


def format_docs(chunks):
    """ format retrieved documents into a string to be added in the prompt template's context variable """
    # this string will be added in the prompt template's {context} variable and will be used by the LLM to answer the question based on the retrieved documents
    return "\n\n".join([f"Source {i+1}: {chunk.page_content}" for i, chunk in enumerate(chunks)])


def create_retrieval_chain():
    retrieval_chain = (
                        RunnablePassthrough.assign(
                            context = itemgetter("question") | retriever | format_docs
                        )                       # to create a dict with new variable "context" by retrieving the relevant documents for the input question and formatting them into a string
                        | prompt_template
                        | llm
                        | StrOutputParser()     # to parse the LLM's response into a string
                    )
    return retrieval_chain


if __name__ == "__main__":
    query = "what is pinecone in machine learning?"
    # docs = retriever.invoke(query)      # to check the retrieved documents for the query
    # context = format_docs(docs)        # formatting the retrieved documents into a string to be added in the prompt template's context variable
    # final_prompt = prompt_template.format_prompt(context=context, question=query)
    # response = llm.invoke(final_prompt)
    # print(response.content)

    retrieval_chain = create_retrieval_chain()
    response = retrieval_chain.invoke({"question": query})
    print(response)
