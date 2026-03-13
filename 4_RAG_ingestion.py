from dotenv import load_dotenv
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()


if __name__ == "__main__":
    loader = TextLoader("vector_db_medium_doc.txt")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    print(f"Number of chunks: {len(texts)}")



    #============== to check the Pinecone index's dimension

    # dimension should be "3072" while creating the index in Pinecone dashboard

    # from pinecone import Pinecone
    # import os
    # pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    # index = pc.Index(os.environ["PINECONE_INDEX_NAME"])
    # print(pc.describe_index(os.environ["PINECONE_INDEX_NAME"]))

    

   #============== to check the available embedding models in gemini api

    # from google import genai
    # client = genai.Client(api_key="GEMINI_API_KEY")
    # for m in client.models.list():
    #     print(m.name)



    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", api_key=os.environ.get("GEMINI_API_KEY"))



    #============== to check the embeddings

    # vector = embeddings.embed_query("hello world")
    # print(vector)



    vector_store = PineconeVectorStore.from_documents(
        texts,
        embedding=embeddings,
        index_name=os.environ.get("PINECONE_INDEX_NAME")
    )
    print("Ingested data into Pinecone vector store successfully!")
