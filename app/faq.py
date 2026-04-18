import pandas as pd
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
import os

from dotenv import load_dotenv
from groq import Groq
from rich import prompt

load_dotenv()

faqs_path = Path(__file__).parent / "resources/faq_data.csv"
chroma_clients = chromadb.Client()
collection_name_faq = 'faqs'

ef=embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name = 'all-MiniLM-L6-v2'
)

def ingest_faq_data(path):
    if collection_name_faq not in [c.name for c in chroma_clients.list_collections()]:
        print("Creating collection...")
        collection = chroma_clients.get_or_create_collection(
            name=collection_name_faq,
            embedding_function=ef
        )

        df = pd.read_csv(faqs_path)
        docs = df['question'].tolist()
        metadata = [{'answer':ans} for ans in df['answer'].tolist()]
        ids = [f"id_{i}" for i in range(len(docs))]

        collection.add(
            documents=docs,
            metadatas=metadata,
            ids=ids
        )
        print("Collection created!")
    else:
        print("Collection already exists")

def get_relevant_qa(query):
    collection = chroma_clients.get_collection(
        name=collection_name_faq
    )
    result = collection.query(
        query_texts=[query],
        n_results=2
    )
    return result

def faq_chain(query):
    result = get_relevant_qa(query)
    context = ''.join([r.get('answer') for r in result["metadatas"][0]])
    answer = generate_answer(query,context)
    return answer

def generate_answer(query,context):
    prompt = f'''Given the question and context below , generate the  answer based on content only.
        If don't find the answer inside the context then say i don't know
        Do not make things up.

        Question : {query}

        Context : {context}
        '''
    # call lmm
    client = Groq(
        api_key=os.environ.get("GROQ_API_KEY"),
    )
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model=os.environ['GROQ_MODEL'],
    )

    return chat_completion.choices[0].message.content

if __name__ == "__main__":
    # print(faqs_path)
    ingest_faq_data(faqs_path)
    query = "what's your policy on defective product"
    # result = get_relevant_qa(query)
    answer = faq_chain(query)
    print(answer)