import sys
sys.path.append("..")
import os.path
import pandas as pd
import time
from tqdm import tqdm
import chromadb
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from openai import OpenAI

from openai import OpenAI

import yaml
# Read YAML file
with open("config.yaml", 'r') as stream:
    CONFIG = yaml.safe_load(stream)

# Wrapper for DeepInfraEmbeddings generation
class DeepInfraEmbeddings:
    def __init__(self, api_key, base_url, model="BAAI/bge-large-en-v1.5"):
        """Intialise client to access embedding model

        Args:
            api_key (str): Deep-Infra API key
            base_url (str): URL to access the embeddings
            model (str, optional): 1024 dimension embeddings. Defaults to "BAAI/bge-large-en-v1.5".
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def embed_documents(self, texts):
        """Converts given INPUT data to corresponding embeddings

        Args:
            texts (str): INPUT database contents as string.

        Returns:
            list: List of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.client.embeddings.create(
            model=self.model,
            input=texts,
            encoding_format="float"
        )

        return [embedding.embedding for embedding in embeddings.data]

    def embed_query(self, text):
        return self.embed_documents([text])[0]



# CREATE A LOCAL CHROMA_DB WITH  PERSISTENT STORAGE
client = chromadb.PersistentClient(path=os.path.join(os.getcwd(), 'vector_stores'))

# LOAD THE DATA_PATH
file_path = os.path.join(CONFIG["DATA_PATH"])
df = pd.read_csv(file_path)
metadatas = [{'source': int(df.loc[i][0]), 'row': i} for i in range(len(df))]
docs = df.apply(lambda x: x.to_json(), axis=1).tolist()

# Initialize DeepInfraEmbeddings with your API key and base URL
embeddings = DeepInfraEmbeddings(
    api_key=CONFIG["API_KEY"],
    base_url=CONFIG["BASE_URL"]
)

# Create Chroma collection
vector_store = Chroma(
    collection_name=CONFIG["COLLECTION_NAME"],
    embedding_function=embeddings,  # Pass the DeepInfraEmbeddings instance
    client=client,
    persist_directory = os.path.join(os.getcwd(), 'vector_stores')
)

# Store the processed embeddings into the vector_store in chunks
retries_dict = {}
CHUNK_SIZE = 32
for i in tqdm(range(0, len(docs), CHUNK_SIZE)):
    try:
        vector_store.add_texts(
            texts=docs[i:i+CHUNK_SIZE],
            metadatas=metadatas[i:i+CHUNK_SIZE],
            ids=[str(x) for x in range(i, i+CHUNK_SIZE)]
        )
    except Exception as e:
        print(i, e)
        i = i - CHUNK_SIZE
        retries_dict[i] = retries_dict.get(i, 0) + 1
        if retries_dict[i] > 5:
            print(f"Failed to add documents at index {i} after 3 retries. Skipping...")
            i += CHUNK_SIZE
            continue
        time.sleep(1)

