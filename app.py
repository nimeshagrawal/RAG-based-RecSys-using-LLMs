import sys
sys.path.append("..")

import os
import json
import time
import pandas as pd
from dotenv import load_dotenv
import streamlit as st
from streamlit_extras.mention import mention
import chromadb

from openai import OpenAI
import json
from langchain.vectorstores import Qdrant
from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.memory import ConversationSummaryMemory
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQAWithSourcesChain

# load_dotenv('../.env')

# QDRANT_URL = os.getenv('QDRANT_URL')
# QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')
COLLECTION_NAME = "big-basket-products-all"

K=4

#############################################################################################################
#############################################################################################################

# Promp Template to be used for generating questions
@st.cache_resource(show_spinner=False)
def PROMPT():
    prompt_template = '''
    About: You are a Product Recommendation Agent who gets his context from the retrieved descriptions of the products that matches best with the User's query. 
        User is a human who, as a customer, wants to buy a product from this application.

        Given below is the summary of conversation between you (AI) and the user (Human):
        Context: {chat_history}

        Now use this summary of previous conversations and the retrieved descriptions of products to answer the following question asked by the user:
        Question: {question}

        Note: 
        - Give your answer in a compreshenive manner in enumerated format.
        - Do not generate any information on your own, striclty stick to the provided data. 
        - Also, do not repeat the information that is already present in the context.
        - If, you feel there is redundant information (or) an product is being described twice, specify that as well in the response.
        - The tone of the answer should be like a polite and friendly AI Assistant.
    '''

    return PromptTemplate(
        template=prompt_template, input_variables=["chat_history", "question"]
    )

@st.cache_resource(show_spinner=False)
def load_model():
    try:
        model = ChatOpenAI(
            model='meta-llama/Meta-Llama-3.1-70B-Instruct',
            api_key="7E4hdDQrPP9mLi52rX4zCkJ2rFKIadOk",
            base_url="https://api.deepinfra.com/v1/openai",
            max_tokens = 70000,
        )
    except Exception as e:
        st.error(e)
        model = None
    return model

llm = load_model()

# Memory to store the conversation history
def memory():
    if 'memory' not in st.session_state:
        st.session_state.memory = ConversationSummaryMemory(
            llm=llm,
            memory_key="chat_history",
            return_messages=True,
            input_key="question",
            output_key='answer'
        )
    return st.session_state.memory

class DeepInfraEmbeddings:
    def __init__(self, api_key, base_url, model="BAAI/bge-base-en-v1.5"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def embed_documents(self, texts):
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

# Retriever to retrieve the products from the database
@st.cache_resource(show_spinner=False)
def retriever(K):
    client = chromadb.PersistentClient(path=os.path.join(os.getcwd(), 'vector_stores'))

    embeddings = DeepInfraEmbeddings(
                        api_key="7E4hdDQrPP9mLi52rX4zCkJ2rFKIadOk",
                        base_url="https://api.deepinfra.com/v1/openai"
                    )

    vector_store = Chroma(
                        collection_name=COLLECTION_NAME,
                        embedding_function=embeddings,  # Pass the DeepInfraEmbeddings instance
                        client=client,
                        persist_directory = os.path.join(os.getcwd(), 'vector_stores')
                    )

    retriever = vector_store.as_retriever(search_kwargs={'k':K})

    return retriever

# Chain to chain the retriever with memory
def Chain():
    global K
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever(K),
        memory=memory(),
        return_source_documents=True,
    )

    return chain

# Search function to search for the products
@st.cache_data(show_spinner=False)
def search(_chain, user_question):
    gen_prompt = PROMPT().format(question=user_question, chat_history=memory().load_memory_variables({})['chat_history'][0].content)
    try:
        res = _chain(gen_prompt)
    except Exception as e:
        st.error(e)
        res = None
    return res

#############################################################################################################
#############################################################################################################

# Initialize the app
def init():
    global K

    st.set_page_config(
        page_title="BigBasket Products",
        page_icon="🧺",
        layout="centered",
        initial_sidebar_state="expanded",
    )

    with st.sidebar:

        # mention(
        #     label="Document-Retrieval",
        #     icon="github",
        #     url="https://github.com/AbhishekPardhi/Document-Retrieval",
        # )

        st.subheader('Parameters')
        K = st.slider('K', 1, 10, K, help='Sets max number of products  \nthat can be retrieved')
        
    st.header('BigBasket Products',divider=True)

# Display the retrieved products
def display_data(res):
    try:
        srcs = [json.loads(row.page_content) for row in res['source_documents']]

        df = pd.DataFrame(srcs)
    except Exception as e:
        st.error(e)
        return

    df1 = df[['product','brand', 'sale_price', 'rating', 'description']]

    # Remove duplicates
    df1 = df1.drop_duplicates()

    st.dataframe(
        df1,
        column_config={
            "product": st.column_config.Column(
                "Product Name",
                width="medium"
            ),
            "brand": st.column_config.Column(
                "Brand",
                width="medium"
            ),
            "sale_price": st.column_config.NumberColumn(
                "Sale Price",
                help="The price of the product in USD",
                min_value=0,
                max_value=1000,
                format="₹%f",
            ),
            "rating": st.column_config.NumberColumn(
                "Rating",
                help="Rating of the product",
                format="%f ⭐",
            ),
            "description": "Description",
        },
        hide_index=True,
    )

def main():

    init()

    # Initialize chat history
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello 👋\n\n I am here to help you choose the product that you wanna buy!"}
        ]

    chain = Chain()

    if prompt:=st.chat_input("Say something"): # Prompt for user input and save to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

    for message in st.session_state.messages: # Display the prior chat messages
        with st.chat_message(message["role"]):
            st.write(message["content"], unsafe_allow_html=False)
    
    # If last message is not from assistant, generate a new response
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                start_time = time.time()
                res = search(chain, prompt)
                end_time = time.time()
                st.toast(f'Search completed in :green[{end_time - start_time:.2f}] seconds', icon='✅')
                if res is None:
                    st.error("Something went wrong. Please try again.")
                    return

                answer = res['answer']
                print('answer', answer)
                message = {"role": "assistant", "content": answer}
                st.session_state.messages.append(message) # Add response to message history

                # Display assistant response in chat message container
                message_placeholder = st.empty()
                full_response = ""

                # Simulate stream of response with milliseconds delay
                for chunk in answer.split():
                    full_response += chunk + " "
                    time.sleep(0.05)
                    # Add a blinking cursor to simulate typing
                    message_placeholder.markdown(full_response + "▌", unsafe_allow_html=False)
                message_placeholder.markdown(full_response, unsafe_allow_html=False)

                # Dsiplay product details
                display_data(res)

if __name__ == "__main__":
    main()
