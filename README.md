# RAG based RecSys using LLMs

This work presents a real-time product recommendation system based on **Retrieval-Augmented Generation (RAG)**, designed to engage users with insightful recommendations. The system leverages the **ChromaDB** vector database, supported with the **BAAI/bge-large-en-v1.5** model, and employs a RAG-integrated **LangChain** pipeline for advanced semantic search. Powered by the open-source **LLama-3.1-70b-instruct model**, this solution is deployed as a **Streamlit API**, enabling efficient and interactive querying.


### Demo 

### Design overview
#### Vector-database
- We utilise the Chromadb package to generate a localised version of vector-database. To generate the database, we create the embbeddings of the product database using an open-source **BAAI/bge-large-en-v1.5** embedding generator with embedding dimension of 1024. The vector database are further stored in chunks of 32 for faster retreival.
- Based on the user-query, a total of K recommendations can be extracted.

#### Langchain
- A customised prompt that analyses chat history and current user query which further powers the LLM to provide valuable suggestions to the user.
- The RecSys utilizes the **RetrievalQAWithSourcesChain** module alongside **ConversationSummaryMemory** to preserve chat history, integrating it with retrieved product documents from ChromaDB database to effectively address follow-up queries.
- We introduced a feature that allows seamless **context switching** within the same window through an **auxiliary agent**, which detects when the user seeks recommendations in a different category. This functionality leverages a secondary prompt to analyze chat history and the current query, determining contextual similarity. If a shift is detected, the system resets memory and begins anew, ensuring smooth transitions without disrupting the LLM's behavior.

#### Streamlit Frontend
- A web-interface is designed to abstract the underlying framework and provides the user the with a conversational recommendation system.
- The search parameter K can be adjusted in the web-interface, to limit the maximum number of products retrieved by the chain.
- The streamlit app can be run locally using the following command:
      ```
        streamlit run app.py
      ```

### Setup

### Sample
