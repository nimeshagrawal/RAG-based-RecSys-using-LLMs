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
1. Install the required packages:\
   ```pip install -r requirements.txt```
2. Create your account in `DeepInfra` and generate your API key for utilising the embedding and LLM models.
3. Create your own local vector-database in chromaDB by running the following:\
   ``
### Sample conversation
```
[USER] : Suggest some good beverages
[AGENT]: Here are some good beverage options for you:

      1. **Tea - Yellow Label** by Lipton: A high-quality tea with a rich taste and aroma, perfect for a boost of energy and a burst of sunshine.
      2. **Peach Tea Flavoured Syrup** by DaVinci Gourmet: A delicious syrup that can be used to make peach ice teas for a house party or for your loved ones.
      3. **Lemon Tea Flavoured Syrup** by DaVinci Gourmet: Similar to the peach syrup, this lemon syrup can be used to make lemon ice teas for a refreshing drink.
      
      Note that the **100% Anti Hangover Drink - Blackberry** by ARMR is not a traditional beverage, but rather a pre-party shot made from herbs and ingredients to help prevent hangovers.
      
      Also, while the **Muffin/Cup Cake - Vanilla** by Fresho Signature is not a beverage itself, it is mentioned as a great accompaniment to tea time.

[USER] : "Okay, give me some good recommendation for Baby Care products"
[AGENT]: Here are some baby care products that might interest you:

      1. **Baby Premium Diaper - Small** by Mee Mee: Made from Cloth-like Breathable Fabric, eco-friendly, with a leak-lock system and SAP for absorbency. (Note: This product has already been recommended earlier, but I'm mentioning it again as it's a great option for babies with sensitive skin.)
      2. **Powder Puff - Soft Feel, Green** by Mee Mee: A soft and premium quality powder puff designed for babies, allowing for smooth and even application of powder without causing scratches or itching on delicate skin.
      3. **Diaper Pants - XX-Large** by Pampers: Provides breathable dryness overnight, with magic gel that locks wetness away for up to 12 hours, and a comfortable fit with a flexible waistband.
      4. **Diaper Pants - X-Large** by Pampers: Similar to the previous product, with breathable dryness, magic gel, and a comfortable fit, but in a different size.
      
      Please note that I've only provided information on products that match your query and are relevant to baby care. If you'd like more options or have specific preferences, feel free to let me know!

[USER] : "Tell me more about the diaper product"
[AGENT]: I'm happy to help you with your question about the diaper product!

      Here are some additional details about the **Baby Premium Diaper - Small** by Mee Mee:
      
      1. The diaper is made from a cloth-like breathable fabric, which is eco-friendly.
      2. The leak-lock system turns liquids into a gel, providing extra protection against leaks.
      3. The SAP (Superabsorbent Polymers) absorbs multiple wettings, ensuring that the diaper stays dry for a longer period.
      4. The diaper features leak-proof technology, giving you and your baby peace of mind.
      
      Please note that the provided information is strictly based on the retrieved product description, and I've avoided repeating any information already present in the context.
      
      If you have any further questions or would like to know more about other products, feel free to ask!

[USER] : "Compare all the diaper products and give me a good recommendation with reason"
[AGENT]: Based on the provided product descriptions, here's a comparison of the diaper products and a recommendation:

      1. **Baby Premium Diaper - Small** by Mee Mee: This diaper features a cloth-like breathable fabric, eco-friendly design, and a leak-lock system that turns liquids into a gel. It also uses SAP (Superabsorbent Polymers) for multiple wettings and has leak-proof technology.
      
      2. **Diaper Pants - XX-Large** by Pampers: This diaper provides breathable dryness overnight with its air channels and Magic Gel that locks wetness away for up to 12 hours. It also has a flexible waistband and a top layer with cotton-like soft material.
      
      3. **Diaper Pants - X-Large** by Pampers: This diaper has similar features to the XX-Large version, including air channels, Magic Gel, and a flexible waistband.
      
      4. **Easy Adult Diaper - Medium** by Friends: This adult diaper provides up to 8 hours of protection and has a high absorbency rate with a wetness indicator.
      
      5. **Baby Happy Diapers - Maxi** by Bella: This diaper features a 360° comfort system, anatomical absorbent core, and airflow technology for breathability. It also has a quick dry system and flexi-fit elastic elements.
      
      Recommendation: Based on the provided information, I would recommend the **Baby Happy Diapers - Maxi** by Bella. This diaper seems to offer a comprehensive set of features that prioritize the baby's comfort and skin health, including breathability, quick dryness, and a flexible fit system. Additionally, it is free from perfume, chlorine, and latex, making it a suitable option for sensitive skin.
      
      Please note that the **Easy Adult Diaper - Medium** by Friends is not suitable for babies, so it's not considered in the recommendation.
      
      Also, the **Diaper Pants - XX-Large** and **Diaper Pants - X-Large** by Pampers have similar features, so they are not described separately to avoid redundancy.
```
