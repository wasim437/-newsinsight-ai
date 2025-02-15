import os
import pickle
import time
import requests
import streamlit as st
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# ✅ Streamlit App
st.title("NewsInsight: AI-Powered News Research Tool 📰")
st.sidebar.title("News Article URLs")

# ✅ Collect URLs
urls = [st.sidebar.text_input(f"URL {i+1}") for i in range(3)]
process_url_clicked = st.sidebar.button("Process URLs")

file_path = "faiss_store_hf.pkl"
main_placeholder = st.empty()

# ✅ Function to validate URLs
def is_url_accessible(url):
    try:
        response = requests.get(url, timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False

# ✅ Process URLs
if process_url_clicked:
    valid_urls = [url.strip() for url in urls if url.strip() and is_url_accessible(url)]

    if valid_urls:
        main_placeholder.text("Loading data... ✅✅✅")

        # ✅ Load data from URLs
        try:
            loader = WebBaseLoader(valid_urls)
            data = loader.load()
        except Exception as e:
            st.error(f"Error loading URLs: {e}")
            data = []

        if not data:
            st.warning("Failed to fetch data. Please check your URLs.")
        else:
            # ✅ Split data
            text_splitter = RecursiveCharacterTextSplitter(
                separators=['\n\n', '\n', '.', ','],
                chunk_size=1000
            )
            main_placeholder.text("Splitting text... ✅✅✅")
            docs = text_splitter.split_documents(data)

            if not docs:
                st.warning("No documents found after text splitting. Please check your URLs.")
            else:
                # ✅ Create embeddings and save them to FAISS index
                try:
                    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
                    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
                    vectorstore_hf = FAISS.from_documents(docs, embeddings)
                    main_placeholder.text("Building Embedding Vectors... ✅✅✅")
                    time.sleep(2)

                    # ✅ Save FAISS index
                    with open(file_path, "wb") as f:
                        pickle.dump(vectorstore_hf, f)
                except Exception as e:
                    st.error(f"Error creating embeddings: {e}")

    else:
        st.warning("No valid URLs found. Please enter correct URLs.")

# ✅ Handle Query
query = main_placeholder.text_input("Ask a question:")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            try:
                vectorstore = pickle.load(f)
                retriever = vectorstore.as_retriever()
                docs = retriever.get_relevant_documents(query)

                # ✅ Display Answer
                st.header("Relevant Information")
                for doc in docs:
                    st.write(doc.page_content)

            except Exception as e:
                st.error(f"Error loading FAISS index: {e}")
    else:
        st.warning("No FAISS index found. Please process URLs first.")
