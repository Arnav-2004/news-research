import os
import pickle
import time

import streamlit as st
from dotenv import load_dotenv

from langchain.chains import RetrievalQAWithSourcesChain
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAI, OpenAIEmbeddings


load_dotenv()
os.getenv("OPENAI_API_KEY")

st.title("News Research Tool")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i + 1}")
    urls.append(url)

main_placeholder = st.empty()

process_url_clicked = st.sidebar.button("Process URLs")
if process_url_clicked:
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Started loading the data...")
    data = loader.load()

    if data:
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", '.', ','],
            chunk_size=1000
        )
        main_placeholder.text("Started splitting the text...")
        docs = text_splitter.split_documents(data)

        if docs:
            embeddings = OpenAIEmbeddings()
            vectorindex_openai = FAISS.from_documents(docs, embeddings)
            main_placeholder.text("Started building the embedding vector...")
            time.sleep(2)
            with open("vector_index.pkl", "wb") as f:
                pickle.dump(vectorindex_openai, f)
        else:
            main_placeholder.text("Text Splitter produced empty documents. Please check the data!")
    else:
        main_placeholder.text("Data loading failed. Please check the URLs or network connection!")

query = main_placeholder.text_input("Question:")
if query:
    if os.path.exists("vector_index.pkl"):
        with open("vector_index.pkl", "rb") as f:
            vector_index = pickle.load(f)
        chain = RetrievalQAWithSourcesChain.from_llm(llm=OpenAI(temperature=0.9, max_tokens=500), retriever=vector_index.as_retriever())
        result = chain({"question": query}, return_only_outputs=True)
        st.subheader("Answer:")
        st.write(result["answer"])

        sources = result.get("sources", '')
        if sources:
            st.subheader("Sources:")
            sources_list = sources.split("\n")
            for source in sources_list:
                st.write(source)
