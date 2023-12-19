import os
import streamlit as st
import time
from langchain.llms import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

# setting up enviroment variables 
load_dotenv()

st.title("News search tool")

st.sidebar.title("News article urls")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")

file_path = "faiss_index.pkl"

main_placeholder = st.empty()

llm = OpenAI(temperature=0.9, max_tokens=500)
embeddings = OpenAIEmbeddings()

if process_url_clicked:
    # load data
    loader = UnstructuredURLLoader(urls=urls)

    main_placeholder.text("Data loading...")
    data = loader.load()

    # split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n','\n','.',','],
        chunk_size=1000
    )
    main_placeholder.text("Text splitting...")
    docs = text_splitter.split_documents(data)
    # Create embeddings and save it to FAISS index
    vectorstore_openai = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Building Embedding Vector...")
    time.sleep(2)
    # save the FAISS index to a local folder
    vectorstore_openai.save_local("faiss_store", index_name="index")
    main_placeholder.text("Store FAISS Index Locally")

query = main_placeholder.text_input("Enter your Question")
if query:
    index_path = "faiss_store\index"
    if os.path.exists(f"{index_path}.pkl"):
        vectorstore = FAISS.load_local("faiss_store\\", embeddings)
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
        result = chain({"question":query}, return_only_outputs=True)
        st.header("Answer")
        st.write(result["answer"])

        source = result.get("sources", "")
        if source:
            st.subheader("Sources")
            sources = source.split("\n")
            for s in sources:
                st.write(s)
