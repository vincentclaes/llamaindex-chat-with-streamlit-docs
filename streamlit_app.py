import os
from pathlib import Path
import streamlit as st
import openai
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.retrievers import SVMRetriever


# def load_data():
#     # load the documents
#     loader = DirectoryLoader('./data', glob="**/*.pdf", show_progress=True, loader_cls=PyPDFLoader)
#     docs = loader.load()
#     # replace all new lines with spaces
#     [setattr(doc, "page_content", doc.page_content.replace("\n", " ")) for doc in docs]
#     print(docs)
#
#     # split the documents into chunks
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)
#     all_splits = text_splitter.split_documents(docs)
#
#     # construct vector store
#     vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())
#     # https://python.langchain.com/docs/use_cases/question_answering.html#go-deeper-3
#     svm_retriever = SVMRetriever.from_documents(all_splits, OpenAIEmbeddings())
#     return svm_retriever, vectorstore

# svm_retriever, vectorstore = load_data()

st.set_page_config(
    page_title="Chat with the Randstad Digital docs.",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)
openai.api_key = os.environ["OPENAI_API_KEY"]
st.title("Chat with the Randstad Digital docs.")


if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me a question about Randstad Digital docs.",
        }
    ]


@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(
        text="Loading and indexing the documents – hang tight! This should take 1-2 minutes."
    ):
        # index = data_embedding.main(data_dir=Path(__file__).resolve().parent / "data", query="what is Ausy?")
        # return index

        # load the documents
        loader = DirectoryLoader(
            "./data", glob="**/*.pdf", show_progress=True, loader_cls=PyPDFLoader
        )
        docs = loader.load()
        # replace all new lines with spaces
        [
            setattr(doc, "page_content", doc.page_content.replace("\n", " "))
            for doc in docs
        ]
        print(docs)

        # split the documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        all_splits = text_splitter.split_documents(docs)

        # construct vector store
        vectorstore = Chroma.from_documents(
            documents=all_splits, embedding=OpenAIEmbeddings()
        )
        # https://python.langchain.com/docs/use_cases/question_answering.html#go-deeper-3
        # svm_retriever = SVMRetriever.from_documents(all_splits, OpenAIEmbeddings())
        return vectorstore


# index = load_data()
vectorstore = load_data()

# if "chat_engine" not in st.session_state.keys():  # Initialize the chat engine
#     st.session_state.chat_engine = index.as_chat_engine(
#         chat_mode="condense_question", verbose=True
#     )

if prompt := st.chat_input(
    "Your question"
):  # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:  # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
            qa_chain = RetrievalQA.from_chain_type(
                llm,
                retriever=vectorstore.as_retriever(search_kwargs={"k": 6}),
                return_source_documents=True,
            )
            result = qa_chain({"query": prompt})

            output = f"""============RESULT==============
            \n
            {result["result"]}
            \n
            ============SOURCES=============
            """

            # Initialize an empty list to hold the lines
            lines = []

            source_docs = [
                (x.metadata["source"], x.page_content)
                for x in result["source_documents"]
            ]
            for i, doc in enumerate(source_docs):
                lines.append(f"* CHUNK: {i} *")
                lines.append(f"original doc: {doc[0]}")
                lines.append(f"{doc[1]}")
                lines.append("")  # for a newline between chunks

            # Join the lines with a newline character to get the multi-line string
            output += "\n".join(lines)
            # return output

            # response = st.session_state.chat_engine.chat(prompt)
            # st.write(response.response)
            # message = {"role": "assistant", "content": response.response}
            # st.session_state.messages.append(message)  # Add response to message history
            # lines = []
            # for i, doc in enumerate(response.source_nodes, start=1):
            #     lines.append(f"### Document Chunk: {i}")
            #     lines.append(f"__document name: {doc.metadata['file_name']} ({doc.node_id})__")
            #     lines.append(f"{doc.text}")
            #     lines.append('')  # for a newline between chunks
            # with st.expander("Source Document"):
            #     # Hack to get around st.markdown rendering LaTeX
            #     st.markdown('\n'.join(lines), unsafe_allow_html=True)
