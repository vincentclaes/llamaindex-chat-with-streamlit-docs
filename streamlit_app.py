import os
from pathlib import Path
import streamlit as st
import openai

import data_embedding

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
        index = data_embedding.main(data_dir=Path(__file__).resolve().parent / "data", query="what is Ausy?")
        return index


index = load_data()

if "chat_engine" not in st.session_state.keys():  # Initialize the chat engine
    st.session_state.chat_engine = index.as_chat_engine(
        chat_mode="condense_question", verbose=True
    )

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
            response = st.session_state.chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message)  # Add response to message history
            lines = []
            for i, doc in enumerate(response.source_nodes, start=1):
                lines.append(f"### Document Chunk: {i}")
                lines.append(f"__document name: {doc.metadata['file_name']} ({doc.node_id})__")
                lines.append(f"{doc.text}")
                lines.append('')  # for a newline between chunks
            with st.expander("Source Document"):
                # Hack to get around st.markdown rendering LaTeX
                st.markdown('\n'.join(lines), unsafe_allow_html=True)
