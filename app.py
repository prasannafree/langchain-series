from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


from langchain_ollama import ChatOllama

import streamlit as st

# Load .env
load_dotenv()

# ----------------------------
# Prompt Template
# ----------------------------

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer the user's question clearly."
        ),
        (
            "user",
            "Question: {question}"
        )
    ]
)

# ----------------------------
# Ollama Model
# ----------------------------

llm = ChatOllama(
    model="qwen3:8b",
    temperature=0
)
# ----------------------------
# Output Parser
# ----------------------------

output_parser = StrOutputParser()

# ----------------------------
# Chain
# ----------------------------

chain = prompt | llm | output_parser

# ----------------------------
# Streamlit UI
# ----------------------------

st.markdown("Powered by Qwen3-8B via Ollama")

input_text = st.text_input(
    "Ask a question"
)

if input_text:
    response = chain.invoke(
        {
            "question": input_text
        }
    )

    st.write(response)

print("prasanna")