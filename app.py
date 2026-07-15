from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage , HumanMessage , AIMessage


from langchain_ollama import ChatOllama

import streamlit as st



# Load environment variables from .env
load_dotenv()

ollama1model = ChatOllama(model="qwen3:8b", temperature=0,)    # local model 
ollama2model = ChatOllama(model="gemma4:latest", temperature=0,)  # local model


chat_history = []  # Initialize chat history

system_message = SystemMessage(content="You are a helpful AI assistant.")
chat_history.append(system_message) # Add system message to chat history

# Chat loop
while True:
    query = input("You: ")
    if query.lower() == "exit":
        break
    chat_history.append(HumanMessage(content=query)) # Add user message
    
    # Get AI response using history
    result = ollama2model.invoke(chat_history)
    response = result.content
    chat_history.append(AIMessage(content=response)) # Add AI message
    
    print(f"AI: {response}")
    
    

print("---- Message History ----")
print(chat_history)




