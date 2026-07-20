import os
from datetime import datetime

from dotenv import load_dotenv

from google.cloud import firestore
from google.oauth2 import service_account

from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
)

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

from langchain_ollama import ChatOllama
from langchain_google_firestore import FirestoreChatMessageHistory


load_dotenv()



credentials = service_account.Credentials.from_service_account_file("firebase-key.json")
firestore_client = firestore.Client(
    credentials=credentials,
    project=credentials.project_id
)




current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
session_id = f"chat_session_{current_time}"

chat_history = FirestoreChatMessageHistory(
    session_id=session_id,
    collection="chat_history",
    client=firestore_client
)

messages = chat_history.messages


if len(messages) == 0:

    system_msg = SystemMessage(
        content="You are a AI Chatbot. You are helpful, creative, clever, and very friendly."
    )

    messages.append(system_msg)



model = ChatOllama(model="gemma4:latest",temperature=0)
output_parser = StrOutputParser()




chain = (RunnableLambda(lambda msgs: msgs) | model| output_parser)



print("=" * 60)
print(f"Session ID : {session_id}")
print("Type 'stop' to quit.")
print("=" * 60)

while True:

    user_query = input("\nYou : ").strip()

    if user_query.lower() == "stop":
        print("\nGoodbye!")
        break

    if not user_query:
        continue



    human_msg = HumanMessage( content=user_query)
    messages.append(human_msg)

    ai_response = chain.invoke(messages)
    print("\nAI :", ai_response)

    ai_msg = AIMessage(content=ai_response )
    messages.append(ai_msg)


    chat_history.add_user_message(user_query)
    chat_history.add_ai_message(ai_response)