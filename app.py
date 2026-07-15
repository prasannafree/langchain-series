from dotenv import load_dotenv
from google.cloud import firestore
from langchain_google_firestore import FirestoreChatMessageHistory
from langchain_ollama import ChatOllama

# Load environment configuration variables 
load_dotenv()

# Setup Firebase Firestore parameters
PROJECT_ID = "langchain-series"   # <-- Replace with your actual project ID string
SESSION_ID = "user_session_2"  
COLLECTION_NAME = "chat_history"

# Initialize Firestore Client connections
print("Initializing Firestore Client...")
client = firestore.Client(project=PROJECT_ID)

# Initialize persistent Firestore Chat Message History backend
print("Initializing Firestore Chat Message History...")
chat_history = FirestoreChatMessageHistory(
    session_id=SESSION_ID,
    collection=COLLECTION_NAME,
    client=client,
)
print("Chat History Initialized.")
print("Current Chat History:", chat_history.messages)

# Initialize your local Ollama Chat Model instance pipeline
model = ChatOllama(model="gemma4:latest", temperature=0)

print("\nStart chatting with the AI. Type 'exit' to quit.")

while True:
    human_input = input("\nUser: ")
    if human_input.lower() == "exit":
        break

    # Commit user input straight into Cloud storage logs
    chat_history.add_user_message(human_input)

    # Process prompt using complete chat records context
    ai_response = model.invoke(chat_history.messages)
    
    # Save the AI response text directly back into Firestore
    chat_history.add_ai_message(ai_response.content)

    print(f"AI: {ai_response.content}")
