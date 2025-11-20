from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
import os

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

api_key = os.environ.get('GROQ_API_KEY')
if not api_key:
    raise ValueError("❌ GROQ_API_KEY environment variable not set")

llm = ChatGroq(
    model="llama-3.1-8b-instant", 
    temperature=0.7
)

samsung_system_prompt = SystemMessage(content="""You are an official Samsung Product Expert assistant. Your role is to help users with:

PRODUCT KNOWLEDGE:
- Galaxy smartphones (S series, Z series, A series, Note series)
- Tablets, watches, buds, and other Samsung devices
- Specifications, features, and comparisons
- One UI features and Android updates

TROUBLESHOOTING HELP:
- Common issues (battery, performance, connectivity)
- Step-by-step fixes for software problems
- Settings optimization tips
- Error code explanations

SETUP & USAGE:
- Device setup guidance
- Feature tutorials (DeX, Secure Folder, Edge panels)
- Personalization tips
- Ecosystem integration (with other Samsung devices)

SUPPORT INFORMATION:
- Warranty details
- Service center locations
- Trade-in programs
- Current promotions

IMPORTANT GUIDELINES:
- Always be helpful and professional
- For hardware issues, recommend visiting service centers
- Don't provide repair instructions for hardware
- Escalate complex software issues to live support
- Use simple, clear steps for troubleshooting
- Mention specific menu paths: "Go to Settings > Battery > Power saving"
- Answer in BDT anything related to price or money

Start by welcoming users as a Samsung Expert and asking how you can help with their Samsung device.""")


def chat_node(state: ChatState):
    messages = state['messages']
    
    all_messages = [samsung_system_prompt] + messages
    
    response = llm.invoke(all_messages)
    return {'messages': [response]}


checkpointer = MemorySaver()
graph = StateGraph(ChatState)

graph.add_node('chat_node', chat_node)
graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

chatbot = graph.compile(checkpointer=checkpointer)