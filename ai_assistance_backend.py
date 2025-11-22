from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langchain_core.tools import Tool
import os

def check_warranty(__arg1: str) -> str:  
    device_model = __arg1
    return f"📋 Warranty status for {device_model}: Active until December 2025. 1 year manufacturer warranty remaining."

def find_service_center(__arg1: str) -> str:
    location = __arg1
    return f"📍 Nearest Samsung service centers in {location}:\n• Samsung Plaza Downtown (2.3 km)\n• Authorized Service Partner Mall (4.1 km)\n• QuickFix Mobile Repair (1.8 km)"

def check_product_availability(__arg1: str) -> str:
    product_name = __arg1
    return f"📦 {product_name} availability:\n• Galaxy S24: In stock\n• Galaxy Buds3: Limited stock\n• Galaxy Watch6: Available"

tools = [
    Tool(
        name="check_warranty",
        func=check_warranty,
        description="Check Samsung device warranty status. Input: device model like 'Galaxy S23'"
    ),
    Tool(
        name="find_service_center", 
        func=find_service_center,
        description="Find nearest Samsung service centers. Input: location like 'Delhi' or 'Mumbai'"
    ),
    Tool(
        name="check_product_availability",
        func=check_product_availability,
        description="Check availability of Samsung products. Input: product name like 'Galaxy S24'"
    )
]

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

api_key = os.environ.get('GROQ_API_KEY')
if not api_key:
    raise ValueError("GROQ_API_KEY environment variable not set")

llm = ChatGroq(
    model="llama-3.1-8b-instant", 
    temperature=0.7
)

llm_with_tools = llm.bind_tools(tools)

samsung_system_prompt = SystemMessage(content="""You are an official Samsung Product Expert assistant. Your role is to help users with:

CONVERSATION FLOW:
- If user says greetings (hello, hi, hey): Respond warmly but briefly and ask how you can help
- If user says thanks/thank you: Acknowledge politely with "You're welcome!" or "Happy to help!" 
- If user says goodbye (bye, see you): Say goodbye politely
- If your reply has a question at the end, and the user replies to that question, answer accordingly
- Otherwise: Provide Samsung-specific help as outlined above

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
- Use emojis where neccessary

Start by welcoming users as a Samsung Expert and asking how you can help with their Samsung device.""")


def chat_node(state: ChatState):
    messages = state['messages']
    all_messages = [samsung_system_prompt] + messages
    response = llm_with_tools.invoke(all_messages) 
    return {'messages': [response]}

def tool_node(state: ChatState):
    last_message = state['messages'][-1]
    
    results = []
    for tool_call in last_message.tool_calls:
        tool_function = next((tool for tool in tools if tool.name == tool_call['name']), None)
        if tool_function:
            try:
                result = tool_function.func(**tool_call['args'])
                results.append(f"Tool Result: {result}")
            except Exception as e:
                results.append(f"ool error: {str(e)}")
    
    return {'messages': [HumanMessage(content="\n".join(results))]}

def route_tools(state: ChatState):
    last_message = state['messages'][-1]
    
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "use_tools"
    
    return "continue_chat"


checkpointer = MemorySaver()
graph = StateGraph(ChatState)

graph.add_node('chat_node', chat_node)
graph.add_node('tool_node', tool_node)

graph.add_edge(START, 'chat_node')
graph.add_conditional_edges(
    'chat_node',
    route_tools,
    {
        "use_tools": "tool_node",  
        "continue_chat": END        
    }
)
graph.add_edge('tool_node', 'chat_node') 

chatbot = graph.compile(checkpointer=checkpointer)