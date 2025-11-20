import streamlit as st
from ai_assistance_backend import chatbot
from langchain_core.messages import HumanMessage

st.set_page_config(
    page_title="Samsung Expert Assistant",
    page_icon="📱",
    layout="wide"
)

st.markdown("""
<style>
    .stApp {
        <!-- background-color: #ffffff; -->
    }
    .user-message {
        width: fit-content;
        background-color: #252524;
        color: white;
        padding: 12px 16px;
        border-radius: 15px 15px 0px 15px;
        margin: 8px 0;
        max-width: 70%;
        margin-left: auto;
    }
    .samsung-message {
        background-color: #2346a6;
        color: #ffffff;
        padding: 12px 16px;
        border-radius: 15px 15px 15px 0px;
        margin: 8px 0;
        max-width: 70%;
        margin-right: auto;
        border-left: 4px solid #1428a0;
    }
    .samsung-header {
        background: linear-gradient(135deg, #1428a0, #00a0e9);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
        margin-top:-50px;
    }
</style>
""", unsafe_allow_html=True)

CONFIG = {'configurable': {'thread_id': 'samsung-thread-1'}}

st.markdown("""
<div class="samsung-header">
    <h1>🤖 Samsung Product Expert</h1>
    <p>Your 24/7 assistant for all Samsung devices</p>
</div>
""", unsafe_allow_html=True)

if 'message_history' not in st.session_state:
    st.session_state.message_history = []


st.subheader("🚀 Quick Help")
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("📱 Battery Issues", use_container_width=True):
        st.session_state.quick_question = "My Samsung phone battery is draining too fast. What can I do?"

with col2:
    if st.button("⚡ Performance", use_container_width=True):
        st.session_state.quick_question = "My Samsung phone is running slow. How can I optimize it?"

with col3:
    if st.button("📶 Connectivity", use_container_width=True):
        st.session_state.quick_question = "I'm having Wi-Fi or Bluetooth connection issues with my Samsung device."

with col4:
    if st.button("🆕 Setup Help", use_container_width=True):
        st.session_state.quick_question = "I just got a new Samsung phone. What are the first things I should set up?"


for message in st.session_state.message_history:
    if message['role'] == 'user':
        st.markdown(f'<div class="user-message">{message["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="samsung-message">{message["content"]}</div>', unsafe_allow_html=True)


placeholder = "Ask about your Samsung device... (e.g., 'How do I use DeX mode?', 'My S23 camera is blurry')"
if 'quick_question' in st.session_state:
    placeholder = st.session_state.quick_question

user_input = st.chat_input(placeholder)

if user_input:
    if 'quick_question' in st.session_state:
        del st.session_state.quick_question
    
    st.session_state.message_history.append({'role': 'user', 'content': user_input})
    st.markdown(f'<div class="user-message">{user_input}</div>', unsafe_allow_html=True)
    
    with st.spinner("Samsung Expert is thinking..."):
        try:
            response = chatbot.invoke({'messages': [HumanMessage(content=user_input)]}, config=CONFIG)
            samsung_response = response['messages'][-1].content
        except Exception as e:
            samsung_response = f"I apologize, but I'm experiencing technical difficulties. Please try again or contact Samsung support directly. Error: {str(e)}"
    
    st.session_state.message_history.append({'role': 'assistant', 'content': samsung_response})
    st.markdown(f'<div class="samsung-message">{samsung_response}</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("📞 Samsung Resources")
    st.markdown("""
    **Official Support:**
    - 📞 1-800-SAMSUNG
    - 🌐 samsung.com/support
    - 🏪 Find Service Centers
    
    **Quick Links:**
    - User Manuals
    - Software Updates
    - Warranty Check
    - Trade-In Program
    """)
    
    if st.button("🔄 Clear Conversation", use_container_width=True):
        st.session_state.message_history = []
        st.rerun()