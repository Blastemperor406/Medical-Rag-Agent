import streamlit as st
from agent import initialize_agent, process_query
from langgraph import visualize_chain

def run_streamlit_app():
    st.set_page_config(page_title="Medical Information Assistant", layout="wide")
    st.title("Medical Information Assistant")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if prompt := st.chat_input("What would you like to know?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get bot response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                agent, tools = initialize_agent()
                response = process_query(agent, tools, prompt)


                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                #visualize_chain(chain_of_thought)
