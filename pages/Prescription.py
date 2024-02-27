import os
import time

import openai
import streamlit as st
from dotenv import load_dotenv
from openai.error import RateLimitError


load_dotenv()

st.title(":blue[AI Prescription]")
st.markdown("#")

# Instantiate session state variables
if "API_KEY" not in st.session_state:
    st.session_state.API_KEY = ""

if "RATE_EXCEEDED" not in st.session_state:
    st.session_state.RATE_EXCEEDED = False

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "system",
            "content": "You are an AI assistant specialized in assisting farmers with accurate prescriptions and possible chemicals \
            or treatments that can cure the disease for all types of plant diseases. Farmers can provide you with information\
            about the health of their crops and seek guidance on managing and treating plant diseases",
        },
    ]


# Load api key
if (st.session_state.API_KEY != "") and (st.session_state.RATE_EXCEEDED != True):
    openai.api_key = st.session_state.API_KEY

elif st.session_state.RATE_EXCEEDED:
    openai.api_key = os.getenv("OPENAI_API_KEY")

# sidebar
with st.sidebar:
    st.header("⚙️**Setup**")
    api_inp = st.text_input(
        "🔑OpenAI API Key",
        type="password",
        placeholder="Paste your OpenAI API key here (sk-...)",
        help="You can get your API key from https://platform.openai.com/account/api-keys.",
    )

    if api_inp and api_inp[:3] != "sk-":
        st.error("Invalid API Key!")
    elif api_inp[:3] == "sk-" and len(api_inp) == 51:
        st.session_state.API_KEY = api_inp


def get_ai_response(messages):
    """
    Function to call chat gpt api
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=messages,
        )
        return response["choices"][0]["message"]["content"]

    except RateLimitError:
        st.session_state.RATE_EXCEEDED = True
        st.warning("You've exhausted your API credit. Set a new one from the sidebar.")


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    if message["role"] == "system":
        continue
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Accept user input
if prompt := st.chat_input("Send a message..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        with st.status("Generating response..."):
            assistant_response = get_ai_response(st.session_state.messages)

            # Simulate stream of response with milliseconds delay
            if assistant_response is not None:
                for chunk in assistant_response.split():
                    full_response += chunk + " "
                    time.sleep(0.06)
                    # Add a blinking cursor to simulate typing
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
