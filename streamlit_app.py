import streamlit as st
from core_chatbot import ask_bot
from stt import get_voice_input
from tts import speak_response
st.title("🧠 Medical Chatbot")

if "chat" not in st.session_state:
    st.session_state.chat = []

user_input = st.text_input("Ask a medical question:")


if user_input:
    if any(k in user_input.lower() for k in ["symptom", "treatment", "pain", "medicine", "doctor", "disease"]):
        response = ask_bot(user_input)
    else:
        response = "⚠️ I only respond to medical-related questions."

    st.session_state.chat.append(("You", user_input))
    st.session_state.chat.append(("Bot", response))

if st.button(" Use Voice Input"):
    voice_input = get_voice_input()
    st.text_input("You said:", value=voice_input, disabled=True)

    if voice_input:
        if any(k in voice_input.lower() for k in ["symptom", "treatment", "pain", "medicine", "doctor", "disease"]):
            response = ask_bot(voice_input)

        else:
            response = "⚠️ I only respond to medical-related questions."

        st.session_state.chat.append(("You (voice)", voice_input))
        st.session_state.chat.append(("Bot", response))
        
   

for speaker, msg in reversed(st.session_state.chat):
    st.markdown(f"**{speaker}:** {msg}")

    