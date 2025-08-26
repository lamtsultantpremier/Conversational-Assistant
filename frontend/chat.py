import streamlit as st
import numpy as np
import configs
import requests
import services
import time
st.set_page_config(layout = "wide")

#Header for authentication
headers = {
   "Authorization" : f"Bearer {st.session_state["token"]}"
}

#User Authentication
try:
   current_user = requests.get(f"{configs.BACKEND_URL}/auth/me" , headers = headers)
   user_connected = current_user.json()
except:
      st.error("Vous devriez vous reconnecter svp")
      with st.status(""):
         time.sleep(1)
      services.logout()

#CSS Style
st.markdown(
    """
    <style>
        section[data-testid="stSidebar"] {
            width: 300px !important;
            background-color: #008000 !important;
            border-right: 1px solid #e2e8f0 !important;
        }

        div[data-testid="stSidebarContent"] {
            width: 300px !important;
            padding: 1rem !important;
        }

        .stButton button {
            width: 100% !important;
            margin: 0.25rem 0 !important;
            padding: 0.75rem 1rem !important;
            border-radius: 0.5rem !important;
            text-align: left !important;
            transition: all 0.2s !important;
            background-color: white !important;
            border: 1px solid #e2e8f0 !important;
            color: #1e293b !important;
        }

        .stButton button:hover {
            background-color: #f1f5f9 !important;
            border-color: #cbd5e1 !important;
        }

        .stForm button{
            background-color: #3b82f6 !important;
            color: white !important;
            border: none !important;
            padding: 0.75rem 1rem !important;
            border-radius: 0.5rem !important;
            margin-bottom: 1rem !important;
            width: 100% !important;
            font-weight: 500 !important;
        }

        .stForm button:hover {
            background-color: #2563eb !important;
        }


        /* Conteneur principal des messages */
    .space-y-4 > div {
        margin-bottom: 2px;
    }
    
    /* Message de l'assistant */
    .assistant-message {
        margin-top : 10px;
        margin-bottom : 20px;
        background-color: #f3f4f6;
        color: #1f2937;
        border-radius: 1rem;
        border-bottom-left-radius: 0;
        padding: 0.75rem 1rem;
        max-width: 100%;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        margin-right: auto;
    }
    
    /* Message de l'utilisateur */
    .user-message {
        background-color: #3b82f6;
        color: white;
        border-radius: 1rem;
        border-bottom-right-radius: 0;
        padding: 0.75rem 1rem;
        max-width: 80%;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        margin-left: auto;
    }
    
    /* Conteneur des messages */
    .messages-container {
        padding-bottom: 5rem;
    }

    .centered-image {
            display: block;
            margin-left: auto;
            margin-right: auto;
            border-radius: 50%;
            width: 150px;
            height: 150px;
            object-fit: cover;
        }</style>""", unsafe_allow_html=True)


if "messages" not in st.session_state:
   st.session_state["messages"]=[]

if "session_id" not in st.session_state:
     session_id = requests.post(f"{configs.BACKEND_URL}/sessions" , headers = headers).json()
     st.session_state["session_id"] = session_id

if "waiting" not in st.session_state:
   st.session_state["waiting"]= False

with st.sidebar:
         with st.form("my_conversation_form" , border = False):
            if st.form_submit_button(label = "Nouvelle Conversation" , icon="💬"):
               session_id = requests.post(f"{configs.BACKEND_URL}/sessions" , headers = headers).json()
               st.session_state["session_id"] = session_id
               st.session_state.messages= []

         st.markdown("### _Historique des messages_")

         sessions  = []

         if "sessions" in user_connected.keys():
            for session in user_connected["sessions"] :
               sessions.append({"session_id" : session["session_id"] , "messages" : session["messages"]})
         else:
            with st.status(""):
               time.sleep(1)
               services.logout()
         

         for session in sessions:
             if session["messages"] != []:
               message_split = session["messages"][0]["content"][0:25]+"......."
               if st.button(label = message_split, key = session["session_id"] ,use_container_width = True):
                  st.session_state.messages = session["messages"]

with st.container():
   col2 , col1 = st.columns([0.8,0.2])
   with col1:
      deconnexion = st.button(label = "Deconnexion" , use_container_width = True)
      if deconnexion:
         services.logout()

             
with st.container():
   with st.container():
      col5 , col6,col7 = st.columns([0.4,0.4,0.3])
      with col6:
         st.image("images/chatbot.jpg", width=150, caption="Melissa")
      st.markdown("""
                  <div style="text-align:center;">
                     <h4> Melissa, le chatbot qui vous aide sur toutes vos questions relatives au Foncier en Côte d'Ivoire</h4>
                  <div>
                  """ , unsafe_allow_html = True)
      
      chat_placeholder = st.container()
      with chat_placeholder:
         st.markdown("""<div class ="messages-container">""", unsafe_allow_html = True)
         for msg in st.session_state["messages"]:
            if msg["role"] == "user":
               st.chat_message(name = "user" , avatar = ":material/account_circle:" ).markdown(msg["content"])
            
            if msg["role"] == "assistant":
               st.markdown(
                     f"""<div class="assistant-message">
                        {msg["content"]}
                      </div>""",unsafe_allow_html=True)
            
      #Afficage de l'input si uniquement on attend pas.
      if not st.session_state["waiting"]:
         prompt = st.chat_input(placeholder = "Comment puis-je vous aidez ? ")
      else:
         prompt = None

      if prompt:
         if "session_id" in st.session_state:
            with chat_placeholder:
               st.chat_message(name = "user" , avatar =  ":material/account_circle:").markdown(prompt)
                
            user_message = {"session_id" : st.session_state["session_id"] , "role" : "user" , "content" : prompt}
             
            with chat_placeholder:
               with st.status(label = "Melissa est en train de vous repondre......") as status:
                     chatbot_response = requests.post(f"{configs.BACKEND_URL}/messages" , json = user_message ,headers = headers).json()
                     status.update(label = "Reponse de Melissa ✅" , state = "complete")

               #On attend la reponse de Melissa
               st.session_state["waiting"] = True

               st.markdown(
                     f"""<div class="assistant-message">
                      {chatbot_response}
                       """, unsafe_allow_html=True)
            #Ajoute la reponse à l'historique
            st.session_state["messages"].append({"role" : "user" , "content" : prompt})
            st.session_state["messages"].append({"role" : "assistant" , "content" : chatbot_response})

            st.session_state["waiting"] = False

