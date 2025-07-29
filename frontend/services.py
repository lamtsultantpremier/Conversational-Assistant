import streamlit as st
import requests
import time
import re
import configs

if "token" not in st.session_state:
    st.session_state["token"] = None

def is_authenticated():
    return st.session_state["token"] is not None

def login():
    
    st.markdown("##### Connexion")
    with st.form("my_form"):
        username = st.text_input(label = "Username")
        password = st.text_input(label = "Password" , type = "password")

        submit = st.form_submit_button("Se connecter")
        if submit:
            # verifier le username et le mot de pass

            login_url = "http://127.0.0.1:8000/auth/login"

            data = {
                "username" : username,
                "password" :password
            }
            try : 
                response = requests.post(login_url , data = data)

                if response.status_code == 200:
                    token_data = response.json()
                    token = token_data["access_token"]
                    st.session_state["token"] = token
                    with st.status("Veuillez patientientez un momment"):
                        time.sleep(2)
                    st.success(body = "Connexion reuissi" , icon = ":material/check:")

                    st.rerun()
                else :
                    st.error(body = "Identifiant invalid" , icon = ":material/error:")
            except Exception as e: 
                st.error(e)

    
def logout():
        st.session_state["token"] = None
        with st.status("Veuillez patientez un moment"):
             time.sleep(1)
        st.success(body = "Deconnexion réuissi" , icon = ":material/check:")
        st.rerun()

def register():
    st.markdown("""
     <div style = "text-align : center;">           
        <h4>Créer un Compte</h4>
    </div>
    """,
    unsafe_allow_html = True)
    with st.form("register_form"):
        user_firstname = st.text_input(label ="Entrer votre Nom" , placeholder = "Nom")
        user_lastname = st.text_input(label = "Entrer votre Prenom" , placeholder = "Prenom")
        user_email = st.text_input(label = "Entrer votre Email" , placeholder = "Email")
        user_password = st.text_input(label = "Entrer votre mot de pass", type = "password" , placeholder = "Mot de pass")
        submit = st.form_submit_button("Enregistrer")
        if submit : 
            if user_email:
                if not validate_email(user_email):
                    st.error("Entrer un email valid")
            payload = {"nom" : user_firstname , "prenom" :user_lastname , "email":user_email , "password" : user_password}
            user_created = requests.post(f"{configs.BACKEND_URL}/auth/register" , json = payload)
            if user_created.status_code == 409 :
                st.error("User with this email already exist")
            if user_created.status_code == 200 :
                pass

def validate_email(email : str)-> str:
    email_regex = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{3}$"
    return re.match(email_regex , email)
        
           

