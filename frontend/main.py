import streamlit as st
import requests
import time
import services

if "token" not in st.session_state:
    st.session_state["token"] = None


login_page = st.Page(page = services.login, title = "Connexion", icon = ":material/account_circle:")
register_page = st.Page(page = services.register , title = "Creer un compte", icon = ":material/person_add:")

chat_page = st.Page(page = "chat.py" , title = "Chat", icon = ":material/chat:")
dashboard_page = st.Page(page = "dashboard.py" , title = "Dashboard", icon = ":material/bar_chart_4_bars:")

if services.is_authenticated():

    pg = st.navigation({
        "Chat" : [chat_page]})
else:
    pg = st.navigation([login_page,register_page])
pg.run()