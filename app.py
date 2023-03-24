import streamlit as st
from multiapp import MultiApp
import demo


app = MultiApp()
st.set_page_config(page_title="AttnGAN", initial_sidebar_state="auto")

# Adding a custom logo
st.sidebar.image("logo.png", width=100)

# Adding a custom title
st.sidebar.title("AttnGAN Web Application")

# Adding a custom description
st.sidebar.markdown("A web application demonstrating the AttnGAN model")

# Adding a divider
st.sidebar.empty()

# Adding navigation options
st.sidebar.title("Navigation")

# Add all application here

app.add_app("Demo", demo.demo_gan)


# The main app
app.run()

