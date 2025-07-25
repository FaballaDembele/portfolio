import pandas as pd
import time
import streamlit as st

def load_data(file):
    return pd.read_csv(file, delimiter=",", encoding='utf-8')

def show_loading():
    """Displays a loading page and then disappears."""
    placeholder = st.empty()

    placeholder.markdown(
        """
        <style>
        .loader-container {
            display:flex;
            flex-direction:column;
            justify-content:center;
            align-items:center;
            height:80vh;
            font-family:'Segoe UI', sans-serif;
            color:#6C63FF;
        }
        .spinner {
            border:8px solid #f3f3f3;
            border-top:8px solid #6C63FF;
            border-radius:50%;
            width:60px;
            height:60px;
            animation:spin 1s linear infinite;
            margin-bottom:20px;
        }
        @keyframes spin {
            0% { transform:rotate(0deg); }
            100% { transform:rotate(360deg); }
        }
        </style>

        <div class="loader-container">
            <div class="spinner"></div>
            <h2>Chargement en cours…</h2>
            <p>Veuillez patienter quelques instants.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    time.sleep(2.5)  
    placeholder.empty()  