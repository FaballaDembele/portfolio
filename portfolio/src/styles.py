# CSS styles for customizing the appearance of the Streamlit app
import streamlit as st
st.markdown(
    """
    <style>
    /* Main page */
    .main {
        background-color: #fafafa;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #111111;
    }

    /* Headers */
    h1, h2, h3 {
        color: #6C63FF;
        font-family: 'Segoe UI', sans-serif;
    }

    /* KPI cards */
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #e6e6e6;
        border-radius: 8px;
        padding: 5% 5% 5% 10%;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
    """,
    unsafe_allow_html=True,
)