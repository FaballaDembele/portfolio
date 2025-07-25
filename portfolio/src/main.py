from streamlit_option_menu import option_menu
import streamlit as st
from accueil import accueil
from import_clean import import_clean
from exploratory_analysis import exploratory_analysis
from forecasting import forecasting
from advanced_features import advanced_features
from utils import show_loading

def main():
    with st.sidebar:
        selection = option_menu(
            menu_title="Navigation",
            options=["Accueil", "Importation", "Analyse Exploratoire", "Prévision", "Fonctionnalités Avancées"],
            icons=["house", "cloud-upload", "search", "graph-up", "rocket"],
            default_index=0,
            styles={
                "container": {"padding": "5px", "background-color": "#1E1E1E"},
                "icon": {"color": "white", "font-size": "20px"},
                "nav-link": {"color": "white", "font-size": "16px", "text-align": "left"},
                "nav-link-selected": {"background-color": "#6C63FF", "color": "white", "font-size": "16px", "text-align": "left"},
            }
        )
    pages = {
        "Accueil": accueil,
        "Importation": import_clean,
        "Analyse Exploratoire": exploratory_analysis,
        "Prévision": forecasting,
        "Fonctionnalités Avancées": advanced_features
    }
    pages[selection]()
    show_loading()

if __name__ == "__main__":
    main()