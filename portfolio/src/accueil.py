from streamlit import title, markdown, image, dataframe,subheader
import pandas as pd


def accueil():
    title("📊 Plateforme d'Analyse Budgétaire Gouvernementale")
    
    markdown("""
    ## 🏠 Bienvenue dans l'outil d'analyse budgétaire
    Cette application permet d'analyser les données budgétaires gouvernementales avec :
    
    - **📥 Importation et nettoyage** des données financières
    - **🔍 Analyse exploratoire** interactive
    - **🔮 Prévisions** avec modèles ARIMA et LSTM
    - **🚀 Fonctionnalités avancées** d'analyse comparative
        
    ### 📋 Comment utiliser cette application :
    1. Importez votre fichier CSV dans l'onglet 'Importation'
    2. Nettoyez et préparez vos données
    3. Explorez les données dans l'onglet 'Analyse Exploratoire'
    4. Effectuez des prévisions dans l'onglet 'Prévision'
    5. Utilisez les outils avancés dans l'onglet 'Fonctionnalités Avancées'
    """)
    
    image("https://cdn.pixabay.com/photo/2017/08/01/00/38/man-2562325_1280.jpg", use_container_width=True)
    
    markdown("---")
    subheader("📄 Exemple de données")
    example_data = {
        'ANNEE': [2018, 2018, 2018],
        'CATEGORIE': ['CHARGES COMMUNES', 'CHARGES COMMUNES', 'INSTITUTION'],
        'LIBELLE SECTION': ['CHARGES COMMUNES', 'CHARGES COMMUNES', 'AGENCE NATIONALE DE LA SECURITE D\'ETAT'],
        'LIBELLE PROG': ['Dette', 'Provisions pour imprévus', 'Securite d\'Etat'],
        'ENGAGEMENT': [73.08, 0.0, 7.61],
        'INITIALE': [78.34, 0.5, 5.25],
        'LIQUIDATION': [71.3, 0.0, 7.61]
    }
    dataframe(pd.DataFrame(example_data))