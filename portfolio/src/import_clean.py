from streamlit import file_uploader, session_state, success, warning, checkbox, button, dataframe, title, markdown, slider, multiselect, expander, download_button
import pandas as pd
import streamlit as st
def load_data(file):
    return pd.read_csv(file, delimiter=",", encoding='utf-8')

def import_clean():
    title("📥 Importation et Nettoyage des Données")
    
    uploaded_file = file_uploader("Téléchargez votre fichier CSV", type="csv")
    
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        session_state.df = df
        
        success("✅ Fichier chargé avec succès!")
        st.subheader("👀 Aperçu des données")
        dataframe(df.head())
        
        st.subheader("📊 Statistiques descriptives")
        dataframe(df.describe(include='all'))
        
        st.subheader("🔧 Nettoyage des données")
        
        missing = df.isnull().sum()
        if missing.sum() > 0:
            warning(f"⚠️ Valeurs manquantes détectées: {missing.sum()}")
            if checkbox("Afficher les valeurs manquantes par colonne"):
                dataframe(missing[missing > 0])
            
            if button("Remplacer les valeurs manquantes par 0"):
                df = df.fillna(0)
                session_state.df = df
                success(f"✅ {missing.sum()} lignes remplacer par 0")
        
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            warning(f"⚠️ Doublons détectés: {duplicates}")
            if button("Supprimer les doublons"):
                df = df.drop_duplicates()
                session_state.df = df
                success(f"✅ {duplicates} doublons supprimés!")
        
        st.subheader("🔍 Filtrer les données")
        if 'ANNEE' in df.columns:
            years = slider(
                "Sélectionnez la plage d'années",
                min_value=int(df['ANNEE'].min()),
                max_value=int(df['ANNEE'].max()),
                value=(int(df['ANNEE'].min()), int(df['ANNEE'].max()))
            )
            df = df[(df['ANNEE'] >= years[0]) & (df['ANNEE'] <= years[1])]
            session_state.df = df
        
        st.subheader("🧼 Données nettoyées")
        dataframe(df.head(100))
        
        csv = df.to_csv(index=False).encode('utf-8')
        download_button(
            label="💾 Télécharger les données nettoyées (CSV)",
            data=csv,
            file_name='donnees_nettoyees.csv',
            mime='text/csv'
        )