
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from statsmodels.tsa.api import VAR
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from io import BytesIO
import warnings
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go

# Désactiver les warnings
warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="Analyse Budgétaire Gouvernementale",
    page_icon="📊",
    layout="wide"
)

# Fonction pour charger les données
@st.cache_data
def load_data(file):
    return pd.read_csv(file, delimiter=",", encoding='utf-8')

# Page d'accueil
def accueil():
    st.title("📊 Plateforme d'Analyse Budgétaire Gouvernementale")
    
    st.markdown("""
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
    
    st.image("https://cdn.pixabay.com/photo/2017/08/01/00/38/man-2562325_1280.jpg", use_container_width=True)
    
    st.markdown("---")
    st.subheader("📄 Exemple de données")
    example_data = {
        'ANNEE': [2018, 2018, 2018],
        'CATEGORIE': ['CHARGES COMMUNES', 'CHARGES COMMUNES', 'INSTITUTION'],
        'LIBELLE SECTION': ['CHARGES COMMUNES', 'CHARGES COMMUNES', 'AGENCE NATIONALE DE LA SECURITE D\'ETAT'],
        'LIBELLE PROG': ['Dette', 'Provisions pour imprévus', 'Securite d\'Etat'],
        'ENGAGEMENT': [73.08, 0.0, 7.61],
        'INITIALE': [78.34, 0.5, 5.25],
        'LIQUIDATION': [71.3, 0.0, 7.61]
    }
    st.dataframe(pd.DataFrame(example_data))
import time

def show_loading():
    """Affiche une page de chargement puis disparaît."""
    placeholder = st.empty()

    # HTML / CSS centré
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

    # Durée simulée (ou remplacer par un vrai traitement)
    time.sleep(2.5)  # <-- ajustez ou retirez
    placeholder.empty()   # fait disparaître le bloc
# Page d'importation et nettoyage
def import_clean():
    st.title("📥 Importation et Nettoyage des Données")
    
    uploaded_file = st.file_uploader("Téléchargez votre fichier CSV", type="csv")
    
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        st.session_state.df = df
        
        st.success("✅ Fichier chargé avec succès!")
        st.subheader("👀 Aperçu des données")
        st.dataframe(df.head())
        
        st.subheader("📊 Statistiques descriptives")
        st.dataframe(df.describe(include='all'))
        
        st.subheader("🔧 Nettoyage des données")
        
        # Vérification des valeurs manquantes
        missing = df.isnull().sum()
        if missing.sum() > 0:
            st.warning(f"⚠️ Valeurs manquantes détectées: {missing.sum()}")
            if st.checkbox("Afficher les valeurs manquantes par colonne"):
                st.dataframe(missing[missing > 0])
            
            if st.button("Remplacer les valeurs manquantes par 0"):
                df = df.fillna(0)
                st.session_state.df = df
                st.success(f"✅ {missing.sum()} lignes remplacer par 0")
        
        # Suppression des doublons
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            st.warning(f"⚠️ Doublons détectés: {duplicates}")
            if st.button("Supprimer les doublons"):
                df = df.drop_duplicates()
                st.session_state.df = df
                st.success(f"✅ {duplicates} doublons supprimés!")
        
        # Filtrage des données
        st.subheader("🔍 Filtrer les données")
        if 'ANNEE' in df.columns:
            years = st.slider(
                "Sélectionnez la plage d'années",
                min_value=int(df['ANNEE'].min()),
                max_value=int(df['ANNEE'].max()),
                value=(int(df['ANNEE'].min()), int(df['ANNEE'].max()))
            )
            df = df[(df['ANNEE'] >= years[0]) & (df['ANNEE'] <= years[1])]
            st.session_state.df = df
        
        st.subheader("🧼 Données nettoyées")
        st.dataframe(df.head(100))
        
        # Téléchargement des données nettoyées
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="💾 Télécharger les données nettoyées (CSV)",
            data=csv,
            file_name='donnees_nettoyees.csv',
            mime='text/csv'
        )

# Page d'analyse exploratoire
def exploratory_analysis():
    st.title("🔍 Tableau de bord – Analyse exploratoire")

    if 'df' not in st.session_state:
        st.warning("⚠️ Importez d'abord vos données dans l’onglet « Importation ».")
        return

    df = st.session_state.df

    # ---------- KPI ROW ----------
    num_cols = df.select_dtypes(include=np.number).columns.tolist()

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("📊 Lignes", f"{len(df):,}")
    with kpi2:
        st.metric("🗂️ Colonnes", f"{len(df.columns)}")
    with kpi3:
        total_eng = df['ENGAGEMENT'].sum() if 'ENGAGEMENT' in num_cols else 0
        st.metric("💰 Engagement", f"{total_eng:,.0f}")
    with kpi4:
        if 'INITIALE' in num_cols and 'ENGAGEMENT' in num_cols:
            taux = (df['ENGAGEMENT'].sum() / df['INITIALE'].sum()) * 100
            st.metric("✅ Taux d’exec.", f"{taux:.1f} %")
        else:
            st.metric("✅ Taux d’exec.", "—")

    st.markdown("---")

    # ---------- 2×2 GRID ----------
    c1, c2 = st.columns(2)
    c3, c4 = st.columns(2)

    # 1. Histogramme numérique
    with c1:
        var_num = st.selectbox("Variable numérique", num_cols, key="hist")
        fig = px.histogram(df, x=var_num, nbins=40, title=f"Distribution de {var_num}")
        st.plotly_chart(fig, use_container_width=True)

    # 2. Top 10 catégoriel
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    with c2:
        if cat_cols:
            var_cat = st.selectbox("Variable catégorielle", cat_cols, key="bar")
            top = df[var_cat].value_counts().nlargest(10).reset_index()
            top.columns = [var_cat, "count"]
            fig = px.bar(top, x=var_cat, y="count", title=f"Top 10 – {var_cat}")
            st.plotly_chart(fig, use_container_width=True)

    # 3. Matrice de corrélation
    with c3:
        if len(num_cols) > 1:
            corr = df[num_cols].corr()
            fig = px.imshow(corr, text_auto=".2f", aspect="auto",
                            title="Matrice de corrélation")
            st.plotly_chart(fig, use_container_width=True)

    # 4. Série temporelle
    with c4:
        if 'ANNEE' in df.columns:
            ts_var = st.selectbox("Variable temporelle", num_cols, key="ts")
            ts_df = df.groupby('ANNEE', as_index=False)[ts_var].sum()
            fig = px.line(ts_df, x='ANNEE', y=ts_var,
                          markers=True, title=f"Évolution de {ts_var}")
            st.plotly_chart(fig, use_container_width=True)

    if 'CATEGORIE' in df.columns and 'ANNEE' in df.columns:

        st.subheader("📊 Analyse filtrée par Année et Catégorie")
    # Sélection du type de budget (colonne numérique)
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    budget_type = st.selectbox("🔢 Choisir le type de budget", numeric_cols, index=0)

    # Filtres : Années et Catégories
    selected_years = st.multiselect(
        "📅 Choisissez les années", 
        sorted(df['ANNEE'].unique()), 
        default=sorted(df['ANNEE'].unique())
    )

    selected_categories = st.multiselect(
        "🏷️ Choisissez les catégories", 
        sorted(df['CATEGORIE'].unique()), 
        default=sorted(df['CATEGORIE'].unique())
    )

    # Application des filtres
    filtered_df = df[(df['ANNEE'].isin(selected_years)) & (df['CATEGORIE'].isin(selected_categories))]

    if filtered_df.empty:
        st.warning("⚠️ Aucune donnée pour les filtres sélectionnés.")
    else:
        col1, col2 = st.columns(2)

        #  Col1: Barres groupées (ANNEE + CATEGORIE) 
        with col1:
            grouped = filtered_df.groupby(['ANNEE', 'CATEGORIE'],as_index=False)[budget_type].sum().reset_index()
            fig1 = px.bar(grouped, 
                          x='ANNEE', y=budget_type, color='CATEGORIE', 
                          barmode='group',
                          title=f"Évolution de {budget_type} par Année et Catégorie")
            st.plotly_chart(fig1)

        #  Col2: Pie chart par catégorie 
        with col2:
            category_sum = filtered_df.groupby('CATEGORIE')[budget_type].sum().reset_index()
            fig2 = px.pie(category_sum, names='CATEGORIE', values=budget_type,
                          title=f"Répartition de {budget_type} par Catégorie")
            st.plotly_chart(fig2)


# Page de prévision
def forecasting():
    st.title("🔮 Prévisions Budgétaires – Dashboard")

    if 'df' not in st.session_state:
        st.warning("⚠️ Importez d'abord vos données dans l’onglet « Importation ».")
        return

    df = st.session_state.df
    if 'ANNEE' not in df.columns:
        st.error("❌ Colonne 'ANNEE' manquante."); return

    # ---- 1. Préparation ----
    num_vars= df.select_dtypes(include=np.number).columns.tolist()
    targets= st.multiselect("Sélectionner la/les variable(s) à prédire",
                                num_vars, default=['ENGAGEMENT'])
        
    df_s= df.groupby('ANNEE')[targets].sum().sort_index().reset_index()
    last_year  = int(df_s['ANNEE'].max())
    next_year  = st.slider("Année cible", last_year+1, last_year+10,last_year+1)

    # ---- 2. Paramètres modèles ----
    with st.expander("⚙️ Paramètres des modèles"):
        c1, c2, c3 = st.columns(3)
        with c1: look_back = st.slider("Look-back", 1, 10, 3)
        with c2: epochs   = st.slider("Epochs LSTM", 50, 500, 200)
        with c3: batch_sz = st.number_input("Batch-size", 1, 128, 4)

    # ---- 3. RUN LSTM ----
    run_lstm, run_var = st.columns(2)
    preds_lstm, preds_var = None, None
    with run_lstm:
        if st.button("🚀 Lancer LSTM", use_container_width=True):
            preds_lstm = _lstm_forecast(df_s, targets, look_back, epochs, batch_sz,
                                        steps=next_year-last_year)
    with run_var:
        if st.button("🔁 Lancer VAR", use_container_width=True):
            preds_var = _var_forecast(df_s, targets, look_back,
                                      steps=next_year-last_year)

    # ---- 4. Affichage ----
    if preds_lstm is None and preds_var is None:
        st.info("Cliquez sur un modèle pour générer les prévisions.")
        return

    # 4-a KPI Cards
    kpi1, kpi2, kpi3 = st.columns(3)
    last_vals = df_s[df_s['ANNEE']==last_year][targets].values[0]
    if preds_lstm is not None:
        next_vals = preds_lstm[preds_lstm['ANNEE']==next_year][targets].values[0]
    elif preds_var is not None:
        next_vals = preds_var[preds_var['ANNEE']==next_year][targets].values[0]

    with kpi1: st.metric("📅 Dernière année connue", f"{last_year}")
    with kpi2: st.metric("🔮 Prévision", f"{next_vals[0]:,.0f}")
    with kpi3: st.metric("Δ %", f"{(next_vals[0]/last_vals[0]-1)*100:+.1f} %")

    # 4-b Graphique
    fig = go.Figure()
    # Courbe historique
    for t in targets:
        fig.add_trace(go.Scatter(
            x=df_s['ANNEE'], y=df_s[t],
            mode='lines+markers', name=f"{t} (Historique)"))
    # LSTM
    if preds_lstm is not None:
        for t in targets:
            fig.add_trace(go.Scatter(
                x=preds_lstm['ANNEE'], y=preds_lstm[t],
                mode='lines+markers', name=f"{t} (LSTM)",
                line=dict(dash='dot')))
    # VAR
    if preds_var is not None:
        for t in targets:
            fig.add_trace(go.Scatter(
                x=preds_var['ANNEE'], y=preds_var[t],
                mode='lines+markers', name=f"{t} (VAR)",
                line=dict(dash='dash')))

    fig.update_layout(height=500,
                      xaxis_title="Année",
                      yaxis_title="Valeur",
                      hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    # 4-c Tableau détaillé
    st.subheader("📄 Données complètes")
    if preds_lstm is not None:
        st.dataframe(pd.concat([df_s, preds_lstm]).reset_index(drop=True).style.format("{:,.0f}"))
    elif preds_var is not None:
        st.dataframe(pd.concat([df_s, preds_var]).reset_index(drop=True).style.format("{:,.0f}"))

def _lstm_forecast(df_s, targets, look_back, epochs, batch_size, steps):
    """Return a DataFrame with future forecasts (LSTM)."""
    data = df_s[targets].values
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    def make_seq(data, lb):
        X, y = [], []
        for i in range(lb, len(data)):
            X.append(data[i-lb:i])
            y.append(data[i])
        return np.array(X), np.array(y)

    X, y = make_seq(data_scaled, look_back)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train.shape[1], X_train.shape[2])),
        tf.keras.layers.LSTM(50, return_sequences=True),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.LSTM(50,return_sequences=False),
        tf.keras.layers.Dense(y_train.shape[1])
    ])
    model.compile(optimizer='adam', loss='mse')
    lstm_model=model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
              validation_data=(X_test, y_test), verbose=0,callbacks=[tf.keras.callbacks.EarlyStopping(
                  monitor='val_loss',patience=5,restore_best_weights=True)])
    with st.expander("📉 Loss Curves (LSTM)", expanded=False):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=lstm_model.history['loss'],
            mode='lines',
            name='Train Loss',
            line=dict(color='#636EFA')))
        fig.add_trace(go.Scatter(
            y=lstm_model.history['val_loss'],
            mode='lines',
            name='Validation Loss',
            line=dict(color='#EF553B')))
        fig.update_layout(
            title='Évolution de la perte',
            xaxis_title='Épochs',
            yaxis_title='MSE',
            height=300)
        st.plotly_chart(fig, use_container_width=True)
    # Forecast loop
    current = data_scaled[-look_back:]
    preds = []
    for _ in range(steps):
        p = model.predict(current.reshape(1, look_back, len(targets)))[0]
        preds.append(p)
        current = np.vstack([current[1:], p])
    preds = scaler.inverse_transform(np.array(preds))
    future_years = list(range(df_s['ANNEE'].max()+1, df_s['ANNEE'].max()+steps+1))
    return pd.DataFrame(preds, columns=targets).assign(ANNEE=future_years)

def _var_forecast(df_s, targets, look_back, steps):
    work = df_s[targets].copy()

    # ✅ Supprimer les colonnes constantes
    non_constant_cols = work.columns[work.nunique() > 1]
    work = work[non_constant_cols]

    if work.empty:
        raise ValueError("Toutes les variables sont constantes. Aucune donnée exploitable.")

    model = VAR(work)
    res = model.fit(maxlags=look_back)
    forecast = res.forecast(work.values[-look_back:], steps=steps)
    out_cols = work.columns.tolist()

    future_years = list(range(df_s['ANNEE'].max() + 1,
                              df_s['ANNEE'].max() + steps + 1))
    return pd.DataFrame(forecast, columns=out_cols).assign(ANNEE=future_years)

st.markdown(
    """
    <style>
    div[data-testid="metric-container"] {
        background-color:#ffffff;
        border:1px solid #e6e6e6;
        border-radius:8px;
        padding:5% 5% 5% 10%;
        box-shadow:0 2px 4px rgba(0,0,0,0.05);
    }
    h1, h2, h3 { color:#6C63FF; }
    </style>
    """,
    unsafe_allow_html=True,
)

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
# Page des fonctionnalités avancées
def advanced_features():
    st.markdown(
        """
        <script>
            window.scrollTo(0, 0);
        </script>
        """,
        unsafe_allow_html=True
    )
    st.title("🚀 Fonctionnalités Avancées")
    
    if 'df' not in st.session_state:
        st.warning("⚠️ Veuillez d'abord importer des données dans l'onglet 'Importation'")
        return
    
    df = st.session_state.df
    
    st.subheader("🏛️ Analyse Comparative entre Institutions")
    
    selected_institutions = st.multiselect(
        "Sélectionnez les institutions à comparer", 
        df['LIBELLE SECTION'].unique(),
        default=df['LIBELLE SECTION'].unique()[:2] if 'LIBELLE SECTION' in df.columns else []
    )
    
    if selected_institutions and 'LIBELLE SECTION' in df.columns:
        inst_comparison = df[df['LIBELLE SECTION'].isin(selected_institutions)]
        
        # KPI pour chaque institution
        kpi_cols = st.columns(len(selected_institutions))
        for i, institution in enumerate(selected_institutions):
            inst_data = inst_comparison[inst_comparison['LIBELLE SECTION'] == institution]
            total_engagement = inst_data['ENGAGEMENT'].sum() if 'ENGAGEMENT' in df.columns else 0
            avg_liquidation = inst_data['LIQUIDATION'].mean() * 100 if 'LIQUIDATION' in df.columns else 0
            
            with kpi_cols[i]:
                st.metric(f"📊 Engagement Total - {institution}", f"{total_engagement:,.2f} milliards FCFA")
                st.metric(f"💧 Taux de Liquidation Moyen", f"{avg_liquidation:.1f}%")
        
        # Graphique comparatif
        if 'ENGAGEMENT' in df.columns:
            inst_summary = inst_comparison.groupby('LIBELLE SECTION')['ENGAGEMENT'].sum().reset_index()
            fig = px.bar(
                inst_summary,
                x='LIBELLE SECTION',
                y='ENGAGEMENT',
                color='LIBELLE SECTION',
                title="Comparaison des Engagements par Institution"
            )
            st.plotly_chart(fig)
    
    st.subheader("📉 Analyse des Écarts Budgétaires")
    
    if 'INITIALE' in df.columns and 'ENGAGEMENT' in df.columns:
        df['ECART'] = df['INITIALE'] - df['ENGAGEMENT']
        df['ECART_PCT'] = (df['ECART'] / df['INITIALE']) * 100
        
        threshold = st.slider(
            "Seuil d'écart significatif (%)", 
            min_value=0, 
            max_value=100, 
            value=10
        )
        
        significant_diff = df[abs(df['ECART_PCT']) > threshold]
        st.write(f"**🔍 {len(significant_diff)} lignes avec écart > {threshold}%**")
        
        # Visualisation des plus grands écarts
        if not significant_diff.empty:
            top_gaps = significant_diff.nlargest(10, 'ECART_PCT')
            fig = px.bar(
                top_gaps,
                x='LIBELLE SECTION' if 'LIBELLE SECTION' in df.columns else 'CATEGORIE',
                y='ECART_PCT',
                color='LIBELLE PROG' if 'LIBELLE PROG' in df.columns else None,
                title="Top 10 des Écarts Budgétaires (%)",
                labels={'ECART_PCT': 'Écart (%)'}
            )
            st.plotly_chart(fig)
    
    st.subheader("📅 Analyse Temporelle Avancée")
    
    if 'ANNEE' in df.columns:
        year_range = st.slider(
            "Sélectionnez la plage d'années",
            min_value=int(df['ANNEE'].min()),
            max_value=int(df['ANNEE'].max()),
            value=(int(df['ANNEE'].min()), int(df['ANNEE'].max()))
        )
        
        filtered_df = df[(df['ANNEE'] >= year_range[0]) & (df['ANNEE'] <= year_range[1])]
        
        # Sélection de la métrique parmi les colonnes numériques
        num_metrics = filtered_df.select_dtypes(include=np.number).columns.tolist()
        if num_metrics:
            metric = st.selectbox("Sélectionnez la métrique", num_metrics)
            
            # Agrégation par année
            yearly_data = filtered_df.groupby('ANNEE', as_index=False)[metric].sum()
            
            # Tendance
            fig = px.line(
                yearly_data,
                x='ANNEE',
                y=metric,
                title=f"📈 Évolution de {metric} sur la période sélectionnée",
                markers=True
            )
            st.plotly_chart(fig)
            
            # Comparaison par catégorie
            if 'CATEGORIE' in df.columns:
                category_data = filtered_df.groupby(['ANNEE', 'CATEGORIE'], as_index=False)[metric].sum()
                fig = px.area(
                    category_data,
                    x='ANNEE',
                    y=metric,
                    color='CATEGORIE',
                    title=f"🏷️ Répartition de {metric} par Catégorie"
                )
                st.plotly_chart(fig)
    
    st.subheader("💾 Export des Données")
    
    export_format = st.radio("Format d'export", ['CSV', 'Excel'])
    
    if st.button("Exporter les données"):
        if export_format == 'CSV':
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="💾 Télécharger CSV",
                data=csv,
                file_name='analyse_budgetaire.csv',
                mime='text/csv'
            )
        else:
            excel_buffer = BytesIO()
            df.to_excel(excel_buffer, index=False)
            st.download_button(
                label="💾 Télécharger Excel",
                data=excel_buffer.getvalue(),
                file_name='analyse_budgetaire.xlsx',
                mime='application/vnd.ms-excel'
            )

# Navigation avec icônes
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
    show_loading()
    pages[selection]()
    

if __name__ == "__main__":
    main()

