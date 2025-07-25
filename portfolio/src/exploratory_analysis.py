import numpy as np
import streamlit as st
import plotly.express as px

def exploratory_analysis():
    st.title("🔍 Tableau de bord – Analyse exploratoire")

    if 'df' not in st.session_state:
        st.warning("⚠️ Importez d'abord vos données dans l’onglet « Importation ».")
        return

    df = st.session_state.df

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

    c1, c2 = st.columns(2)
    c3, c4 = st.columns(2)

    #Histogramme numérique
    with c1:
        var_num = st.selectbox("Variable numérique", num_cols, key="hist")
        fig = px.histogram(df, x=var_num, nbins=40, title=f"Distribution de {var_num}")
        st.plotly_chart(fig, use_container_width=True)

    #categorie
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    with c2:
        if cat_cols:
            var_cat = st.selectbox("Variable catégorielle", cat_cols, key="bar")
            top = df[var_cat].value_counts().nlargest(10).reset_index()
            top.columns = [var_cat, "count"]
            fig = px.bar(top, x=var_cat, y="count", title=f"Top 10 – {var_cat}")
            st.plotly_chart(fig, use_container_width=True)

    #Matrice de corrélation
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