import streamlit as st
import plotly.express as px
from io import BytesIO
import numpy as np

def advanced_features():
   
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