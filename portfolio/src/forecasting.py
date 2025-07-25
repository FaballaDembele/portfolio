import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from statsmodels.tsa.api import VAR
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from sklearn.model_selection import train_test_split

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
    kpi1, kpi2 = st.columns(2)
    #last_vals = df_s[df_s['ANNEE']==last_year][targets].values[0]
    if preds_lstm is not None:
        next_vals = preds_lstm[preds_lstm['ANNEE']==next_year][targets].values[0]
    elif preds_var is not None:
        next_vals = preds_var[preds_var['ANNEE']==next_year][targets].values[0]

    with kpi1: st.metric("📅 Dernière année connue", f"{last_year}")
    with kpi2: st.metric("🔮 Prévision", f"{next_vals[0]:,.0f}")
    #with kpi3: st.metric("Δ %", f"{(next_vals[0]/last_vals[0]-1)*100:+.1f} %")

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