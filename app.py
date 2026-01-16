import streamlit as st
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, f1_score,
    mean_squared_error, r2_score
)
from xgboost import XGBClassifier, XGBRegressor
import plotly.express as px

st.title("🤖 Projeto AutoML Inteligente")

# =========================================
# 1. Upload e preparação dos dados
# =========================================
arquivo = st.file_uploader("Carregue seu dataset (CSV)", type=["csv"])
if arquivo is not None:
    df = pd.read_csv(arquivo)
    st.write("📊 Visualização inicial dos dados:", df.head())

    # =========================================
    # 2. Seleção da coluna alvo
    # =========================================
    alvo = st.selectbox("🎯 Selecione a coluna alvo", df.columns)
    y = df[alvo]
    X = df.drop(columns=[alvo])

    # =========================================
    # 3. Detecção do tipo de problema
    # =========================================
    if pd.api.types.is_numeric_dtype(y) and y.nunique() > 15:
        problema = "regressao"
        y = pd.to_numeric(y, errors="coerce").fillna(y.mean())
    else:
        problema = "classificacao"
        y = y.fillna(y.mode()[0])

    st.info(f"🔎 Detectado problema de **{problema.upper()}**")

    # =========================================
    # 4. Split dos dados
    # =========================================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # =========================================
    # 5. Definição dos modelos
    # =========================================
    if problema == "regressao":
        modelos = {
            "Regressão Linear": LinearRegression(),
            "Random Forest Regressor": RandomForestRegressor(),
            "XGBRegressor": XGBRegressor()
        }
    else:
        modelos = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest Classifier": RandomForestClassifier(),
            "XGBClassifier": XGBClassifier()
        }

    resultados, modelos_treinados = {}, {}
    st.subheader("🤖 Treinamento dos Modelos")

    # =========================================
    # 6. Loop de treinamento
    # =========================================
    for nome, modelo in modelos.items():
        inicio = time.time()

        if len(y_train) == 0 or len(X_train) == 0:
            st.error("⚠️ Conjunto de treino vazio. Use mais linhas ou desative o modo rápido.")
            continue

        try:
            modelo.fit(X_train, y_train)
            tempo = time.time() - inicio

            y_pred = modelo.predict(X_test) if len(y_test) > 0 else []
            metricas = {}

            if problema == "classificacao" and len(y_test) > 0:
                metricas["accuracy"] = accuracy_score(y_test, y_pred)
                metricas["f1"] = f1_score(y_test, y_pred, average="weighted")
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                metricas["f1_cv"] = cross_val_score(modelo, X, y, cv=cv, scoring="f1_weighted").mean()
            elif problema == "regressao" and len(y_test) > 0:
                metricas["R2"] = r2_score(y_test, y_pred)
                metricas["RMSE"] = np.sqrt(mean_squared_error(y_test, y_pred))
                cv = 5
                metricas["R2_cv"] = cross_val_score(modelo, X, y, cv=cv, scoring="r2").mean()

            metricas["tempo"] = tempo
            resultados[nome] = metricas
            modelos_treinados[nome] = modelo

            st.success(f"{nome} treinado em {tempo:.2f}s")
            st.write("📈 Métricas:", metricas)

        except ValueError as e:
            st.error(f"❌ Erro ao treinar {nome}: {e}")

    # =========================================
    # 7. Gráfico comparativo das métricas
    # =========================================
    if resultados:
        st.subheader("📊 Comparativo de Modelos")
        df_resultados = pd.DataFrame(resultados).T.reset_index().rename(columns={"index": "Modelo"})
        if problema == "regressao":
            fig = px.bar(df_resultados, x="Modelo", y="R2", title="Comparação de R² entre modelos")
        else:
            fig = px.bar(df_resultados, x="Modelo", y="f1", title="Comparação de F1 entre modelos")
        st.plotly_chart(fig, use_container_width=True)