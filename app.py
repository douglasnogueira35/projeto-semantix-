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

    # Sidebar para escolher quantidade de linhas
    st.sidebar.header("⚙️ Configurações")
    max_linhas = len(df)
    qtd_linhas = st.sidebar.slider(
        "📏 Quantidade de linhas para usar",
        min_value=50,
        max_value=max_linhas,
        value=min(1000, max_linhas),
        step=50
    )
    df = df.head(qtd_linhas)
    st.sidebar.write(f"✅ Usando {qtd_linhas} linhas do dataset")

    # =========================================
    # 2. Seleção da coluna alvo
    # =========================================
    alvo = st.selectbox("🎯 Selecione a coluna alvo", df.columns)
    y = df[alvo]
    X = df.drop(columns=[alvo])

    # =========================================
    # 3. Pré-processamento dos dados
    # =========================================
    if "data_ref" in X.columns:
        X["data_ref"] = pd.to_datetime(X["data_ref"], errors="coerce").astype(int) / 10**9

    X = pd.get_dummies(X, drop_first=True)
    X = X.fillna(0)

    # =========================================
    # 4. Detecção do tipo de problema
    # =========================================
    if pd.api.types.is_numeric_dtype(y) and y.nunique() > 15:
        problema = "regressao"
        y = pd.to_numeric(y, errors="coerce").fillna(y.mean())
    else:
        problema = "classificacao"
        y = y.fillna(y.mode()[0])

    st.info(f"🔎 Detectado problema de **{problema.upper()}**")

    # =========================================
    # 5. Split dos dados
    # =========================================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # =========================================
    # 6. Definição dos modelos
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
    # 7. Loop de treinamento
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
    # 8. Abas para resultados e relatório final
    # =========================================
    aba_resultados, aba_relatorio = st.tabs(["📊 Resultados", "📑 Relatório Final"])

    with aba_resultados:
        if resultados:
            st.subheader("📊 Comparativo de Modelos")
            df_resultados = pd.DataFrame(resultados).T.reset_index().rename(columns={"index": "Modelo"})
            if problema == "regressao":
                fig = px.bar(df_resultados, x="Modelo", y="R2", title="Comparação de R² entre modelos")
            else:
                fig = px.bar(df_resultados, x="Modelo", y="f1", title="Comparação de F1 entre modelos")
            st.plotly_chart(fig, use_container_width=True)

            st.download_button(
                label="📥 Baixar métricas em CSV",
                data=df_resultados.to_csv(index=False).encode("utf-8"),
                file_name="metricas_modelos.csv",
                mime="text/csv"
            )

    with aba_relatorio:
        st.subheader("📑 Relatório Final e Insights de Negócio")

        if resultados:
            melhor_modelo = max(resultados.items(), key=lambda x: x[1].get("R2", x[1].get("f1", 0)))
            nome_modelo, metricas = melhor_modelo

            st.write(f"✅ O melhor modelo foi **{nome_modelo}** com desempenho:")
            st.write(metricas)

            # Insights de negócio
            if problema == "classificacao":
                st.info("🔎 Insights: O modelo de classificação pode ajudar a prever perfis de clientes, "
                        "identificar riscos de inadimplência ou segmentar públicos para campanhas.")
            else:
                st.info("🔎 Insights: O modelo de regressão pode apoiar previsões de vendas, "
                        "estimativas de receita futura ou análise de impacto de variáveis econômicas.")

            # Relatório textual consolidado
            relatorio = f"""
            Relatório Final:
            - Tipo de problema: {problema.upper()}
            - Melhor modelo: {nome_modelo}
            - Principais métricas: {metricas}
            - Potenciais aplicações de negócio: {('Previsão de vendas, análise financeira, planejamento estratégico'
                                                 if problema == 'regressao' else
                                                 'Segmentação de clientes, análise de risco, campanhas direcionadas')}
            """
            st.text(relatorio)