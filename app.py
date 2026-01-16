import streamlit as st
import pandas as pd
import numpy as np
import time
import sqlite3
from math import sqrt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score
)

from xgboost import XGBClassifier, XGBRegressor
import plotly.express as px

# Configuração inicial
st.set_page_config(page_title="AutoML Inteligente", layout="wide")
st.title("🚀 AutoML Inteligente – Análise Automática de Dados")
# Função para normalizar nome do arquivo
def normalizar_nome_arquivo(file):
    nome = file.name  # corrigido: pega o nome do arquivo enviado
    nome_base, extensao = os.path.splitext(nome)
    return nome_base.lower(), extensao.lower()

# Função para carregar dados
@st.cache_data
def carregar_dados(file):
    nome_base, extensao = normalizar_nome_arquivo(file)

    if extensao == ".csv":
        df = pd.read_csv(file)
    elif extensao == ".pkl":
        df = pd.read_pickle(file)
    elif extensao == ".ftr":
        df = pd.read_feather(file)
    else:
        st.error("Formato de arquivo não suportado. Use CSV, PKL ou FTR.")
        return None

    return df


def calcular_metricas(y_real, y_previsto, problema="classificacao"):
    if problema == "classificacao":
        n_classes = len(np.unique(y_real))
        media = "binary" if n_classes == 2 else "weighted"
        return {
            "acuracia": accuracy_score(y_real, y_previsto),
            "precisao": precision_score(y_real, y_previsto, average=media, zero_division=0),
            "recall": recall_score(y_real, y_previsto, average=media, zero_division=0),
            "f1": f1_score(y_real, y_previsto, average=media, zero_division=0),
        }
    else:
        mse = mean_squared_error(y_real, y_previsto)
        return {
            "MSE": mse,
            "RMSE": sqrt(mse),
            "MAE": mean_absolute_error(y_real, y_previsto),
            "R2": r2_score(y_real, y_previsto)
        }

# Sidebar
st.sidebar.header("⚙️ Configurações")
modo_ultra_rapido = st.sidebar.checkbox("⚡ Modo ULTRA-RÁPIDO (100 linhas)", value=True)
amostra = st.sidebar.slider("Quantidade de linhas", 50, 1000, 100)

# Upload
arquivo = st.file_uploader("📂 Envie seu arquivo CSV", type=["csv"])

if arquivo:
    df = pd.read_csv(arquivo)

    if modo_ultra_rapido and len(df) > amostra:
        df = df.sample(amostra, random_state=42)
        st.warning(f"⚡ Dataset reduzido para {amostra} linhas (modo rápido)")

    st.subheader("🔍 Visualização dos Dados")
    st.dataframe(df.head())

    alvo = st.selectbox("🎯 Selecione a coluna alvo", df.columns)
    y = df[alvo]

    # Tratamento de NaN
    if y.isna().sum() > 0:
        if pd.api.types.is_numeric_dtype(y):
            y = y.fillna(y.mean())
        else:
            y = y.fillna(y.mode()[0])
        st.warning("⚠️ Valores NaN no alvo foram tratados automaticamente.")

    # Detectar tipo de problema
    if pd.api.types.is_numeric_dtype(y) and y.nunique() > 15:
        problema = "regressao"
    else:
        problema = "classificacao"

    st.info(f"🔎 Detectado problema de **{problema.upper()}**")

    X = df.drop(columns=[alvo]).select_dtypes(include=[np.number])
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    X = pipeline.fit_transform(X)

    # Split adaptativo
    test_size = 0.2 if len(y) >= 10 else 0.1
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42,
            stratify=y if problema == "classificacao" else None
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

    # Modelos
    if problema == "classificacao":
        modelos = {
            "Regressão Logística": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(n_estimators=150, random_state=42),
            "XGBoost": XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=6,
                                     use_label_encoder=False, eval_metric="logloss", random_state=42)
        }
    else:
        modelos = {
            "Regressão Linear": LinearRegression(),
            "Random Forest Regressor": RandomForestRegressor(n_estimators=150, random_state=42),
            "XGBRegressor": XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42)
        }

    resultados, modelos_treinados = {}, {}
st.subheader("🤖 Treinamento dos Modelos")

for nome, modelo in modelos.items():
    inicio = time.time()

    # Proteção contra dataset vazio
    if len(y_train) == 0 or len(X_train) == 0:
        st.error("⚠️ Conjunto de treino vazio. Use mais linhas ou desative o modo rápido.")
        continue

    try:
        modelo.fit(X_train, y_train)
        tempo = time.time() - inicio

        # Previsões
        y_pred = modelo.predict(X_test) if len(y_test) > 0 else []

        # Métricas
        metricas = calcular_metricas(y_test, y_pred, problema) if len(y_test) > 0 else {}
        metricas["tempo"] = tempo

        # Validação cruzada
        if problema == "classificacao":
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            f1_cv = cross_val_score(modelo, X, y, cv=cv, scoring="f1_weighted").mean()
            metricas["f1_cv"] = f1_cv
        else:
            cv = 5
            r2_cv = cross_val_score(modelo, X, y, cv=cv, scoring="r2").mean()
            metricas["R2_cv"] = r2_cv

        resultados[nome] = metricas
        modelos_treinados[nome] = modelo
        st.success(f"{nome} treinado em {tempo:.2f}s")

    except ValueError as e:
        st.error(f"❌ Erro ao treinar {nome}: {e}")
    # Abas
    aba1, aba2, aba3 = st.tabs([
        "📊 Comparação dos Modelos",
        "📑 Relatório Final",
        "📈 Dashboard Interativo"
    ])

    with aba1:
        st.subheader("📊 Comparação dos Modelos")
        df_resultados = pd.DataFrame(resultados).T
        st.dataframe(df_resultados)

    with aba2:
        st.subheader("📑 Relatório Final")

        if problema == "classificacao":
            melhor_modelo = df_resultados["f1_cv"].idxmax()
            criterio = "F1-Score médio em validação cruzada"
            melhor_valor = df_resultados.loc[melhor_modelo, "f1_cv"]
        else:
            melhor_modelo = df_resultados["R2_cv"].idxmax()
            criterio = "R² médio em validação cruzada"
            melhor_valor = df_resultados.loc[melhor_modelo, "R2_cv"]

        relatorio_texto = f"""
        O modelo escolhido foi {melhor_modelo} porque apresentou o melhor desempenho
        segundo o critério de {criterio}, com valor de {melhor_valor:.4f}.
        """
        st.markdown(relatorio_texto)

        st.subheader("💾 Exportar Relatórios")

        # Exportar CSV
        csv_buffer = "Relatório Final\n" + relatorio_texto + "\n\n" + df_resultados.to_csv()
        csv = csv_buffer.encode("utf-8")
        st.download_button("⬇️ Download CSV", csv, "relatorio.csv", "text/csv")

        # Exportar Excel
        with pd.ExcelWriter("relatorio.xlsx", engine="xlsxwriter") as writer:
            df_resultados.to_excel(writer, sheet_name="Resultados")
            pd.DataFrame({"Relatório": [relatorio_texto]}).to_excel(writer, sheet_name="Relatorio")
        with open("relatorio.xlsx", "rb") as f:
            st.download_button("⬇️ Download Excel", f, "relatorio.xlsx")

        # Exportar SQLite
        conn = sqlite3.connect("relatorio.db")
        df_resultados.to_sql("resultados", conn, if_exists="replace", index=False)
        pd.DataFrame({"Relatório": [relatorio_texto]}).to_sql("relatorio", conn, if_exists="replace", index=False)
        conn.close()
        with open("relatorio.db", "rb") as f:
            st.download_button("⬇️ Download SQLite", f, "relatorio.db")

    with aba3:
        st.subheader("📈 Dashboard Interativo")

                # Distribuição do alvo
        if len(y) > 0:
            dist_df = y.value_counts().reset_index()
            dist_df.columns = ["Classe/Valor", "Contagem"]

            fig = px.bar(
                dist_df,
                x="Classe/Valor",
                y="Contagem",
                labels={"Classe/Valor": "Classe/Valor", "Contagem": "Contagem"},
                title="Distribuição do Alvo"
            )
            st.plotly_chart(fig, use_container_width=True)