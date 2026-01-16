import streamlit as st
import pandas as pd
import numpy as np
import time
import io
import sqlite3
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from xgboost import XGBClassifier, XGBRegressor
import plotly.express as px
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

st.title("🤖 Projeto AutoML Inteligente")

# =========================================
# 1. Upload de arquivos locais
# =========================================
arquivo = st.file_uploader(
    "Carregue seu arquivo (CSV, Excel, SQLite)", 
    type=["csv", "xlsx", "xls", "db", "sqlite"]
)

df = None

if arquivo is not None:
    nome = arquivo.name.lower()
    try:
        if nome.endswith(".csv"):
            df = pd.read_csv(arquivo)
        elif nome.endswith((".xlsx", ".xls")):
            df = pd.read_excel(arquivo)
        elif nome.endswith((".db", ".sqlite")):
            con = sqlite3.connect(arquivo)
            tabelas = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", con)
            st.write("📋 Tabelas disponíveis:", tabelas)
            primeira_tabela = tabelas.iloc[0,0]
            df = pd.read_sql(f"SELECT * FROM {primeira_tabela}", con)

        st.success(f"✅ Arquivo {nome} carregado com sucesso!")
        st.write("📊 Visualização inicial dos dados:", df.head())

    except Exception as e:
        st.error(f"❌ Erro ao carregar arquivo: {e}")

# =========================================
# 2. Conexão com bancos SQL externos
# =========================================
st.sidebar.subheader("🔗 Conexão com Banco SQL")
tipo_banco = st.sidebar.selectbox("Escolha o banco", ["Nenhum", "MySQL", "PostgreSQL", "SQL Server"])

if tipo_banco != "Nenhum":
    usuario = st.sidebar.text_input("Usuário")
    senha = st.sidebar.text_input("Senha", type="password")
    host = st.sidebar.text_input("Host", "localhost")
    porta = st.sidebar.text_input("Porta", "3306" if tipo_banco=="MySQL" else "5432")
    banco = st.sidebar.text_input("Nome do banco")

    if st.sidebar.button("Conectar"):
        try:
            if tipo_banco == "MySQL":
                engine = create_engine(f"mysql+pymysql://{usuario}:{senha}@{host}:{porta}/{banco}")
            elif tipo_banco == "PostgreSQL":
                engine = create_engine(f"postgresql+psycopg2://{usuario}:{senha}@{host}:{porta}/{banco}")
            elif tipo_banco == "SQL Server":
                engine = create_engine(f"mssql+pyodbc://{usuario}:{senha}@{host}:{porta}/{banco}?driver=ODBC+Driver+17+for+SQL+Server")

            tabelas = pd.read_sql("SELECT table_name FROM information_schema.tables WHERE table_schema='public';", engine)
            st.write("📋 Tabelas disponíveis:", tabelas)

            primeira_tabela = tabelas.iloc[0,0]
            df = pd.read_sql(f"SELECT * FROM {primeira_tabela}", engine)

            st.success("✅ Conexão estabelecida e dados carregados!")
            st.write(df.head())

        except Exception as e:
            st.error(f"❌ Erro na conexão: {e}")

# =========================================
# 3. Fluxo AutoML (se df carregado)
# =========================================
if df is not None:
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

    # Seleção da coluna alvo
    alvo = st.selectbox("🎯 Selecione a coluna alvo", df.columns)
    y = df[alvo]
    X = df.drop(columns=[alvo])

    # Pré-processamento
    if "data_ref" in X.columns:
        X["data_ref"] = pd.to_datetime(X["data_ref"], errors="coerce").astype(int) / 10**9
    X = pd.get_dummies(X, drop_first=True).fillna(0)

    # Detecção do tipo de problema
    if pd.api.types.is_numeric_dtype(y) and y.nunique() > 15:
        problema = "regressao"
        y = pd.to_numeric(y, errors="coerce").fillna(y.mean())
    else:
        problema = "classificacao"
        y = y.fillna(y.mode()[0])
    st.info(f"🔎 Detectado problema de **{problema.upper()}**")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Modelos
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

    resultados = {}
    st.subheader("🤖 Treinamento dos Modelos")

    for nome, modelo in modelos.items():
        inicio = time.time()
        try:
            modelo.fit(X_train, y_train)
            tempo = time.time() - inicio
            y_pred = modelo.predict(X_test)
            metricas = {}

            if problema == "classificacao":
                metricas["accuracy"] = accuracy_score(y_test, y_pred)
                metricas["f1"] = f1_score(y_test, y_pred, average="weighted")
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                metricas["f1_cv"] = cross_val_score(modelo, X, y, cv=cv, scoring="f1_weighted").mean()
            else:
                metricas["R2"] = r2_score(y_test, y_pred)
                metricas["RMSE"] = np.sqrt(mean_squared_error(y_test, y_pred))
                metricas["R2_cv"] = cross_val_score(modelo, X, y, cv=5, scoring="r2").mean()

            metricas["tempo"] = tempo
            resultados[nome] = metricas

            st.success(f"{nome} treinado em {tempo:.2f}s")
            st.write("📈 Métricas:", metricas)

        except Exception as e:
            st.error(f"❌ Erro ao treinar {nome}: {e}")

    # Abas
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

        relatorio = f"""
        Relatório Final:
        - Tipo de problema: {problema.upper()}
        - Melhor modelo: {nome_modelo}
        - Principais métricas: {metricas}
        """

        # TXT
        st.download_button(
            label="📥 Baixar relatório em TXT",
            data=relatorio.encode("utf-8"),
            file_name="relatorio_final.txt",
            mime="text/plain"
        )

        # CSV
        st.download_button(
            label="📥 Baixar relatório em CSV",
            data=pd.DataFrame([metricas]).to_csv(index=False).encode("utf-8"),
            file_name="relatorio_final.csv",
            mime="text/csv"
        )

        # Excel
        buffer_excel = io.BytesIO()
        pd.DataFrame([metricas]).to_excel(buffer_excel, index=False)
        st.download_button(
            label="📥 Baixar relatório em Excel",
            data=buffer_excel.getvalue(),
            file_name="relatorio_final.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        # PDF
        buffer_pdf = io.BytesIO()
        c = canvas.Canvas(buffer_pdf, pagesize=letter)
        c.drawString(50, 750, "Relatório Final")
        c.drawString(50, 730, f"Tipo de problema: {problema.upper()}")
        c.drawString(50, 710, f"Melhor modelo: {nome_modelo}")
        c.drawString(50, 690, f"Métricas: {metricas}")
        c.save()
        pdf_bytes = buffer_pdf.getvalue()
        buffer_pdf.close()

        st.download_button(
            label="📥 Baixar relatório em PDF",
            data=pdf_bytes,
            file_name="relatorio_final.pdf",
            mime="application/pdf"
        )