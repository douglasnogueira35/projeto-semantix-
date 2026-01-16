# 📑 Projeto AutoML – Previsão de Renda

## 📌 Visão Geral
Este projeto utiliza **Streamlit** para criar uma aplicação interativa de **AutoML**, permitindo carregar datasets em diferentes formatos (CSV, Excel, SQLite, MySQL, PostgreSQL, SQL Server), treinar modelos automaticamente e gerar relatórios em múltiplos formatos (TXT, CSV, Excel, PDF e ZIP).

O objetivo principal é prever a **coluna alvo: `renda`**, aplicando técnicas de regressão e comparando diferentes algoritmos.

---

## ⚙️ Funcionalidades
- Upload de arquivos em **CSV, Excel, SQLite**.  
- Conexão com bancos externos (**MySQL, PostgreSQL, SQL Server**) via SQLAlchemy.  
- Detecção automática do tipo de problema (**classificação ou regressão**).  
- Treinamento de múltiplos modelos (Regressão Linear, Random Forest, XGBoost).  
- Avaliação com métricas adequadas (R², RMSE, F1, Accuracy).  
- Visualização interativa dos resultados com **Plotly**.  
- Relatórios finais disponíveis em **TXT, CSV, Excel, PDF e ZIP**.  

---

## 📊 Resultados
- **Coluna alvo:** `renda`  
- **Melhor modelo:** `XGBRegressor`  
- **Desempenho obtido:**
  ```json
  {
    "R2": 0.4059,
    "RMSE": 21439.14,
    "R2_cv": -0.3623,
    "tempo": 0.91
  }
  🔎 Insights de Negócio
O modelo de regressão pode apoiar:
- Previsões de vendas com base em renda estimada.
- Estimativas de receita futura para planejamento estratégico.
- Análise de impacto de variáveis econômicas sobre o poder de compra.
- Segmentação de clientes considerando faixas de renda previstas.

🚀 Como Executar
- Clone este repositório:
git clone <url-do-repositorio>
- Instale as dependências:
pip install -r requirements.txt
- Execute o app:
streamlit run app.py
- 
Claro, Douglas 🙌. Aqui está o README.md completo em um único bloco de texto, pronto para você copiar e colar direto no seu repositório ou projeto:
# 📑 Projeto AutoML – Previsão de Renda

## 📌 Visão Geral
Este projeto utiliza **Streamlit** para criar uma aplicação interativa de **AutoML**, permitindo carregar datasets em diferentes formatos (CSV, Excel, SQLite, MySQL, PostgreSQL, SQL Server), treinar modelos automaticamente e gerar relatórios em múltiplos formatos (TXT, CSV, Excel, PDF e ZIP).

O objetivo principal é prever a **coluna alvo: `renda`**, aplicando técnicas de regressão e comparando diferentes algoritmos.

---

## ⚙️ Funcionalidades
- Upload de arquivos em **CSV, Excel, SQLite**.  
- Conexão com bancos externos (**MySQL, PostgreSQL, SQL Server**) via SQLAlchemy.  
- Detecção automática do tipo de problema (**classificação ou regressão**).  
- Treinamento de múltiplos modelos (Regressão Linear, Random Forest, XGBoost).  
- Avaliação com métricas adequadas (R², RMSE, F1, Accuracy).  
- Visualização interativa dos resultados com **Plotly**.  
- Relatórios finais disponíveis em **TXT, CSV, Excel, PDF e ZIP**.  

---

## 📊 Resultados
- **Coluna alvo:** `renda`  
- **Melhor modelo:** `XGBRegressor`  
- **Desempenho obtido:**
  ```json
  {
    "R2": 0.4059,
    "RMSE": 21439.14,
    "R2_cv": -0.3623,
    "tempo": 0.91
  }



🔎 Insights de Negócio
O modelo de regressão pode apoiar:
- Previsões de vendas com base em renda estimada.
- Estimativas de receita futura para planejamento estratégico.
- Análise de impacto de variáveis econômicas sobre o poder de compra.
- Segmentação de clientes considerando faixas de renda previstas.

🚀 Como Executar
- Clone este repositório:
git clone <url-do-repositorio>
- Instale as dependências:
pip install -r requirements.txt
- Execute o app:
streamlit run app.py
- Acesse no navegador:
http://localhost:8501


Ou utilize diretamente a versão hospedada no Streamlit Cloud:
👉 Abrir aplicação

📥 Relatórios
O usuário pode baixar o relatório final em diferentes formatos:
- .txt → resumo textual
- .csv → tabela de métricas
- .xlsx → planilha Excel
- .pdf → relatório formatado
🛠️ Tecnologias Utilizadas
- Python 3.13
- Streamlit
- Pandas / NumPy
- Scikit-learn
- XGBoost
- Plotly
- ReportLab
- SQLAlchemy
🛠️ Tecnologias Utilizadas
- Python 3.13
- Streamlit
- Pandas / NumPy
- Scikit-learn
- XGBoost
- Plotly
- ReportLab
- SQLAlchemy

