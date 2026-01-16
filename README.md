# 📘 README – Projeto AutoML Inteligente

## 📌 Contexto
Este projeto tem como objetivo aplicar técnicas de **AutoML** para análise automática de dados, utilizando diferentes modelos de aprendizado de máquina para prever variáveis alvo.  
Na execução relatada, foi utilizada a variável **`renda`** como alvo, e o sistema identificou o problema como sendo de **regressão**.

---

## ⚡ Configuração
- Dataset reduzido para **100 linhas** (modo rápido).  
- Ferramenta utilizada: **Streamlit** para interface interativa.  
- Modelos avaliados:  
  - Regressão Linear  
  - Random Forest Regressor  
  - XGBRegressor  

---

## 🔍 Visualização dos Dados
- O usuário pôde explorar os dados carregados diretamente na interface.  
- A coluna alvo selecionada foi **`renda`**.  
- O sistema automaticamente detectou que se trata de um problema de **regressão**, pois a variável alvo é numérica e contínua.

---

## 🤖 Treinamento dos Modelos
Os seguintes modelos foram treinados com o dataset reduzido:

| Modelo                   | Tempo de Treinamento |
|---------------------------|----------------------|
| Regressão Linear          | 0,01s               |
| Random Forest Regressor   | 0,24s               |
| XGBRegressor              | 0,07s               |

---

## 📊 Comparação dos Modelos
Após o treinamento, os modelos foram comparados utilizando métricas de regressão, com destaque para o **R² médio em validação cruzada**.

---

## 📑 Relatório Final
O modelo escolhido foi **XGBRegressor**, pois apresentou o melhor desempenho segundo o critério de **R² médio em validação cruzada**, com valor de **-0.0125**.  

> Observação: O valor negativo de R² indica que os modelos não conseguiram explicar bem a variabilidade da variável alvo `renda` neste dataset reduzido. Isso pode ocorrer devido ao tamanho pequeno da amostra ou à ausência de variáveis explicativas relevantes.

---

## 📈 Dashboard Interativo
O projeto inclui um **dashboard interativo** que permite:
- Visualizar a distribuição da variável alvo.  
- Explorar métricas de desempenho dos modelos.  
- Analisar a matriz de confusão (em problemas de classificação).  
- Avaliar a importância das variáveis (em modelos baseados em árvores, como Random Forest e XGBRegressor).  

---

## 🎯 Conclusões
- A variável alvo escolhida foi **`renda`**.  
- O problema foi corretamente identificado como **regressão**.  
- Entre os modelos testados, o **XGBRegressor** apresentou o melhor desempenho, ainda que com R² negativo.  
- O resultado sugere que, para melhorar a performance, seria necessário:
  - Utilizar um dataset maior (mais linhas).  
  - Incluir variáveis explicativas adicionais.  
  - Realizar ajustes de hiperparâmetros nos modelos.  

---

## 🚀 Próximos Passos
- Expandir o dataset para além das 100 linhas do modo rápido.  
- Explorar novas variáveis e criar features derivadas.  
- Testar técnicas de regularização e tuning de hiperparâmetros.  
- Avaliar métricas adicionais como RMSE e MAE para complementar a análise.
