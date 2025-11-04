# Projeto: Classificação de gatos e cachorros

## Tecnologias e diretivas

- python, e pyenv/pipenv como gerenciador
- usar apenas 1 arquivo notebook (`.ipynb`)
- T-SNE
- fontes públicas
- APIs públicas e gratuítas
- algo simples, porém robusto (para economizar tokens, IMPORTANTE!)
- deve estar contido na pasta `src/project1/`
- configurar opções e parâmetros (variáveis) logo no começo de arquivo, abaixo das importações
- caso seja necessário criar pastas de imagens, gráficos, resultados, fique a vontade
- pesquise sobre limite nesste site <https://ai.google.dev/gemini-api/docs/rate-limits?hl=pt-br>
- Consultar @Pipfile na raiz da pasta

## Partes (para planejamento)

### 1 - Classificação com embeddings

- [ ] criar embeddings para conunto de de dados
- [ ] separar conjuntos de dados em treino teste
- [ ] Visualizar dados de treino com T-SNE
- [ ] Avaliar dados de treino com T-SNE

### 2 - Classificação com LLMs

- [ ] Realizar chamada para LLM através de APIs públicas
- [ ] Configurar LLM para realizar tarefa (com propmt e output)
- [ ] Avaliar desempenho de dados de teste

### 3 - Classificação de gatos e cachorros

- [ ] Criar serviço de classificação
- [ ] Usar KNN para classificação
  - [ ] Embeddar nova imagem
  - [ ] Comparar novo embedding com outros
- [ ] Usar API de LLM para realizar classificação
