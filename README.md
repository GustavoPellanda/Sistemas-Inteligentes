# Sistemas Inteligentes — UTFPR Curitiba

Este repositório reúne as resoluções das tarefas da matéria **Sistemas Inteligentes**, do curso de **Bacharelado em Sistemas de Informação** da **UTFPR - Curitiba**.

As três tarefas utilizam dados de vítimas do sistema VictSim:

- [01 - Classificador.ipynb](01%20-%20Classificador.ipynb): classificação da triagem START (`tri`) com CART e MLP.
- [02 - Regressor.ipynb](02%20-%20Regressor.ipynb): estimativa da probabilidade de sobrevivência (`sobr`) com CART e MLP.
- [03 - Clustering.ipynb](03%20-%20Clustering.ipynb): agrupamento das 410 vítimas do mapa com K-Means, reutilizando as MLPs anteriores para inferir `tri` e `sobr`.

As tarefas de classificação e regressão avaliam três configurações com cinco folds de validação cruzada, retreinam os melhores modelos com 10.000 amostras e os avaliam no teste cego de 1.300 vítimas fornecido pelo professor. O clustering compara K de 2 a 12, primeiro com posição espacial e depois com posição e sobrevivência padronizadas. A estratégia principal selecionou K = 7, com silhueta 0,376392 e cobertura de 100%; a atribuição de agentes e a sequência de socorro são discutidas conceitualmente no relatório.

## Organização

```text
01 - Classificador.ipynb
02 - Regressor.ipynb
03 - Clustering.ipynb
dados/
  treino/              # Datasets das duas tarefas
  teste/               # Dataset fornecido pelo professor
  map.csv              # Mapa original do cenário 75×75 com 410 vítimas
docs/
  enunciados/          # Enunciados das tarefas
  relatorios/          # Relatórios finais em PDF
resultados/
  regressao/           # Gráficos real × predito
  clustering/          # Figuras finais e cluster_*.txt
modelos/
  classificador/       # CART e pipeline da MLP com scaler
  regressor/           # CART, MLP e scaler separado
src/                   # Gerador de dados e funções de clustering
requirements.txt
```

Os relatórios finais estão disponíveis em [Classificador](docs/relatorios/Relatorio_Classificador.pdf), [Regressor](docs/relatorios/Relatorio_Regressor.pdf) e [Clustering](docs/relatorios/Relatorio_Clustering.pdf). Os gráficos da regressão ficam em [resultados/regressao/](resultados/regressao/) e as entregas de agrupamento em [resultados/clustering/](resultados/clustering/). Dados, modelos, gráficos e relatórios das entregas são versionados; o ambiente virtual e os caches são ignorados.

## Execução

Use Python 3.12 e instale as dependências:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Abra os notebooks em um editor com suporte a Jupyter, selecione o Python da `.venv`, reinicie o kernel e execute as células em ordem, com a raiz do repositório como diretório de trabalho. As versões das dependências correspondem ao ambiente usado nos experimentos.

Para a Tarefa 3, basta executar o terceiro notebook: os modelos treinados já estão em `modelos/`. Ele regenera métricas, figuras e arquivos `cluster_*.txt` (um ID por linha), sem repetir os treinamentos anteriores. Também é possível executá-lo pelo terminal:

```bash
jupyter execute --inplace "03 - Clustering.ipynb"
```

Nas tabelas de clustering, **SSE = √inércia** segue a menção à raiz quadrada no enunciado. Os DataFrames distinguem essa medida, a inércia (SSE convencional) e o RMSE; a escolha de K usa os resultados no mesmo espaço e escala. A visualização final distingue o cluster pelo contorno da célula e a classe `tri` pela cor do círculo, com obstáculos e base.

A visualização em Matplotlib utiliza o mapa e as predições. O [programa de referência do VictSim3](https://github.com/tacla/VictSim3/blob/main/data_vis_and_stats/plot_clusters_and_sequencing.py) também exige configuração do cenário, ground truth e Pygame para a execução completa. Seu leitor seleciona qualquer arquivo iniciado por `cluster`; por isso, a imagem se chama `agrupamentos_grid.png`, preservando a leitura dos arquivos `cluster_*.txt` na mesma pasta.

O teste cego fica em `dados/teste/data.csv` e participa somente da avaliação final. A execução direta de `src/gerar_dados_vitimas.py` grava experimentos avulsos em `dados/gerados/1300v/`; os notebooks configuram seus próprios arquivos de treinamento.
