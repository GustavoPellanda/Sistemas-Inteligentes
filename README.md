# Sistemas Inteligentes — UTFPR Curitiba

Este repositório reúne as resoluções das tarefas da matéria **Sistemas Inteligentes**, do curso de **Bacharelado em Sistemas de Informação** da **UTFPR - Curitiba**.

Os experimentos comparam árvores CART e redes neurais MLP em duas tarefas com dados sintéticos de vítimas:

- [Classificador.ipynb](Classificador.ipynb): classificação da triagem START (`tri`).
- [Regressor.ipynb](Regressor.ipynb): estimativa da probabilidade de sobrevivência (`sobr`).

Cada tarefa avalia três configurações com cinco folds de validação cruzada, retreina os melhores modelos com 10.000 amostras e os avalia no teste cego de 1.300 vítimas fornecido pelo professor.

## Organização

```text
Classificador.ipynb
Regressor.ipynb
dados/
  treino/              # Datasets das duas tarefas
  teste/               # Dataset fornecido pelo professor
docs/
  enunciados/          # Enunciados das tarefas
  relatorios/          # Relatórios finais em PDF
graficos/regressor/    # Gráficos real × predito
modelos/
  classificador/       # CART e pipeline da MLP com scaler
  regressor/           # CART, MLP e scaler separado
src/                   # Gerador de dados sintéticos
requirements.txt
```

Os relatórios finais estão disponíveis em [Classificador](docs/relatorios/Relatorio_Classificador.pdf) e [Regressor](docs/relatorios/Relatorio_Regressor.pdf). Dados, modelos, gráficos e relatórios das entregas são versionados; o ambiente virtual e os caches são ignorados.

## Execução

Use Python 3.12 e instale as dependências:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Abra os notebooks em um editor com suporte a Jupyter, selecione o Python da `.venv`, reinicie o kernel e execute as células em ordem, com a raiz do repositório como diretório de trabalho. As versões das dependências correspondem ao ambiente usado nos experimentos.

O teste cego fica em `dados/teste/data.csv` e participa somente da avaliação final. A execução direta de `src/gerar_dados_vitimas.py` grava experimentos avulsos em `dados/gerados/1300v/`; os notebooks configuram seus próprios arquivos de treinamento.
