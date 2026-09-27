# Trabalhos — Recuperação de Informação e Busca Semântica

Trabalhos práticos da disciplina **Tendências em Ciência da Computação / Tópicos Avançados** (UNIPÊ), Prof. Me. Ricardo Roberto de Lima.

| Pasta | Trabalho | Valor | Entregáveis |
|---|---|---|---|
| [`lab04_agrosearch/`](lab04_agrosearch) | **Lab 04 — AgroSearch**: pré-processamento, índice invertido e TF-IDF implementados do zero | 0,5 extra na AV1 | `agrosearch_app.py`, `RELATORIO.pdf` |
| [`lab05_healthsearch/`](lab05_healthsearch) | **Lab 05 — HealthSearch**: busca híbrida BM25 + embeddings com Reciprocal Rank Fusion (+ bônus Cross-Encoder) | 1,0 extra na Unidade II | `healthsearch_app.py`, `RELATORIO.pdf` |
| [`desafio_ouvidoria/`](desafio_ouvidoria) | **Ouvidoria Inteligente**: representações vetoriais, duplicatas, chunking e buscador semântico | Desafio prático | 3 notebooks, `app_ouvidoria.py`, `RELATORIO.pdf` |

## Como executar

```bash
pip install -r requirements.txt

streamlit run lab04_agrosearch/agrosearch_app.py
streamlit run lab05_healthsearch/healthsearch_app.py
streamlit run desafio_ouvidoria/app_ouvidoria.py
```

Na primeira execução, os modelos (`intfloat/multilingual-e5-small`, `paraphrase-multilingual-MiniLM-L12-v2` e os Cross-Encoders) são baixados do Hugging Face. Os notebooks da Ouvidoria já estão salvos com as saídas; para reexecutar, abra a partir da pasta `desafio_ouvidoria/`, porque eles leem `data/manifestacoes.json`.

## Lab 04 — AgroSearch

- **Sem scikit-learn**, como exige o enunciado: tokenização, normalização, stopwords, stemmer por regras, índice invertido, TF, IDF (`log10(N/df)`) e TF-IDF feitos à mão.
- Checkboxes para ligar/desligar stopwords e stemming e ver o vocabulário mudar.
- Ranking do maior para o menor TF-IDF acumulado, com o vencedor destacado e o cálculo termo a termo.
- **Bônus:** similaridade de cosseno entre o vetor TF-IDF da consulta e o de cada documento.

## Lab 05 — HealthSearch

- BM25 (`rank_bm25`) com sliders de `k1` e `b`, e tabela com o cálculo termo a termo.
- Busca semântica com `intfloat/multilingual-e5-small`, que acerta o Top-1 nas consultas médicas de teste; os modelos MiniLM ficam como opção de comparação.
- RRF: `α·1/(k+Rank_BM25) + (1−α)·1/(k+Rank_Sem)`, com `k = 60`.
- Abas Léxico, Semântico, Híbrido RRF e Matriz Comparativa, com latência de cada motor.
- **Bônus:** re-ranking do Top-3 com Cross-Encoder (multilíngue mMARCO ou o `ms-marco-MiniLM-L-6-v2` do enunciado).

## Ouvidoria Inteligente

| Entrega | Arquivo | Resultado principal |
|---|---|---|
| 1 | `analise_comparativa.ipynb` | BoW/TF-IDF dão **0** para "posto sem médico" × "falta atendimento no PSF"; o embedding multilíngue dá **0,67** |
| 2 | `deteccao_duplicatas.ipynb` | Com o limiar 0,85 nada é encontrado; **0,65** tem o melhor F1 (0,667) e coincide com o percentil 99 |
| 3 | `chunking_manifestacoes.ipynb` | 3 configurações testadas; (300, 60) preserva melhor a denúncia, e o overlap ajuda onde a frase é cortada |
| 4 | `app_ouvidoria.py` | Abas Busca Semântica, Base Completa, Espaço Vetorial e Chunking (+ Duplicatas) |

> **Sobre a base de dados:** o arquivo `manifestacoes.json` citado no enunciado não veio junto com o material. `data/manifestacoes.json` é uma base montada conforme a especificação (40 manifestações, 5 categorias, 15% de duplicatas, 5 textos longos, com os textos de M003, M017, M008, M022 e M031 do enunciado), gerada por `data/gerar_base.py`. Para usar a base oficial, substitua o arquivo e atualize `data/duplicatas_gabarito.json`.

## Antes de entregar

- Preencha os nomes dos integrantes nos relatórios (`[Integrante 1]`, `[Integrante 2]`, …).
- O enunciado da Ouvidoria escreve `análise_comparativa.ipynb` com acento; aqui o nome ficou sem acento para evitar problemas de codificação. Renomeie se o professor exigir o nome exato.
