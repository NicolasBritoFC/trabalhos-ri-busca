# Busca léxica, semântica e híbrida · trabalhos práticos

Entregas da disciplina **Tendências em Ciência da Computação / Tópicos Avançados** (UNIPÊ), com o Prof. Me. Ricardo Roberto de Lima. Cada pasta é um trabalho independente, com o próprio app Streamlit e relatório em PDF.

```
.
├── lab04_agrosearch/        Lab 04 · TF-IDF e índice invertido feitos à mão
├── lab05_healthsearch/      Lab 05 · BM25 + embeddings + Reciprocal Rank Fusion
└── desafio_ouvidoria/       Desafio · representações, duplicatas, chunking e buscador
    ├── nucleo_ouvidoria.py  lógica compartilhada entre o app e os notebooks
    └── data/                base de manifestações e gabarito de duplicatas
```

## Preparar o ambiente

```bash
python -m venv .venv
.venv\Scripts\activate          # Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

Na primeira execução, os apps de embeddings baixam os modelos do Hugging Face (alguns minutos).

---

### Lab 04 · AgroSearch

`streamlit run lab04_agrosearch/agrosearch_app.py`

Busca nos manuais da AgroTech com ranking por TF-IDF. O código é organizado em três classes, todas sem scikit-learn, como pede o enunciado:

- `PipelineTexto`: tokenização, normalização, stopwords e um stemmer em quatro estágios inspirado no RSLP
- `IndiceInvertido`: postings list, df e IDF (`log10(N/df)`)
- `RanqueadorTFIDF`: TF, TF-IDF acumulado e, como bônus, o cosseno entre os vetores TF-IDF

A tela abre direto no resultado, com o trecho vencedor em destaque. Os bastidores (pipeline etapa por etapa, índice invertido em tabela e JSON, memória de cálculo e cosseno) ficam na seção "Por dentro do motor". Stopwords e stemming ligam e desligam na barra lateral.

### Lab 05 · HealthSearch

`streamlit run lab05_healthsearch/healthsearch_app.py`

Busca híbrida em protocolos clínicos: BM25 acerta códigos exatos (`CÓD-ECG-12D`) e os embeddings acertam sinônimos ("ataque cardíaco" → infarto).

| Fase | Onde está |
|---|---|
| 1 · limpeza (minúsculas, acentos, caracteres especiais, stopwords) | `LimpadorClinico` |
| 2 · BM25 com k1 (0–3, padrão 1,2) e b (0–1, padrão 0,75) na sidebar | `MotorLexico` |
| 3 · embeddings + cosseno (`intfloat/multilingual-e5-small` por padrão) | `MotorSemantico` |
| 4 · `α·1/(k+Rank_BM25) + (1−α)·1/(k+Rank_Sem)`, k = 60 | `fundir_rrf` |
| Bônus · Cross-Encoder sobre o Top-3 do RRF | `reranquear` |

Os três motores devolvem o mesmo objeto `Ranking`. A tela mostra um pódio com o Top-3 de cada motor e as abas Léxico, Semântico, Híbrido RRF e Matriz Comparativa.

### Desafio · Ouvidoria Inteligente

`streamlit run desafio_ouvidoria/app_ouvidoria.py`

| Entrega | Arquivo | O que mostra |
|---|---|---|
| 1 | `analise_comparativa.ipynb` | BoW e TF-IDF dão cosseno **0** para "posto sem médico" × "falta atendimento no PSF"; o embedding multilíngue dá **0,67** |
| 2 | `deteccao_duplicatas.ipynb` | `detectar_duplicatas` com o limiar 0,85 não acha nada; **0,65** tem o melhor F1 e bate com o percentil 99 |
| 3 | `chunking_manifestacoes.ipynb` | Três configurações de `(chunk_size, overlap)`; (300, 60) preserva melhor cada denúncia |
| 4 | `app_ouvidoria.py` | Abas 🔍 Busca Semântica, 📋 Base Completa, 🌐 Espaço Vetorial, 🧩 Chunking e 🔁 Duplicatas |

Os notebooks estão salvos com as saídas. Para reexecutar, abra a partir de `desafio_ouvidoria/`, porque eles importam `nucleo_ouvidoria.py`.

**Sobre a base:** o `manifestacoes.json` do enunciado não veio no material. A base em `data/` foi montada seguindo a especificação: 40 manifestações, 5 categorias, 15% de duplicatas, 5 textos longos e os textos de M003, M017, M008, M022 e M031 citados no enunciado. Ela é gerada por `data/gerar_base.py`. Para usar a base oficial, troque o arquivo e atualize `data/duplicatas_gabarito.json`.
