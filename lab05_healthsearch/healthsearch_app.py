# =============================================================================
# HealthSearch — Motor de Busca Híbrido (BM25 + Busca Semântica + RRF)
# Laboratório Prático 05 — Desafio Integrador
# Disciplina: Tendências em Ciência da Computação — Prof. Me. Ricardo Roberto de Lima
#
# Fases implementadas:
#   1. Ingestão do corpus médico e pré-processamento (minúsculas, sem acentos,
#      sem caracteres especiais, sem stopwords em português)
#   2. Motor léxico Okapi BM25 com k1 e b ajustáveis na sidebar
#   3. Motor semântico com embeddings densos + similaridade de cosseno
#   4. Fusão Reciprocal Rank Fusion (RRF) ponderada por alfa
#   Bônus: re-ranking com Cross-Encoder sobre o Top-3 do RRF
#
# Execução:  streamlit run healthsearch_app.py
# =============================================================================

import re
import time
import unicodedata

import numpy as np
import pandas as pd
import streamlit as st
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

# -----------------------------------------------------------------------------
# CORPUS MÉDICO (hardcode obrigatório do enunciado)
# -----------------------------------------------------------------------------
CORPUS = [
    {"id": "Doc 1", "titulo": "Protocolo Emergência ECG",
     "conteudo": "Pacientes com dor precordial aguda e suspeita de síndrome coronariana devem realizar "
                 "eletrocardiograma CÓD-ECG-12D em até 10 minutos."},
    {"id": "Doc 2", "titulo": "Guia de Farmacologia Cardíaca",
     "conteudo": "O uso imediato de ácido acetilsalicílico e antiagregantes plaquetários reduz a "
                 "mortalidade no infarto agudo do miocárdio."},
    {"id": "Doc 3", "titulo": "Diretriz de Hipertensão Arterial",
     "conteudo": "A crise hipertensiva severa requer administração de anti-hipertensivos venosos e "
                 "monitoramento contínuo da pressão arterial na UTI."},
    {"id": "Doc 4", "titulo": "Manual de AVC Isquêmico",
     "conteudo": "O acidente vascular cerebral isquêmico agudo deve ser tratado com trombolíticos "
                 "venosos em até quatro horas e meia do início dos sintomas."},
    {"id": "Doc 5", "titulo": "Protocolo de Reanimação RCR",
     "conteudo": "Parada cardiorrespiratória em adultos exige compressões torácicas contínuas de alta "
                 "qualidade e desfibrilação precoce no código azul."},
    {"id": "Doc 6", "titulo": "Procedimentos de UTI Geral",
     "conteudo": "Para diagnóstico do protocolo CÓD-ECG-12D em arritmias complexas, recomenda-se a "
                 "monitorização cardíaca contínua por telemetria."},
]

# Consultas que expõem os pontos cegos de cada motor
CONSULTAS_TESTE = [
    "ataque cardíaco",                 # sinônimo leigo de infarto → só o semântico acha Doc 2
    "CÓD-ECG-12D",                     # código exato → BM25 acerta Doc 1 e Doc 6
    "derrame cerebral",                # sinônimo leigo de AVC → Doc 4
    "coração parou de bater",          # parada cardiorrespiratória → Doc 5
    "exame CÓD-ECG-12D para infarto",  # mistura código + conceito → a híbrida brilha
    "pressão muito alta",              # crise hipertensiva → Doc 3
]

MODELOS_EMBEDDING = [
    "intfloat/multilingual-e5-small",  # melhor desempenho nas consultas médicas de teste
    "paraphrase-multilingual-MiniLM-L12-v2",
    "sentence-transformers/all-MiniLM-L6-v2",
]
# A família E5 foi treinada com prefixos: "query: " na consulta e "passage: " nos documentos
PREFIXOS = {"intfloat/multilingual-e5-small": ("query: ", "passage: ")}

MODELOS_CROSS = [
    "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",  # multilíngue (entende português)
    "cross-encoder/ms-marco-MiniLM-L-6-v2",        # sugerido no enunciado (treinado em inglês)
]

STOPWORDS_PT = {
    "a", "o", "as", "os", "um", "uma", "uns", "umas", "de", "do", "da", "dos", "das", "em",
    "no", "na", "nos", "nas", "por", "pelo", "pela", "pelos", "pelas", "para", "com", "sem",
    "e", "ou", "que", "se", "ao", "aos", "ate", "sobre", "entre", "ser", "sao", "foi", "deve",
    "devem", "pode", "muito", "mais", "meia", "seu", "sua", "seus", "suas", "este", "esta",
    "esse", "essa", "isso", "como", "quando", "onde", "qual", "tem", "ter", "ha", "nao",
    "recomenda", "requer", "exige", "uso",
}


# =============================================================================
# FASE 1 — PRÉ-PROCESSAMENTO
# =============================================================================
def normalizar(texto: str) -> str:
    """Minúsculas, remove acentos e caracteres especiais (preserva hífens de códigos)."""
    texto = unicodedata.normalize("NFD", texto.lower())
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    texto = re.sub(r"[^\w\s-]", " ", texto)
    return re.sub(r"\s+", " ", texto).strip()


def tokenizar(texto: str) -> list[str]:
    """Normaliza, divide em tokens e remove stopwords em português."""
    tokens = [t.strip("-") for t in normalizar(texto).split()]
    return [t for t in tokens if t and t not in STOPWORDS_PT]


def texto_completo(doc: dict) -> str:
    return f"{doc['titulo']}. {doc['conteudo']}"


# =============================================================================
# FASE 2 — MOTOR LÉXICO (Okapi BM25)
# =============================================================================
def busca_bm25(consulta: str, k1: float, b: float):
    corpus_tokens = [tokenizar(texto_completo(d)) for d in CORPUS]
    bm25 = BM25Okapi(corpus_tokens, k1=k1, b=b)
    q_tokens = tokenizar(consulta)
    scores = bm25.get_scores(q_tokens) if q_tokens else np.zeros(len(CORPUS))
    # Só entra no ranking léxico quem tem pelo menos um termo da consulta (score > 0).
    ordem = [i for i in np.argsort(scores)[::-1] if scores[i] > 0]
    ranks = {CORPUS[i]["id"]: pos + 1 for pos, i in enumerate(ordem)}
    return scores, ranks, bm25, corpus_tokens, q_tokens


# =============================================================================
# FASE 3 — MOTOR SEMÂNTICO (Embeddings + Cosseno)
# =============================================================================
@st.cache_resource(show_spinner="Carregando modelo de embeddings...")
def carregar_modelo(nome: str) -> SentenceTransformer:
    return SentenceTransformer(nome)


@st.cache_resource(show_spinner="Carregando Cross-Encoder...")
def carregar_cross_encoder(nome: str) -> CrossEncoder:
    return CrossEncoder(nome)


@st.cache_data(show_spinner="Gerando embeddings do corpus...")
def embeddings_corpus(nome_modelo: str) -> np.ndarray:
    modelo = carregar_modelo(nome_modelo)
    # normalize_embeddings=True → vetores de norma 1: o produto escalar é o cosseno
    _, pref_doc = PREFIXOS.get(nome_modelo, ("", ""))
    return modelo.encode([pref_doc + texto_completo(d) for d in CORPUS], normalize_embeddings=True)


def busca_semantica(consulta: str, nome_modelo: str):
    modelo = carregar_modelo(nome_modelo)
    emb_docs = embeddings_corpus(nome_modelo)
    pref_q, _ = PREFIXOS.get(nome_modelo, ("", ""))
    emb_q = modelo.encode([pref_q + consulta], normalize_embeddings=True)[0]
    sims = emb_docs @ emb_q
    ordem = np.argsort(sims)[::-1]
    ranks = {CORPUS[i]["id"]: pos + 1 for pos, i in enumerate(ordem)}
    return sims, ranks


# =============================================================================
# FASE 4 — RECIPROCAL RANK FUSION
# =============================================================================
def rrf(ranks_bm25: dict, ranks_sem: dict, alpha: float, k_rrf: int) -> dict:
    """Score_RRF(D) = α·1/(k + Rank_BM25) + (1−α)·1/(k + Rank_Sem).
    Documento ausente de uma lista não recebe contribuição dela."""
    scores = {}
    for doc in CORPUS:
        d = doc["id"]
        s = 0.0
        if d in ranks_bm25:
            s += alpha * (1.0 / (k_rrf + ranks_bm25[d]))
        if d in ranks_sem:
            s += (1 - alpha) * (1.0 / (k_rrf + ranks_sem[d]))
        scores[d] = s
    return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))


# =============================================================================
# INTERFACE
# =============================================================================
st.set_page_config(page_title="HealthSearch", page_icon="🩺", layout="wide")
st.title("🩺 HealthSearch — Busca Híbrida BM25 + Semântica (RRF)")
st.caption("HealthTech Solutions · Protótipo de motor de busca para protocolos clínicos de pronto-socorro")

with st.sidebar:
    st.header("🔤 Motor Léxico (BM25)")
    k1 = st.slider("k1 — saturação de frequência", 0.0, 3.0, 1.2, 0.1,
                   help="0 = só presença/ausência do termo · alto = comportamento quase linear (TF-IDF)")
    b = st.slider("b — normalização por comprimento", 0.0, 1.0, 0.75, 0.05,
                  help="0 = sem penalizar documentos longos · 1 = normalização total por |D|/avgdl")
    st.header("🧠 Motor Semântico")
    modelo_nome = st.selectbox("Modelo de embeddings", MODELOS_EMBEDDING)
    st.header("🔀 Fusão RRF")
    alpha = st.slider("α — peso do BM25", 0.0, 1.0, 0.5, 0.05,
                      help="α = 1 → só BM25 · α = 0 → só semântico")
    k_rrf = st.number_input("k_RRF (suavização de posição)", 1, 200, 60, 1)
    st.header("⭐ Bônus")
    usar_cross = st.checkbox("Re-ranking com Cross-Encoder (Top-3)", value=False)
    cross_nome = st.selectbox("Modelo Cross-Encoder", MODELOS_CROSS, disabled=not usar_cross)

with st.expander("📚 Corpus médico indexado (6 diretrizes)"):
    st.dataframe(pd.DataFrame(CORPUS).rename(columns={"id": "ID", "titulo": "Título", "conteudo": "Conteúdo"}),
                 use_container_width=True, hide_index=True)

col_q1, col_q2 = st.columns([2, 1])
with col_q2:
    exemplo = st.selectbox("Consultas de teste:", ["(digitar a minha)"] + CONSULTAS_TESTE, index=1)
with col_q1:
    consulta = st.text_input("🔎 Consulta clínica:",
                             value="" if exemplo == "(digitar a minha)" else exemplo)

if not consulta.strip():
    st.info("Digite uma consulta ou escolha uma consulta de teste.")
    st.stop()

# ---- Execução dos motores (com medição de latência) ----
t0 = time.perf_counter()
bm25_scores, ranks_bm25, bm25, corpus_tokens, q_tokens = busca_bm25(consulta, k1, b)
t_bm25 = (time.perf_counter() - t0) * 1000

t0 = time.perf_counter()
sims, ranks_sem = busca_semantica(consulta, modelo_nome)
t_sem = (time.perf_counter() - t0) * 1000

t0 = time.perf_counter()
scores_rrf = rrf(ranks_bm25, ranks_sem, alpha, k_rrf)
t_rrf = (time.perf_counter() - t0) * 1000

ids = [d["id"] for d in CORPUS]
titulo = {d["id"]: d["titulo"] for d in CORPUS}
conteudo = {d["id"]: d["conteudo"] for d in CORPUS}
top_rrf = list(scores_rrf.keys())

m1, m2, m3, m4 = st.columns(4)
m1.metric("Top-1 BM25", min(ranks_bm25, key=ranks_bm25.get) if ranks_bm25 else "nenhum match")
m2.metric("Top-1 Semântico", min(ranks_sem, key=ranks_sem.get))
m3.metric("Top-1 Híbrido RRF", top_rrf[0])
m4.metric("Latência total", f"{t_bm25 + t_sem + t_rrf:.1f} ms",
          help=f"BM25 {t_bm25:.1f} ms · Semântico {t_sem:.1f} ms · RRF {t_rrf:.2f} ms")
st.caption(f"Tokens da consulta após pré-processamento: `{q_tokens}`")

abas = ["🔤 Léxico (BM25)", "🧠 Semântico", "🔀 Híbrido RRF", "📊 Matriz Comparativa"]
if usar_cross:
    abas.append("🎯 Cross-Encoder")
tabs = st.tabs(abas)

# ---------------------------- LÉXICO ---------------------------------------
with tabs[0]:
    st.subheader(f"Ranking BM25 (k1 = {k1}, b = {b})")
    st.latex(r"score(D,Q)=\sum_{q_i\in Q} IDF(q_i)\cdot\frac{f(q_i,D)\,(k_1+1)}{f(q_i,D)+k_1\left(1-b+b\frac{|D|}{avgdl}\right)}")
    df_lex = pd.DataFrame({
        "Rank": pd.array([ranks_bm25.get(d) for d in ids], dtype="Int64"), "Documento": ids,
        "Título": [titulo[d] for d in ids], "Score BM25": np.round(bm25_scores, 4),
        "|D| (tokens)": [len(t) for t in corpus_tokens],
    }).sort_values("Score BM25", ascending=False)
    st.dataframe(df_lex, use_container_width=True, hide_index=True)
    if not ranks_bm25:
        st.warning("Nenhum documento contém os termos da consulta: o BM25 não encontra sinônimos.")

    with st.expander("🔍 Cálculo termo a termo"):
        st.write(f"avgdl = {bm25.avgdl:.2f} tokens · N = {len(CORPUS)} documentos")
        linhas = []
        for i, d in enumerate(ids):
            dl = len(corpus_tokens[i])
            for t in q_tokens:
                f = corpus_tokens[i].count(t)
                idf_t = bm25.idf.get(t, 0.0)
                den = f + k1 * (1 - b + b * dl / bm25.avgdl)
                parcela = idf_t * (f * (k1 + 1)) / den if den else 0.0
                linhas.append({"Documento": d, "Termo": t, "f(t,D)": f, "|D|": dl,
                               "IDF": round(idf_t, 4), "Contribuição": round(parcela, 4)})
        st.dataframe(pd.DataFrame(linhas), use_container_width=True, hide_index=True)
        st.caption("A biblioteca rank_bm25 usa IDF = ln((N − df + 0,5)/(df + 0,5)) com piso ε para termos muito comuns.")

# ---------------------------- SEMÂNTICO -------------------------------------
with tabs[1]:
    st.subheader(f"Ranking por similaridade de cosseno ({modelo_nome})")
    st.latex(r"\cos(\vec q,\vec d)=\frac{\vec q\cdot\vec d}{\lVert\vec q\rVert\,\lVert\vec d\rVert}")
    df_sem = pd.DataFrame({
        "Rank": [ranks_sem[d] for d in ids], "Documento": ids, "Título": [titulo[d] for d in ids],
        "Cosseno": np.round(sims, 4),
    }).sort_values("Rank")
    st.dataframe(df_sem, use_container_width=True, hide_index=True)
    st.bar_chart(df_sem.set_index("Documento")["Cosseno"])
    if modelo_nome in PREFIXOS:
        st.caption("O E5 produz cossenos concentrados numa faixa alta (≈ 0,8–0,9); o que importa é a ordem.")

# ---------------------------- HÍBRIDO ---------------------------------------
with tabs[2]:
    st.subheader(f"Reciprocal Rank Fusion (α = {alpha}, k = {k_rrf})")
    st.latex(r"Score_{RRF}(D)=\alpha\frac{1}{k+Rank_{BM25}}+(1-\alpha)\frac{1}{k+Rank_{Sem}}")
    df_rrf = pd.DataFrame([{
        "Posição": pos + 1, "Documento": d, "Título": titulo[d],
        "Rank BM25": ranks_bm25.get(d), "Rank Semântico": ranks_sem[d],
        "Parcela BM25": round(alpha / (k_rrf + ranks_bm25[d]), 5) if d in ranks_bm25 else 0.0,
        "Parcela Semântica": round((1 - alpha) / (k_rrf + ranks_sem[d]), 5),
        "Score RRF": round(s, 5),
    } for pos, (d, s) in enumerate(scores_rrf.items())])
    df_rrf["Rank BM25"] = df_rrf["Rank BM25"].astype("Int64")
    st.dataframe(df_rrf, use_container_width=True, hide_index=True)
    st.caption("Rank BM25 vazio = o documento não tem nenhum termo da consulta.")
    for d in top_rrf[:3]:
        st.markdown(f"**{d} — {titulo[d]}**  \n{conteudo[d]}")
    st.caption("O RRF usa apenas a posição de cada documento, por isso não precisa normalizar "
               "as escalas diferentes do BM25 (0 a ~3) e do cosseno (−1 a 1).")

# ---------------------------- MATRIZ ----------------------------------------
with tabs[3]:
    st.subheader("Posição de cada documento nos três rankings")
    rank_hib = {d: i + 1 for i, d in enumerate(top_rrf)}
    df_mat = pd.DataFrame({
        "Documento": ids, "Título": [titulo[d] for d in ids],
        "Rank BM25": pd.array([ranks_bm25.get(d) for d in ids], dtype="Int64"),
        "Rank Semântico": [ranks_sem[d] for d in ids],
        "Rank Híbrido RRF": [rank_hib[d] for d in ids],
    }).sort_values("Rank Híbrido RRF")
    st.dataframe(df_mat, use_container_width=True, hide_index=True)
    st.caption("Rank BM25 vazio = sem nenhum termo em comum com a consulta.")

    # Gráfico: quanto maior a barra, melhor a posição (7 − rank; sem match = 0)
    graf = pd.DataFrame({
        "BM25": [7 - ranks_bm25[d] if d in ranks_bm25 else 0 for d in ids],
        "Semântico": [7 - ranks_sem[d] for d in ids],
        "Híbrido RRF": [7 - rank_hib[d] for d in ids],
    }, index=ids)
    st.markdown("**Força de posição por motor** (barra = 7 − rank; 0 = não recuperado)")
    st.bar_chart(graf, stack=False)

    st.info(
        "💡 Com **CÓD-ECG-12D**, o BM25 acha exatamente Doc 1 e Doc 6. Com **ataque cardíaco**, "
        "o BM25 não encontra nada e o semântico traz as diretrizes cardíacas. O RRF combina os "
        "dois: documentos bem colocados nas duas listas sobem para o topo."
    )

# ---------------------------- CROSS-ENCODER ---------------------------------
if usar_cross:
    with tabs[4]:
        st.subheader("Re-ranking do Top-3 híbrido com Cross-Encoder")
        st.markdown(
            f"O Cross-Encoder `{cross_nome}` lê o par **(consulta, documento)** junto e dá uma nota "
            "de relevância mais precisa que a do bi-encoder. É mais lento, por isso só reordena os 3 "
            "melhores candidatos do RRF."
        )
        ce = carregar_cross_encoder(cross_nome)
        top3 = top_rrf[:3]
        t0 = time.perf_counter()
        notas = ce.predict([[consulta, texto_completo(next(x for x in CORPUS if x["id"] == d))] for d in top3])
        t_ce = (time.perf_counter() - t0) * 1000
        nova = sorted(zip(top3, notas), key=lambda x: x[1], reverse=True)
        df_ce = pd.DataFrame([{
            "Nova posição": i + 1, "Documento": d, "Título": titulo[d],
            "Posição no RRF": top3.index(d) + 1,
            "Variação": f"{(top3.index(d) + 1) - (i + 1):+d}",
            "Score RRF": round(scores_rrf[d], 5), "Nota Cross-Encoder": round(float(n), 4),
        } for i, (d, n) in enumerate(nova)])
        st.dataframe(df_ce, use_container_width=True, hide_index=True)
        st.caption(f"Latência do Cross-Encoder: {t_ce:.1f} ms · Variação positiva = o documento subiu.")
        if cross_nome.endswith("ms-marco-MiniLM-L-6-v2"):
            st.warning("Este modelo foi treinado em inglês; em consultas em português as notas "
                       "tendem a ser menos confiáveis que as do modelo multilíngue mMARCO.")
