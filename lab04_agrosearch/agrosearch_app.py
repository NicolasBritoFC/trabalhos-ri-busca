# =============================================================================
# AgroSearch — Motor de Busca Inteligente (Laboratório Prático 04)
# Disciplina: Tópicos Avançados — Recuperação de Informação / PLN
# Prof. Me. Ricardo Roberto de Lima — UNIPÊ
#
# Restrição do desafio: nada de scikit-learn / TfidfVectorizer. O pré-processamento,
# o índice invertido, o TF-IDF e a similaridade de cosseno (bônus) são
# implementados do zero, usando apenas a biblioteca padrão do Python.
#
# Execução:  streamlit run agrosearch_app.py
# =============================================================================

import math
import re
import unicodedata
from collections import Counter, defaultdict

import pandas as pd
import streamlit as st

# -----------------------------------------------------------------------------
# BASE DE DOCUMENTOS (hardcode sugerido no enunciado)
# -----------------------------------------------------------------------------
DOCUMENTOS_PADRAO = {
    "Doc 1": "A soja requer irrigação constante durante o período de floração para garantir a produtividade.",
    "Doc 2": "O controle biológico de lagartas na soja pode ser feito com a vespa Trichogramma.",
    "Doc 3": "A adubação verde com leguminosas melhora o nitrogênio no solo para o milho.",
    "Doc 4": "Lagartas desfolhadoras causam grande prejuízo na cultura da soja e do algodão.",
    "Doc 5": "A irrigação por gotejamento economiza água e é ideal para o cultivo orgânico.",
}

# Lista de stopwords do português (escrita já sem acentos, pois a remoção de
# stopwords acontece depois da normalização).
STOPWORDS = {
    "a", "o", "as", "os", "um", "uma", "uns", "umas", "de", "do", "da", "dos", "das",
    "em", "no", "na", "nos", "nas", "por", "pelo", "pela", "pelos", "pelas", "para",
    "pra", "com", "sem", "sob", "sobre", "entre", "ate", "e", "ou", "mas", "que",
    "se", "ao", "aos", "a", "as", "ser", "e", "sao", "foi", "era", "pode", "podem",
    "ja", "nao", "muito", "mais", "menos", "seu", "sua", "seus", "suas", "este",
    "esta", "esse", "essa", "isso", "isto", "como", "quando", "onde", "qual",
    "durante", "tambem", "lhe", "eles", "elas", "ele", "ela", "nosso", "nossa",
}

# Sufixos do português, do mais longo para o mais curto. O stemmer remove o
# primeiro sufixo que casar, desde que sobre um radical com 3+ letras.
SUFIXOS = [
    "amentos", "imentos", "adoras", "adores", "amento", "imento", "idades",
    "mente", "acoes", "icoes", "adora", "ador", "acao", "icao", "idade", "ismos",
    "ismo", "istas", "ista", "aveis", "iveis", "avel", "ivel", "ancia", "encia",
    "ando", "endo", "indo", "aram", "eram", "iram", "ados", "idos", "adas", "idas",
    "ado", "ido", "ada", "ida", "oso", "osa", "am", "em", "ar", "er", "ir",
    "as", "es", "os", "is", "a", "e", "o", "s",
]


# -----------------------------------------------------------------------------
# FASE 1 — PIPELINE DE PRÉ-PROCESSAMENTO (4 etapas)
# -----------------------------------------------------------------------------
def tokenizar(texto: str) -> list[str]:
    """Etapa 1: separa o texto em tokens (sequências de letras/números)."""
    return re.findall(r"\w+", texto)


def normalizar(token: str) -> str:
    """Etapa 2: minúsculas + remoção de acentos (NFD e descarte das marcas)."""
    token = token.lower()
    token = unicodedata.normalize("NFD", token)
    return "".join(c for c in token if unicodedata.category(c) != "Mn")


def remover_stopwords(tokens: list[str]) -> list[str]:
    """Etapa 3: remove palavras de alta frequência e baixo poder discriminativo."""
    return [t for t in tokens if t not in STOPWORDS]


def stem(palavra: str) -> str:
    """Etapa 4: stemmer por regras (remoção de sufixos) implementado do zero."""
    for suf in SUFIXOS:
        if palavra.endswith(suf) and len(palavra) - len(suf) >= 3:
            return palavra[: -len(suf)]
    return palavra


def preprocessar(texto: str, usar_stopwords: bool, usar_stemming: bool) -> dict:
    """Executa o pipeline e devolve o resultado de cada etapa (para exibição)."""
    etapa1 = tokenizar(texto)
    etapa2 = [normalizar(t) for t in etapa1]
    etapa3 = remover_stopwords(etapa2) if usar_stopwords else etapa2
    etapa4 = [stem(t) for t in etapa3] if usar_stemming else etapa3
    return {"tokenizacao": etapa1, "normalizacao": etapa2, "stopwords": etapa3, "final": etapa4}


# -----------------------------------------------------------------------------
# FASE 2 — ÍNDICE INVERTIDO (Termo -> [IDs de Docs])
# -----------------------------------------------------------------------------
def construir_indice_invertido(docs_tokens: dict[str, list[str]]) -> dict[str, list[str]]:
    indice = defaultdict(list)
    for doc_id, tokens in docs_tokens.items():
        for termo in tokens:
            if doc_id not in indice[termo]:
                indice[termo].append(doc_id)
    return dict(sorted(indice.items()))


# -----------------------------------------------------------------------------
# FASE 3 — TF, IDF e TF-IDF (do zero)
# -----------------------------------------------------------------------------
def tf(termo: str, tokens: list[str]) -> float:
    """TF(t, d) = ocorrências de t em d / total de termos em d."""
    return tokens.count(termo) / len(tokens) if tokens else 0.0


def idf(termo: str, indice: dict[str, list[str]], n_docs: int) -> float:
    """IDF(t) = log10(N / df(t)). Termo fora do vocabulário recebe 0."""
    df_t = len(indice.get(termo, []))
    return math.log10(n_docs / df_t) if df_t else 0.0


def ranquear_tfidf(consulta_tokens, docs_tokens, indice):
    """Soma o TF-IDF de cada termo da consulta em cada documento (TF-IDF acumulado)."""
    n = len(docs_tokens)
    termos = list(dict.fromkeys(consulta_tokens))  # termos únicos, mantendo a ordem
    detalhes, ranking = [], []
    for doc_id, tokens in docs_tokens.items():
        total = 0.0
        for t in termos:
            tf_v, idf_v = tf(t, tokens), idf(t, indice, n)
            detalhes.append({
                "Documento": doc_id, "Termo": t,
                "f(t,d)": tokens.count(t), "|d|": len(tokens),
                "TF": round(tf_v, 4), "df(t)": len(indice.get(t, [])),
                "IDF": round(idf_v, 4), "TF-IDF": round(tf_v * idf_v, 4),
            })
            total += tf_v * idf_v
        ranking.append({"Documento": doc_id, "TF-IDF acumulado": round(total, 4)})
    df_rank = pd.DataFrame(ranking).sort_values("TF-IDF acumulado", ascending=False).reset_index(drop=True)
    df_rank.index += 1
    return df_rank, pd.DataFrame(detalhes)


# -----------------------------------------------------------------------------
# BÔNUS — SIMILARIDADE DE COSSENO entre vetor TF-IDF da query e dos documentos
# -----------------------------------------------------------------------------
def vetor_tfidf(tokens, vocabulario, indice, n_docs):
    return [tf(t, tokens) * idf(t, indice, n_docs) for t in vocabulario]


def cosseno(u, v):
    dot = sum(a * b for a, b in zip(u, v))
    nu, nv = math.sqrt(sum(a * a for a in u)), math.sqrt(sum(b * b for b in v))
    return dot / (nu * nv) if nu and nv else 0.0


# =============================================================================
# INTERFACE STREAMLIT
# =============================================================================
st.set_page_config(page_title="AgroSearch", page_icon="🌱", layout="wide")
st.title("🌱 AgroSearch — Motor de Busca Inteligente")
st.caption("Pré-processamento → Índice Invertido → Ranqueamento TF-IDF (implementação from scratch, sem scikit-learn)")

with st.sidebar:
    st.header("⚙️ Pré-processamento")
    usar_stopwords = st.checkbox("Remover stopwords", value=True)
    usar_stemming = st.checkbox("Aplicar stemming", value=True)
    st.markdown("---")
    st.header("📄 Base de documentos")
    texto_docs = st.text_area(
        "Um documento por linha:",
        value="\n".join(DOCUMENTOS_PADRAO.values()),
        height=260,
    )

linhas = [l.strip() for l in texto_docs.split("\n") if l.strip()]
documentos = {f"Doc {i + 1}": l for i, l in enumerate(linhas)}
if not documentos:
    st.warning("Adicione pelo menos um documento na barra lateral.")
    st.stop()

processados = {d: preprocessar(t, usar_stopwords, usar_stemming) for d, t in documentos.items()}
docs_tokens = {d: p["final"] for d, p in processados.items()}
indice = construir_indice_invertido(docs_tokens)
vocabulario = list(indice.keys())

c1, c2, c3 = st.columns(3)
c1.metric("Documentos (N)", len(documentos))
c2.metric("Tamanho do vocabulário", len(vocabulario))
c3.metric("Total de tokens indexados", sum(len(t) for t in docs_tokens.values()))

aba1, aba2, aba3, aba4 = st.tabs([
    "1️⃣ Pré-processamento", "2️⃣ Índice Invertido", "3️⃣ Busca TF-IDF", "⭐ Bônus: Cosseno",
])

# ---------------------------- ABA 1 ----------------------------------------
with aba1:
    st.subheader("Pipeline de pré-processamento, etapa por etapa")
    st.markdown("Ligue/desligue **stopwords** e **stemming** na barra lateral e veja o vocabulário mudar.")
    doc_sel = st.selectbox("Documento para inspecionar:", list(documentos.keys()))
    p = processados[doc_sel]
    st.info(f"**Texto original:** {documentos[doc_sel]}")
    with st.expander("1. Tokenização", expanded=True):
        st.write(p["tokenizacao"])
    with st.expander("2. Normalização (minúsculas e sem acentos)", expanded=True):
        st.write(p["normalizacao"])
    with st.expander("3. Remoção de stopwords " + ("(ativa)" if usar_stopwords else "(desligada)"), expanded=True):
        st.write(p["stopwords"])
    with st.expander("4. Stemming " + ("(ativo)" if usar_stemming else "(desligado)"), expanded=True):
        st.write(p["final"])

    st.markdown("#### Tokens finais de todos os documentos")
    st.dataframe(
        pd.DataFrame([{"Documento": d, "Nº tokens": len(t), "Tokens": " · ".join(t)} for d, t in docs_tokens.items()]),
        use_container_width=True, hide_index=True,
    )

# ---------------------------- ABA 2 ----------------------------------------
with aba2:
    st.subheader("Índice Invertido (Termo → Documentos)")
    df_indice = pd.DataFrame([
        {"Termo": t, "df(t)": len(ds), "Documentos (postings list)": ", ".join(ds)}
        for t, ds in indice.items()
    ]).sort_values(["df(t)", "Termo"], ascending=[False, True])
    st.dataframe(df_indice, use_container_width=True, hide_index=True)
    with st.expander("Ver como JSON"):
        st.json(indice)
    st.caption("Buscar um termo aqui é um acesso direto ao dicionário, sem varrer todos os documentos.")

# ---------------------------- ABA 3 ----------------------------------------
with aba3:
    st.subheader("Busca e ranqueamento por TF-IDF")
    st.latex(r"TF(t,d)=\frac{f(t,d)}{|d|}\qquad IDF(t)=\log_{10}\frac{N}{df(t)}\qquad score(d)=\sum_{t\in q} TF(t,d)\cdot IDF(t)")
    consulta = st.text_input("Digite a consulta:", value="lagartas na soja")
    q = preprocessar(consulta, usar_stopwords, usar_stemming)["final"]
    st.write("**Termos da consulta após o pipeline:**", q)

    fora = [t for t in q if t not in indice]
    if fora:
        st.warning(f"Termos fora do vocabulário (IDF = 0, não pontuam): {', '.join(fora)}")

    if q:
        df_rank, df_det = ranquear_tfidf(q, docs_tokens, indice)
        vencedor = df_rank.iloc[0]
        if vencedor["TF-IDF acumulado"] > 0:
            st.success(f"🏆 Documento vencedor: **{vencedor['Documento']}** "
                       f"(TF-IDF = {vencedor['TF-IDF acumulado']:.4f}) — {documentos[vencedor['Documento']]}")
        else:
            st.error("Nenhum documento contém os termos da consulta.")

        def destacar(linha):
            cor = "background-color: rgba(46, 160, 67, 0.25)" if linha.name == 1 and linha["TF-IDF acumulado"] > 0 else ""
            return [cor] * len(linha)

        st.markdown("#### Ranking (maior → menor TF-IDF acumulado)")
        df_rank_show = df_rank.copy()
        df_rank_show["Texto"] = df_rank_show["Documento"].map(documentos)
        st.dataframe(df_rank_show.style.apply(destacar, axis=1), use_container_width=True)
        st.bar_chart(df_rank.set_index("Documento")["TF-IDF acumulado"])

        with st.expander("🔍 Cálculo detalhado (TF, IDF e TF-IDF por termo e documento)"):
            st.dataframe(df_det, use_container_width=True, hide_index=True)
    else:
        st.info("A consulta ficou vazia depois do pré-processamento.")

# ---------------------------- ABA 4 ----------------------------------------
with aba4:
    st.subheader("Bônus: Similaridade de Cosseno (vetor TF-IDF da query × documentos)")
    st.markdown(
        "A soma de TF-IDF favorece documentos que contêm **um** termo muito raro. "
        "O cosseno compara o **vetor inteiro** da consulta com o de cada documento, "
        "o que funciona melhor em consultas com várias palavras."
    )
    st.latex(r"\cos(\vec q,\vec d)=\frac{\vec q\cdot\vec d}{\lVert\vec q\rVert\,\lVert\vec d\rVert}")
    q2_txt = st.text_input("Consulta (várias palavras):", value="irrigação da soja")
    q2 = preprocessar(q2_txt, usar_stopwords, usar_stemming)["final"]
    n = len(docs_tokens)
    vq = vetor_tfidf(q2, vocabulario, indice, n)
    linhas_cos = []
    for d, toks in docs_tokens.items():
        vd = vetor_tfidf(toks, vocabulario, indice, n)
        soma = sum(tf(t, toks) * idf(t, indice, n) for t in set(q2))
        linhas_cos.append({"Documento": d, "Cosseno": round(cosseno(vq, vd), 4),
                           "TF-IDF acumulado": round(soma, 4), "Texto": documentos[d]})
    df_cos = pd.DataFrame(linhas_cos).sort_values("Cosseno", ascending=False).reset_index(drop=True)
    df_cos.index += 1
    st.dataframe(df_cos, use_container_width=True)
    st.caption(f"Termos da consulta: {q2} · Vetores com {len(vocabulario)} dimensões (tamanho do vocabulário).")
