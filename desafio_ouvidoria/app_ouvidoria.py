# =============================================================================
# Ouvidoria Inteligente — Triagem Semântica de Manifestações Cidadãs
# Entrega 4: Buscador Semântico + App Streamlit
#
# Abas: 🔍 Busca Semântica · 📋 Base Completa · 🌐 Espaço Vetorial · 🧩 Chunking
#       (+ 🔁 Duplicatas, reaproveitando a função da Entrega 2)
#
# Execução:  streamlit run app_ouvidoria.py
# =============================================================================

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
DADOS = Path(__file__).parent / "data" / "manifestacoes.json"

MODELOS = {
    "paraphrase-multilingual-MiniLM-L12-v2": "Multilíngue (recomendado para PT-BR)",
    "sentence-transformers/all-MiniLM-L6-v2": "Inglês (comparação)",
}
CORES_CATEGORIA = {
    "infraestrutura": "#4C72B0", "saúde": "#C44E52", "segurança": "#8172B2",
    "educação": "#DD8452", "meio ambiente": "#55A868",
}


# -----------------------------------------------------------------------------
# CACHE: modelo com cache_resource, dados/embeddings com cache_data
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Carregando modelo de embeddings...")
def carregar_modelo(nome: str) -> SentenceTransformer:
    return SentenceTransformer(nome)


@st.cache_data
def carregar_base() -> pd.DataFrame:
    with open(DADOS, encoding="utf-8") as f:
        return pd.DataFrame(json.load(f))


@st.cache_data(show_spinner="Gerando embeddings das manifestações...")
def embeddings_base(nome_modelo: str, textos: tuple[str, ...]) -> np.ndarray:
    return carregar_modelo(nome_modelo).encode(list(textos), normalize_embeddings=True)


@st.cache_data(show_spinner="Projetando em 2D...")
def projetar(emb: np.ndarray, metodo: str) -> np.ndarray:
    if metodo == "PCA":
        return PCA(n_components=2, random_state=42).fit_transform(emb)
    perp = max(2, min(10, len(emb) - 1))
    return TSNE(n_components=2, perplexity=perp, random_state=42).fit_transform(emb)


def semaforo(score: float) -> str:
    return "🟢" if score > 0.7 else "🟡" if score > 0.5 else "🔴"


def detectar_duplicatas(emb: np.ndarray, ids: list[str], limiar: float) -> list[tuple]:
    sim = emb @ emb.T
    return sorted(
        [(ids[i], ids[j], float(sim[i, j])) for i in range(len(ids)) for j in range(i + 1, len(ids))
         if sim[i, j] >= limiar],
        key=lambda p: p[2], reverse=True,
    )


# =============================================================================
# INTERFACE
# =============================================================================
st.set_page_config(page_title="Ouvidoria Inteligente", page_icon="🏛️", layout="wide")
st.title("🏛️ Ouvidoria Inteligente")
st.caption("Triagem semântica das manifestações cidadãs · busca por significado, não só por palavra-chave")

with st.sidebar:
    st.header("⚙️ Configurações")
    modelo_nome = st.selectbox("Modelo de embedding", list(MODELOS), format_func=lambda m: f"{m.split('/')[-1]} — {MODELOS[m]}")
    top_k = st.slider("Top-K resultados", 1, 10, 5)
    st.markdown("---")
    st.markdown("**Legenda de similaridade**  \n🟢 > 0,70 · 🟡 > 0,50 · 🔴 demais")

df = carregar_base()
emb = embeddings_base(modelo_nome, tuple(df["texto"]))
modelo = carregar_modelo(modelo_nome)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Manifestações", len(df))
c2.metric("Categorias", df["categoria_oficial"].nunique())
c3.metric("Textos longos (> 500 caracteres)", int((df["texto"].str.len() > 500).sum()))
c4.metric("Dimensão dos embeddings", emb.shape[1])

aba_busca, aba_base, aba_espaco, aba_chunk, aba_dup = st.tabs(
    ["🔍 Busca Semântica", "📋 Base Completa", "🌐 Espaço Vetorial", "🧩 Chunking", "🔁 Duplicatas"]
)

# ---------------------------- BUSCA SEMÂNTICA --------------------------------
with aba_busca:
    st.subheader("Descreva o problema com as suas palavras")
    exemplos = ["rua cheia de buracos, meu carro estragou", "não tem doutor no postinho",
                "a praça está escura à noite", "cheiro horrível de lixo perto de casa"]
    exemplo = st.radio("Exemplos:", exemplos, horizontal=True)
    consulta = st.text_area("Manifestação do cidadão:", value=exemplo, height=90)

    if consulta.strip():
        q = modelo.encode([consulta], normalize_embeddings=True)[0]
        sims = emb @ q
        top = np.argsort(sims)[::-1][:top_k]

        if sims[top[0]] >= 0.65:
            st.warning(f"⚠️ Possível duplicata de **{df['id'][top[0]]}** (similaridade {sims[top[0]]:.2f}). "
                       "Limiar 0,65 definido na análise da Entrega 2.")
        # Categoria sugerida: voto ponderado pela similaridade entre os Top-K
        votos = pd.Series(sims[top], index=df["categoria_oficial"][top]).groupby(level=0).sum()
        st.info(f"🏷️ Categoria sugerida para triagem: **{votos.idxmax()}**")

        st.markdown(f"#### {top_k} manifestações mais similares")
        for pos, i in enumerate(top, 1):
            s = float(sims[i])
            with st.container(border=True):
                a, b = st.columns([5, 1])
                a.markdown(f"{semaforo(s)} **#{pos} · {df['id'][i]}** · `{df['categoria_oficial'][i]}` · {df['data'][i]}")
                b.metric("Score", f"{s:.3f}", label_visibility="collapsed")
                st.write(df["texto"][i])

# ---------------------------- BASE COMPLETA ----------------------------------
with aba_base:
    st.subheader("Todas as manifestações")
    filtro = st.multiselect("Filtrar por categoria:", sorted(df["categoria_oficial"].unique()))
    vis = df[df["categoria_oficial"].isin(filtro)] if filtro else df
    st.dataframe(vis.assign(caracteres=vis["texto"].str.len()), use_container_width=True, hide_index=True,
                 column_config={"texto": st.column_config.TextColumn("texto", width="large")})

    if st.button("🔢 Gerar matriz de similaridade"):
        sim = emb @ emb.T
        fig = px.imshow(sim, x=df["id"], y=df["id"], zmin=0, zmax=1, color_continuous_scale="YlOrRd",
                        labels={"color": "cosseno"}, aspect="equal")
        fig.update_layout(height=750, title="Similaridade de cosseno entre as 40 manifestações")
        st.plotly_chart(fig, use_container_width=True)
        iu = np.triu_indices(len(df), 1)
        st.caption(f"Média entre pares: {sim[iu].mean():.3f} · percentil 99: {np.percentile(sim[iu], 99):.3f}")

# ---------------------------- ESPAÇO VETORIAL --------------------------------
with aba_espaco:
    st.subheader("Mapa semântico das manifestações")
    metodo = st.radio("Redução de dimensionalidade:", ["PCA", "t-SNE"], horizontal=True)
    coords = projetar(emb, metodo)
    plot = df.assign(x=coords[:, 0], y=coords[:, 1], resumo=df["texto"].str[:90] + "…")
    fig = px.scatter(plot, x="x", y="y", color="categoria_oficial", text="id",
                     color_discrete_map=CORES_CATEGORIA, hover_data={"resumo": True, "x": False, "y": False},
                     labels={"categoria_oficial": "Categoria oficial"})
    fig.update_traces(marker=dict(size=13, line=dict(width=1, color="black")), textposition="top center",
                      textfont_size=9)
    fig.update_layout(height=600, xaxis_title=f"{metodo} 1", yaxis_title=f"{metodo} 2")
    st.plotly_chart(fig, use_container_width=True)

    # Quanto cada categoria "se mantém unida": vizinho mais próximo é da mesma categoria?
    sim = emb @ emb.T
    np.fill_diagonal(sim, -1)
    vizinho = sim.argmax(axis=1)
    acerto = (df["categoria_oficial"].values == df["categoria_oficial"].values[vizinho])
    por_cat = pd.Series(acerto, index=df["categoria_oficial"]).groupby(level=0).mean()
    st.markdown(f"**Vizinho mais próximo da mesma categoria:** {acerto.mean():.0%} das manifestações")
    st.dataframe(por_cat.rename("% com vizinho da mesma categoria").map("{:.0%}".format).to_frame(),
                 use_container_width=True)
    st.markdown(
        "**Os clusters coincidem com as categorias?** Em parte. Com o modelo multilíngue, 70% das "
        "manifestações têm como vizinho mais próximo uma da mesma categoria. **Saúde** é a mais coesa: "
        "todas as manifestações de saúde têm vizinho de saúde. As fronteiras borram onde os temas se cruzam "
        "de verdade: **infraestrutura e meio ambiente** se misturam (bueiro entupido × terreno com lixo, "
        "ponte sobre o riacho × entulho no rio); iluminação pública fica perto de **segurança**, porque o "
        "cidadão cita a rua escura como risco de assalto; e a escola sem acessibilidade fica perto da "
        "denúncia de drogas na praça ao lado da escola. Os embeddings agrupam pelo **assunto do texto**, "
        "que nem sempre é a categoria administrativa. Por isso a categoria sugerida na busca é só uma "
        "sugestão para o atendente."
    )

# ---------------------------- CHUNKING ---------------------------------------
with aba_chunk:
    st.subheader("Divida uma manifestação longa em chunks")
    longas = df[df["texto"].str.len() > 500]
    base_txt = st.selectbox("Carregar uma manifestação longa da base:", longas["id"],
                            format_func=lambda i: f"{i} — {df.set_index('id').loc[i, 'categoria_oficial']}")
    texto = st.text_area("Texto (cole aqui outra manifestação, se quiser):",
                         value=df.set_index("id").loc[base_txt, "texto"], height=180)
    a, b, c = st.columns(3)
    estrategia = a.selectbox("Estratégia", ["RecursiveCharacter", "Fixed-size (Character)"])
    chunk_size = b.slider("chunk_size (caracteres)", 50, 600, 300, 10)
    chunk_overlap = c.slider("chunk_overlap (caracteres)", 0, 200, 60, 10)

    if chunk_overlap >= chunk_size:
        st.error("O overlap precisa ser menor que o chunk_size.")
    elif texto.strip():
        if estrategia == "RecursiveCharacter":
            splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                                                      separators=["\n\n", "\n", ". ", ", ", " ", ""],
                                                      keep_separator="end")
        else:
            splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separator=" ")
        chunks = splitter.split_text(texto)
        E = modelo.encode(chunks, normalize_embeddings=True)
        e_full = modelo.encode([texto], normalize_embeddings=True)[0]

        m1, m2, m3 = st.columns(3)
        m1.metric("Chunks gerados", len(chunks))
        m2.metric("Tamanho médio", f"{np.mean([len(x) for x in chunks]):.0f} car.")
        coesao = np.mean([E[k] @ E[k + 1] for k in range(len(E) - 1)]) if len(E) > 1 else 1.0
        m3.metric("Coesão entre chunks vizinhos", f"{coesao:.3f}")

        for k, ch in enumerate(chunks):
            with st.expander(f"Chunk {k + 1} · {len(ch)} caracteres · cosseno com o texto completo = {E[k] @ e_full:.3f}"):
                st.write(ch)
                st.caption(f"Embedding ({E.shape[1]}D), primeiros valores: {np.round(E[k][:8], 3).tolist()} …")

        if len(chunks) > 1:
            st.markdown("**Similaridade entre os chunks**")
            fig = px.imshow(E @ E.T, x=[f"C{i + 1}" for i in range(len(chunks))],
                            y=[f"C{i + 1}" for i in range(len(chunks))], zmin=0, zmax=1,
                            color_continuous_scale="Blues", text_auto=".2f")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Busque dentro dos chunks**")
        q_chunk = st.text_input("Pergunta sobre o texto:", value="o que o cidadão está pedindo?")
        if q_chunk:
            s = E @ modelo.encode([q_chunk], normalize_embeddings=True)[0]
            k = int(np.argmax(s))
            st.success(f"Chunk {k + 1} ({s[k]:.3f}): {chunks[k]}")

# ---------------------------- DUPLICATAS -------------------------------------
with aba_dup:
    st.subheader("Detecção de duplicatas semânticas")
    limiar = st.slider("Limiar de similaridade", 0.50, 0.95, 0.65, 0.01,
                       help="0,65 teve o melhor F1 na Entrega 2 e coincide com o percentil 99 da distribuição.")
    pares = detectar_duplicatas(emb, df["id"].tolist(), limiar)
    st.write(f"**{len(pares)} pares** acima de {limiar:.2f}. Sugestões para um atendente confirmar:")
    texto_id = dict(zip(df["id"], df["texto"]))
    for a_id, b_id, s in pares:
        with st.container(border=True):
            st.markdown(f"{semaforo(s)} **{a_id} × {b_id}** · similaridade {s:.3f}")
            x, y = st.columns(2)
            x.caption(texto_id[a_id])
            y.caption(texto_id[b_id])
