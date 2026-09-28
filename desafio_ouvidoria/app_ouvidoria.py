"""
Ouvidoria Inteligente · Entrega 4 · buscador semântico para a triagem de manifestações.

Cada aba é uma função (aba_busca, aba_base, aba_espaco, aba_chunking, aba_duplicatas) que
recebe o contexto já carregado. A lógica compartilhada com os notebooks está em
nucleo_ouvidoria.py.

Rodar:  streamlit run app_ouvidoria.py
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import nucleo_ouvidoria as nucleo


@dataclass
class Contexto:
    base: pd.DataFrame
    vetores: np.ndarray
    vetorizador: nucleo.Vetorizador
    top_k: int


@st.cache_resource(show_spinner="Carregando o modelo de embeddings…")
def obter_vetorizador(nome: str) -> nucleo.Vetorizador:
    return nucleo.Vetorizador(nome)


@st.cache_data(show_spinner="Vetorizando as manifestações…")
def vetorizar_base(nome: str, textos: tuple[str, ...]) -> np.ndarray:
    return obter_vetorizador(nome)(textos)


@st.cache_data(show_spinner="Projetando em 2D…")
def projetar_2d(vetores: np.ndarray, metodo: str) -> np.ndarray:
    if metodo == "PCA":
        return PCA(n_components=2, random_state=42).fit_transform(vetores)
    return TSNE(n_components=2, perplexity=max(2, min(10, len(vetores) - 1)), random_state=42).fit_transform(vetores)


# ─────────────────────────────────── abas ───────────────────────────────────
def aba_busca(ctx: Contexto) -> None:
    st.markdown("#### Qual é o problema?")
    exemplos = ["rua cheia de buracos, meu carro estragou", "não tem doutor no postinho",
                "a praça está escura à noite", "cheiro horrível de lixo perto de casa"]
    exemplo = st.radio("Exemplos de manifestação", exemplos, horizontal=True)
    relato = st.text_area("Descreva com as suas palavras", value=exemplo, height=80)
    if not relato.strip():
        return

    sims = ctx.vetores @ ctx.vetorizador.um(relato)
    ordem = np.argsort(-sims)[: ctx.top_k]
    achados = ctx.base.iloc[ordem].assign(score=sims[ordem])
    achados.insert(0, "faixa", achados["score"].map(nucleo.faixa_de_cor))

    alerta, categoria = st.columns(2)
    melhor = achados.iloc[0]
    if melhor.score >= nucleo.LIMIAR_DUPLICATA:
        alerta.warning(f"Possível duplicata de **{melhor.id}** ({melhor.score:.2f}). "
                       f"Limiar {nucleo.LIMIAR_DUPLICATA} definido na Entrega 2.")
    else:
        alerta.success("Nenhuma manifestação muito parecida: provavelmente é um caso novo.")
    votos = achados.groupby("categoria_oficial")["score"].sum()
    categoria.info(f"Categoria sugerida para a triagem: **{votos.idxmax()}**")

    st.dataframe(
        achados[["faixa", "id", "score", "categoria_oficial", "data", "texto"]],
        hide_index=True, use_container_width=True,
        column_config={
            "faixa": st.column_config.TextColumn("", width="small"),
            "score": st.column_config.ProgressColumn("similaridade", min_value=0, max_value=1, format="%.3f"),
            "categoria_oficial": "categoria",
            "texto": st.column_config.TextColumn("manifestação", width="large"),
        },
    )
    st.caption("🟢 > 0,70 · 🟡 > 0,50 · 🔴 demais")


def aba_base(ctx: Contexto) -> None:
    filtro = st.multiselect("Categorias", sorted(ctx.base["categoria_oficial"].unique()),
                            placeholder="todas as categorias")
    tabela = ctx.base if not filtro else ctx.base[ctx.base["categoria_oficial"].isin(filtro)]
    st.dataframe(tabela.assign(caracteres=tabela["texto"].str.len()), hide_index=True, use_container_width=True,
                 column_config={"texto": st.column_config.TextColumn(width="large")})

    if st.button("Gerar matriz de similaridade", type="primary"):
        S = nucleo.similaridade(ctx.vetores)
        fig, ax = plt.subplots(figsize=(11, 9))
        sns.heatmap(pd.DataFrame(S, index=ctx.base["id"], columns=ctx.base["id"]), cmap="rocket_r", vmin=0,
                    vmax=1, square=True, ax=ax, cbar_kws={"label": "cosseno"}, xticklabels=True, yticklabels=True)
        ax.tick_params(labelsize=6)
        ax.set_title(f"Similaridade entre as {len(ctx.base)} manifestações")
        st.pyplot(fig)
        pares = S[np.triu_indices(len(S), 1)]
        st.caption(f"Média entre pares: {pares.mean():.3f} · percentil 99: {np.percentile(pares, 99):.3f}")


def aba_espaco(ctx: Contexto) -> None:
    metodo = st.radio("Projeção", ["PCA", "t-SNE"], horizontal=True)
    xy = projetar_2d(ctx.vetores, metodo)
    pontos = ctx.base.assign(x=xy[:, 0], y=xy[:, 1], trecho=ctx.base["texto"].str[:90] + "…")
    fig = px.scatter(pontos, x="x", y="y", color="categoria_oficial", symbol="categoria_oficial", text="id",
                     color_discrete_map=nucleo.CORES_CATEGORIA, hover_data={"trecho": True, "x": False, "y": False},
                     labels={"categoria_oficial": "categoria oficial"})
    fig.update_traces(marker=dict(size=12, line=dict(width=1, color="white")), textposition="top center",
                      textfont_size=9)
    fig.update_layout(height=560, xaxis_title=None, yaxis_title=None)
    st.plotly_chart(fig, use_container_width=True)

    S = nucleo.similaridade(ctx.vetores)
    np.fill_diagonal(S, -1)
    cat = ctx.base["categoria_oficial"].to_numpy()
    mesma = cat == cat[S.argmax(axis=1)]
    esq, dir_ = st.columns([1, 2])
    esq.metric("Vizinho mais próximo da mesma categoria", f"{mesma.mean():.0%}")
    esq.dataframe(pd.Series(mesma, index=cat).groupby(level=0).mean().map("{:.0%}".format)
                  .rename("vizinho da mesma categoria"), use_container_width=True)
    dir_.markdown(
        "**Os clusters coincidem com as categorias?** Em parte. Com o modelo multilíngue, 70% das "
        "manifestações têm como vizinho mais próximo uma da mesma categoria. **Saúde** é a mais coesa: "
        "todas as manifestações de saúde têm vizinho de saúde. As fronteiras se misturam onde os temas se cruzam "
        "de verdade: **infraestrutura e meio ambiente** (bueiro entupido × terreno com lixo, ponte sobre o riacho × "
        "entulho no rio); iluminação pública perto de **segurança**, porque o cidadão cita a rua escura como "
        "risco de assalto; e a escola sem acessibilidade perto da denúncia de drogas na praça ao lado da escola. "
        "Os embeddings agrupam pelo **assunto do texto**, que nem sempre é a categoria administrativa, e por isso "
        "a categoria sugerida na busca serve só de apoio ao atendente."
    )


def aba_chunking(ctx: Contexto) -> None:
    longas = ctx.base[ctx.base["texto"].str.len() > 500].set_index("id")
    origem = st.selectbox("Partir de uma manifestação longa da base", longas.index,
                          format_func=lambda i: f"{i} · {longas.loc[i, 'categoria_oficial']} · "
                                                f"{len(longas.loc[i, 'texto'])} caracteres")
    texto = st.text_area("Texto a dividir (pode colar outro)", value=longas.loc[origem, "texto"], height=170)
    c1, c2, c3 = st.columns(3)
    estrategia = c1.selectbox("Estratégia", ["recursiva", "tamanho fixo"],
                              format_func={"recursiva": "RecursiveCharacter", "tamanho fixo": "Fixed-size (Character)"}.get)
    tamanho = c2.number_input("chunk_size (caracteres)", 50, 800, 300, 10)
    sobreposicao = c3.number_input("chunk_overlap (caracteres)", 0, 300, 60, 10)
    if sobreposicao >= tamanho:
        st.error("O overlap precisa ser menor que o chunk_size.")
        return
    if not texto.strip():
        return

    pedacos = nucleo.fatiar(texto, int(tamanho), int(sobreposicao), estrategia)
    V = ctx.vetorizador(pedacos)
    inteiro = ctx.vetorizador.um(texto)
    vizinhos = [float(V[i] @ V[i + 1]) for i in range(len(V) - 1)]

    m1, m2, m3 = st.columns(3)
    m1.metric("Chunks", len(pedacos))
    m2.metric("Tamanho médio", f"{np.mean([len(p) for p in pedacos]):.0f} car.")
    m3.metric("Coesão entre vizinhos", f"{np.mean(vizinhos):.3f}" if vizinhos else "—")

    st.dataframe(pd.DataFrame({
        "nº": range(1, len(pedacos) + 1), "caracteres": [len(p) for p in pedacos],
        "cos. com o texto inteiro": V @ inteiro, "chunk": pedacos,
    }), hide_index=True, use_container_width=True,
        column_config={"cos. com o texto inteiro": st.column_config.ProgressColumn(min_value=0, max_value=1,
                                                                                   format="%.3f"),
                       "chunk": st.column_config.TextColumn(width="large")})

    with st.expander(f"Embeddings dos chunks ({V.shape[0]} × {V.shape[1]}), primeiras 10 dimensões"):
        st.dataframe(pd.DataFrame(V[:, :10], index=[f"C{i}" for i in range(1, len(V) + 1)]).round(3),
                     use_container_width=True)

    pergunta = st.text_input("Pergunte algo sobre o texto", value="o que o cidadão está pedindo?")
    if pergunta:
        s = V @ ctx.vetorizador.um(pergunta)
        melhor = int(np.argmax(s))
        st.success(f"Chunk {melhor + 1} ({s[melhor]:.3f}): {pedacos[melhor]}")


def aba_duplicatas(ctx: Contexto) -> None:
    limiar = st.slider("Limiar de similaridade", 0.50, 0.95, nucleo.LIMIAR_DUPLICATA, 0.01)
    S = nucleo.similaridade(ctx.vetores)
    ids, textos = ctx.base["id"].tolist(), ctx.base["texto"].tolist()
    i, j = np.triu_indices(len(ids), 1)
    sel = S[i, j] >= limiar
    pares = sorted(zip(i[sel], j[sel], S[i, j][sel]), key=lambda p: -p[2])
    st.write(f"**{len(pares)} pares** acima de {limiar:.2f}, para um atendente confirmar:")
    for a, b, s in pares:
        with st.expander(f"{nucleo.faixa_de_cor(s)} {ids[a]} × {ids[b]} · {s:.3f}"):
            x, y = st.columns(2)
            x.markdown(f"**{ids[a]}** · {textos[a]}")
            y.markdown(f"**{ids[b]}** · {textos[b]}")


# ─────────────────────────────────── montagem ───────────────────────────────────
st.set_page_config(page_title="Ouvidoria Inteligente", page_icon="🏛️", layout="wide")

with st.sidebar:
    st.header("Ouvidoria · ajustes")
    nome_modelo = st.selectbox("Modelo de embedding", list(nucleo.MODELOS),
                               format_func=lambda m: f"{m.split('/')[-1]} ({nucleo.MODELOS[m]})")
    top_k = st.number_input("Top-K da busca", min_value=1, max_value=15, value=5)
    st.caption("Duplicata sugerida a partir de similaridade ≥ 0,65.")

base = nucleo.carregar_manifestacoes()
contexto = Contexto(base, vetorizar_base(nome_modelo, tuple(base["texto"])), obter_vetorizador(nome_modelo),
                    int(top_k))

st.title("🏛️ Ouvidoria Inteligente")
st.caption(f"{len(base)} manifestações · {base['categoria_oficial'].nunique()} categorias · "
           f"{int((base['texto'].str.len() > 500).sum())} textos longos · vetores de "
           f"{contexto.vetores.shape[1]} dimensões")

abas = {
    "🔍 Busca Semântica": aba_busca,
    "📋 Base Completa": aba_base,
    "🌐 Espaço Vetorial": aba_espaco,
    "🧩 Chunking": aba_chunking,
    "🔁 Duplicatas": aba_duplicatas,
}
for guia, desenhar in zip(st.tabs(list(abas)), abas.values()):
    with guia:
        desenhar(contexto)
