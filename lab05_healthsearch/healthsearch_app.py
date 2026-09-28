"""
HealthSearch · busca híbrida para protocolos clínicos de pronto-socorro
Laboratório Prático 05 · Desafio Integrador · UNIPÊ

Arquitetura:
    LimpadorClinico  -> Fase 1: minúsculas, sem acentos/caracteres especiais, sem stopwords
    MotorLexico      -> Fase 2: Okapi BM25 (rank_bm25) com k1 e b vindos da sidebar
    MotorSemantico   -> Fase 3: embeddings densos + similaridade de cosseno
    fundir_rrf       -> Fase 4: Reciprocal Rank Fusion ponderado por alfa
    Reranqueador     -> Bônus: Cross-Encoder sobre o Top-3 do RRF

Rodar:  streamlit run healthsearch_app.py
"""

from __future__ import annotations

import re
import time
import unicodedata
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer


# ───────────────────────────── corpus (hardcode do enunciado) ─────────────────────────────
@dataclass(frozen=True)
class Diretriz:
    codigo: str
    titulo: str
    trecho: str

    @property
    def texto(self) -> str:
        return f"{self.titulo}. {self.trecho}"


DIRETRIZES = (
    Diretriz("Doc 1", "Protocolo Emergência ECG",
             "Pacientes com dor precordial aguda e suspeita de síndrome coronariana devem realizar "
             "eletrocardiograma CÓD-ECG-12D em até 10 minutos."),
    Diretriz("Doc 2", "Guia de Farmacologia Cardíaca",
             "O uso imediato de ácido acetilsalicílico e antiagregantes plaquetários reduz a mortalidade "
             "no infarto agudo do miocárdio."),
    Diretriz("Doc 3", "Diretriz de Hipertensão Arterial",
             "A crise hipertensiva severa requer administração de anti-hipertensivos venosos e "
             "monitoramento contínuo da pressão arterial na UTI."),
    Diretriz("Doc 4", "Manual de AVC Isquêmico",
             "O acidente vascular cerebral isquêmico agudo deve ser tratado com trombolíticos venosos em "
             "até quatro horas e meia do início dos sintomas."),
    Diretriz("Doc 5", "Protocolo de Reanimação RCR",
             "Parada cardiorrespiratória em adultos exige compressões torácicas contínuas de alta qualidade "
             "e desfibrilação precoce no código azul."),
    Diretriz("Doc 6", "Procedimentos de UTI Geral",
             "Para diagnóstico do protocolo CÓD-ECG-12D em arritmias complexas, recomenda-se a "
             "monitorização cardíaca contínua por telemetria."),
)
POR_CODIGO = {d.codigo: d for d in DIRETRIZES}

# Consultas que expõem o ponto cego de cada motor
ATALHOS = ["ataque cardíaco", "CÓD-ECG-12D", "derrame cerebral", "coração parou de bater",
           "exame CÓD-ECG-12D para infarto", "pressão muito alta"]

# O E5 foi treinado com prefixos diferentes para consulta e documento.
MODELOS = {
    "intfloat/multilingual-e5-small": ("query: ", "passage: "),
    "paraphrase-multilingual-MiniLM-L12-v2": ("", ""),
    "sentence-transformers/all-MiniLM-L6-v2": ("", ""),
}
CROSS_ENCODERS = ["cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", "cross-encoder/ms-marco-MiniLM-L-6-v2"]


# ───────────────────────────── Fase 1 · limpeza ─────────────────────────────
class LimpadorClinico:
    STOPWORDS = frozenset("""
    a o as os um uma uns umas de do da dos das em no na nos nas por pelo pela pelos pelas para com
    sem e ou que se ao aos ate sobre entre ser sao foi deve devem pode muito mais meia seu sua seus
    suas este esta esse essa isso como quando onde qual tem ter ha nao recomenda requer exige uso
    """.split())

    def __call__(self, texto: str) -> list[str]:
        texto = unicodedata.normalize("NFD", texto.lower())
        texto = "".join(c for c in texto if not unicodedata.combining(c))
        texto = re.sub(r"[^\w\s-]", " ", texto)  # hífen fica: "cód-ecg-12d" continua um token só
        return [t for t in (p.strip("-") for p in texto.split()) if t and t not in self.STOPWORDS]


limpar = LimpadorClinico()


# ───────────────────────────── estrutura comum de resultado ─────────────────────────────
@dataclass
class Ranking:
    motor: str
    pontuacao: dict[str, float]
    posicoes: dict[str, int] = field(default_factory=dict)  # só documentos recuperados
    ms: float = 0.0

    @classmethod
    def a_partir_de(cls, motor: str, pontuacao: dict[str, float], ms: float, exige_positivo: bool = False):
        ordem = sorted(pontuacao, key=pontuacao.get, reverse=True)
        if exige_positivo:
            ordem = [d for d in ordem if pontuacao[d] > 0]
        return cls(motor, pontuacao, {d: i for i, d in enumerate(ordem, start=1)}, ms)

    def topo(self, n: int = 3) -> list[str]:
        return sorted(self.posicoes, key=self.posicoes.get)[:n]


# ───────────────────────────── Fase 2 · BM25 ─────────────────────────────
class MotorLexico:
    def __init__(self, k1: float, b: float):
        self.k1, self.b = k1, b
        self.docs_tokens = [limpar(d.texto) for d in DIRETRIZES]
        self.bm25 = BM25Okapi(self.docs_tokens, k1=k1, b=b)

    def ranquear(self, consulta: str) -> Ranking:
        inicio = time.perf_counter()
        termos = limpar(consulta)
        notas = self.bm25.get_scores(termos) if termos else np.zeros(len(DIRETRIZES))
        pontuacao = {d.codigo: float(n) for d, n in zip(DIRETRIZES, notas)}
        # quem não tem nenhum termo da consulta fica fora do ranking léxico
        return Ranking.a_partir_de("BM25", pontuacao, (time.perf_counter() - inicio) * 1000, exige_positivo=True)

    def explicar(self, consulta: str) -> pd.DataFrame:
        linhas = []
        for d, tokens in zip(DIRETRIZES, self.docs_tokens):
            for t in limpar(consulta):
                f = tokens.count(t)
                idf = self.bm25.idf.get(t, 0.0)
                denominador = f + self.k1 * (1 - self.b + self.b * len(tokens) / self.bm25.avgdl)
                linhas.append({"diretriz": d.codigo, "termo": t, "f(t,D)": f, "|D|": len(tokens), "IDF": idf,
                               "contribuição": idf * f * (self.k1 + 1) / denominador if denominador else 0.0})
        return pd.DataFrame(linhas)


# ───────────────────────────── Fase 3 · embeddings ─────────────────────────────
@st.cache_resource(show_spinner="Carregando modelo de embeddings…")
def _encoder(nome: str) -> SentenceTransformer:
    return SentenceTransformer(nome)


@st.cache_data(show_spinner="Vetorizando as diretrizes…")
def _vetores_corpus(nome: str) -> np.ndarray:
    prefixo_doc = MODELOS[nome][1]
    return _encoder(nome).encode([prefixo_doc + d.texto for d in DIRETRIZES], normalize_embeddings=True)


class MotorSemantico:
    def __init__(self, modelo: str):
        self.modelo = modelo

    def ranquear(self, consulta: str) -> Ranking:
        encoder, corpus = _encoder(self.modelo), _vetores_corpus(self.modelo)  # carga fica fora da medição
        inicio = time.perf_counter()
        prefixo_q = MODELOS[self.modelo][0]
        q = encoder.encode([prefixo_q + consulta], normalize_embeddings=True)[0]
        cossenos = corpus @ q  # vetores normalizados: produto escalar = cosseno
        pontuacao = {d.codigo: float(c) for d, c in zip(DIRETRIZES, cossenos)}
        return Ranking.a_partir_de("Semântico", pontuacao, (time.perf_counter() - inicio) * 1000)


# ───────────────────────────── Fase 4 · RRF ─────────────────────────────
def fundir_rrf(lexico: Ranking, semantico: Ranking, alfa: float, k: int = 60) -> Ranking:
    """Score(D) = α·1/(k + Rank_BM25) + (1−α)·1/(k + Rank_Sem).
    Documento ausente de uma lista não recebe a parcela dela."""
    inicio = time.perf_counter()
    notas = {}
    for d in POR_CODIGO:
        parcela_lex = alfa / (k + lexico.posicoes[d]) if d in lexico.posicoes else 0.0
        parcela_sem = (1 - alfa) / (k + semantico.posicoes[d]) if d in semantico.posicoes else 0.0
        notas[d] = parcela_lex + parcela_sem
    return Ranking.a_partir_de("Híbrido RRF", notas, (time.perf_counter() - inicio) * 1000)


# ───────────────────────────── Bônus · Cross-Encoder ─────────────────────────────
@st.cache_resource(show_spinner="Carregando Cross-Encoder…")
def _cross(nome: str) -> CrossEncoder:
    return CrossEncoder(nome)


def reranquear(consulta: str, candidatos: list[str], modelo: str) -> tuple[list[tuple[str, float]], float]:
    inicio = time.perf_counter()
    notas = _cross(modelo).predict([[consulta, POR_CODIGO[d].texto] for d in candidatos])
    ordem = sorted(zip(candidatos, map(float, notas)), key=lambda x: x[1], reverse=True)
    return ordem, (time.perf_counter() - inicio) * 1000


# ═════════════════════════════════════════ tela ═════════════════════════════════════════
st.set_page_config(page_title="HealthSearch", page_icon="🫀", layout="wide")

with st.sidebar:
    st.markdown("### Calibração")
    st.markdown("**BM25 · motor léxico**")
    k1 = st.slider("k1 · saturação de frequência", 0.0, 3.0, 1.2, 0.1,
                   help="0 = só presença/ausência · alto = quase linear, como o TF-IDF")
    b = st.slider("b · normalização por comprimento", 0.0, 1.0, 0.75, 0.05,
                  help="0 = ignora o tamanho · 1 = normalização total por |D|/avgdl")
    st.markdown("**Embeddings · motor semântico**")
    modelo = st.selectbox("Modelo", list(MODELOS), label_visibility="collapsed")
    st.markdown("**Fusão RRF**")
    alfa = st.slider("α · peso do BM25", 0.0, 1.0, 0.5, 0.05, help="1 = só BM25 · 0 = só semântico")
    k_rrf = st.number_input("k_RRF", 1, 200, 60, help="Constante de suavização de posição (padrão 60)")
    st.divider()
    usar_cross = st.checkbox("⭐ Re-ranking com Cross-Encoder no Top-3")
    modelo_cross = st.selectbox("Cross-Encoder", CROSS_ENCODERS, disabled=not usar_cross)

st.markdown("## 🫀 HealthSearch")
st.caption("Protocolos de pronto-socorro · BM25 para termos e códigos exatos, embeddings para sinônimos, RRF para juntar os dois")

atalho = st.segmented_control("Consultas de teste", ATALHOS, default=ATALHOS[0])
consulta = st.text_input("Consulta clínica", value=atalho or "", placeholder="ex.: dor no peito, AAS, CÓD-ECG-12D")
if not consulta.strip():
    st.info("Digite uma consulta ou escolha uma das consultas de teste.")
    st.stop()

lexico = MotorLexico(k1, b)
r_lex = lexico.ranquear(consulta)
r_sem = MotorSemantico(modelo).ranquear(consulta)
r_rrf = fundir_rrf(r_lex, r_sem, alfa, int(k_rrf))

# pódio: os três motores lado a lado
for coluna, r in zip(st.columns(3), (r_lex, r_sem, r_rrf)):
    with coluna.container(border=True):
        st.markdown(f"**{r.motor}** · {r.ms:.1f} ms")
        if not r.posicoes:
            st.markdown("_nenhum documento com os termos da consulta_")
        for pos, d in enumerate(r.topo(), start=1):
            st.markdown(f"{'🥇🥈🥉'[pos - 1]} {d} · {POR_CODIGO[d].titulo}")
st.caption(f"Tokens da consulta após a limpeza: `{limpar(consulta)}` · tempo total "
           f"{r_lex.ms + r_sem.ms + r_rrf.ms:.1f} ms")

abas = ["🔤 Léxico", "🧠 Semântico", "🔀 Híbrido RRF", "📊 Matriz Comparativa"] + (["⭐ Re-ranking"] if usar_cross else [])
guias = st.tabs(abas)


def barras(r: Ranking, titulo: str, formato: str) -> go.Figure:
    ordem = sorted(POR_CODIGO, key=lambda d: r.pontuacao[d])
    fig = go.Figure(go.Bar(x=[r.pontuacao[d] for d in ordem], y=[f"{d} · {POR_CODIGO[d].titulo}" for d in ordem],
                           orientation="h", text=[format(r.pontuacao[d], formato) for d in ordem],
                           textposition="outside"))
    fig.update_layout(height=300, margin=dict(l=10, r=40, t=30, b=10), title=titulo)
    return fig


with guias[0]:
    st.latex(r"score(D,Q)=\sum_{q_i\in Q} IDF(q_i)\cdot\frac{f(q_i,D)\,(k_1+1)}"
             r"{f(q_i,D)+k_1\left(1-b+b\frac{|D|}{avgdl}\right)}")
    st.plotly_chart(barras(r_lex, f"Score BM25 (k1 = {k1}, b = {b})", ".3f"), use_container_width=True)
    if not r_lex.posicoes:
        st.warning("O BM25 não achou nenhum termo da consulta nas diretrizes: busca léxica não entende sinônimos.")
    st.markdown(f"**Memória de cálculo** · avgdl = {lexico.bm25.avgdl:.2f} tokens, N = {len(DIRETRIZES)}")
    st.dataframe(lexico.explicar(consulta).style.format({"IDF": "{:.4f}", "contribuição": "{:.4f}"}),
                 use_container_width=True, hide_index=True)
    st.caption("A rank_bm25 usa IDF = ln((N − df + 0,5)/(df + 0,5)), com piso ε para termos muito comuns.")

with guias[1]:
    st.latex(r"\cos(\vec q,\vec d)=\frac{\vec q\cdot\vec d}{\lVert\vec q\rVert\,\lVert\vec d\rVert}")
    st.plotly_chart(barras(r_sem, f"Similaridade de cosseno · {modelo.split('/')[-1]}", ".3f"),
                    use_container_width=True)
    if MODELOS[modelo][0]:
        st.caption("O E5 concentra os cossenos numa faixa alta (≈ 0,8–0,9); o que decide é a ordem.")

with guias[2]:
    st.latex(r"Score_{RRF}(D)=\alpha\frac{1}{k+Rank_{BM25}}+(1-\alpha)\frac{1}{k+Rank_{Sem}}")
    tabela_rrf = pd.DataFrame([{
        "posição": r_rrf.posicoes[d], "diretriz": d, "título": POR_CODIGO[d].titulo,
        "rank BM25": r_lex.posicoes.get(d), "rank semântico": r_sem.posicoes[d],
        "parcela BM25": alfa / (k_rrf + r_lex.posicoes[d]) if d in r_lex.posicoes else 0.0,
        "parcela semântica": (1 - alfa) / (k_rrf + r_sem.posicoes[d]),
        "score RRF": r_rrf.pontuacao[d],
    } for d in r_rrf.topo(len(DIRETRIZES))]).astype({"rank BM25": "Int64"})
    st.dataframe(tabela_rrf.style.format({c: "{:.5f}" for c in ("parcela BM25", "parcela semântica", "score RRF")}),
                 use_container_width=True, hide_index=True)
    st.caption("rank BM25 vazio = a diretriz não tem nenhum termo da consulta e não recebe a parcela léxica. "
               "O RRF usa só posições, então não precisa normalizar as escalas do BM25 e do cosseno.")
    for d in r_rrf.topo():
        st.markdown(f"**{d} · {POR_CODIGO[d].titulo}** — {POR_CODIGO[d].trecho}")

with guias[3]:
    colunas = ["BM25", "Semântico", "Híbrido RRF"]
    fig = go.Figure()
    for d in POR_CODIGO:
        ys = [r.posicoes.get(d, len(DIRETRIZES) + 1) for r in (r_lex, r_sem, r_rrf)]
        fig.add_trace(go.Scatter(x=colunas, y=ys, mode="lines+markers+text", name=d,
                                 text=["" , "", d], textposition="middle right", line=dict(width=3),
                                 marker=dict(size=11)))
    fig.update_yaxes(autorange="reversed", tickvals=list(range(1, 8)),
                     ticktext=[f"{i}º" for i in range(1, 7)] + ["sem match"], title="posição")
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10),
                      title="Como cada diretriz muda de posição entre os motores")
    st.plotly_chart(fig, use_container_width=True)
    matriz = pd.DataFrame({
        "título": [POR_CODIGO[d].titulo for d in POR_CODIGO],
        "BM25": pd.array([r_lex.posicoes.get(d) for d in POR_CODIGO], dtype="Int64"),
        "Semântico": [r_sem.posicoes[d] for d in POR_CODIGO],
        "Híbrido RRF": [r_rrf.posicoes[d] for d in POR_CODIGO],
    }, index=list(POR_CODIGO)).sort_values("Híbrido RRF")
    st.dataframe(matriz, use_container_width=True)
    st.caption("None na coluna BM25 = a diretriz não tem nenhum termo da consulta (sem match léxico).")
    st.info("Com **CÓD-ECG-12D** só o BM25 acerta em cheio (Doc 1 e Doc 6). Com **ataque cardíaco** o BM25 "
            "não acha nada e o semântico encontra a diretriz de infarto. O RRF premia quem vai bem nas duas listas.")

if usar_cross:
    with guias[4]:
        st.markdown(f"O `{modelo_cross}` lê cada par (consulta, diretriz) junto e dá uma nota de relevância mais "
                    "precisa que o bi-encoder. Por ser caro, só reordena os 3 primeiros do RRF.")
        top3 = r_rrf.topo(3)
        nova_ordem, ms_cross = reranquear(consulta, top3, modelo_cross)
        st.dataframe(pd.DataFrame([{
            "antes (RRF)": top3.index(d) + 1, "depois": i, "variação": f"{top3.index(d) + 1 - i:+d}",
            "diretriz": d, "título": POR_CODIGO[d].titulo, "score RRF": round(r_rrf.pontuacao[d], 5),
            "nota Cross-Encoder": round(nota, 4)} for i, (d, nota) in enumerate(nova_ordem, start=1)]),
            use_container_width=True, hide_index=True)
        st.caption(f"Cross-Encoder: {ms_cross:.0f} ms · variação positiva = subiu de posição.")
        if modelo_cross.endswith("ms-marco-MiniLM-L-6-v2"):
            st.warning("Modelo treinado em inglês: em consultas em português as notas são menos confiáveis "
                       "que as do mMARCO multilíngue.")
