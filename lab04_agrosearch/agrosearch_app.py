"""
AgroSearch · protótipo de busca textual para os manuais da AgroTech Solutions
Laboratório Prático 04 · Recuperação de Informação / PLN · UNIPÊ

Tudo aqui é implementado sem bibliotecas de RI (nada de scikit-learn):
    PipelineTexto   -> tokenização, normalização, stopwords e stemming
    IndiceInvertido -> termo -> documentos, df e IDF
    RanqueadorTFIDF -> TF, TF-IDF acumulado e (bônus) cosseno entre vetores TF-IDF

Rodar:  streamlit run agrosearch_app.py
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from typing import NamedTuple

import pandas as pd
import streamlit as st

MANUAIS = [
    "A soja requer irrigação constante durante o período de floração para garantir a produtividade.",
    "O controle biológico de lagartas na soja pode ser feito com a vespa Trichogramma.",
    "A adubação verde com leguminosas melhora o nitrogênio no solo para o milho.",
    "Lagartas desfolhadoras causam grande prejuízo na cultura da soja e do algodão.",
    "A irrigação por gotejamento economiza água e é ideal para o cultivo orgânico.",
]

# Stopwords já sem acento: a filtragem acontece depois da normalização.
PALAVRAS_VAZIAS = frozenset("""
a o as os um uma uns umas de do da dos das em no na nos nas por pelo pela pelos pelas
para pra com sem sob sobre entre ate e ou mas que se ao aos ser sao foi era pode podem
ja nao muito mais menos seu sua seus suas este esta esse essa isso isto como quando
onde qual durante tambem lhe ele ela eles elas nosso nossa
""".split())

# Stemmer em estágios, inspirado no RSLP (Orengo & Huyck): cada estágio remove
# no máximo um sufixo, do mais longo para o mais curto, e só se sobrar um
# radical de pelo menos 3 letras.
ESTAGIOS_STEMMER = (
    ("plural", ("oes", "aes", "ais", "eis", "ns", "s")),
    ("nominal", ("amentos", "imentos", "amento", "imento", "idades", "idade", "mente",
                 "acoes", "acao", "icao", "adora", "ador", "ancia", "encia", "avel",
                 "ivel", "ismo", "ista", "oso", "osa")),
    ("verbal", ("aram", "eram", "iram", "ando", "endo", "indo", "ado", "ido", "ada",
                "ida", "am", "em", "ar", "er", "ir")),
    ("vogal", ("a", "e", "o")),
)


class Etapas(NamedTuple):
    tokens: list[str]
    normalizados: list[str]
    sem_stopwords: list[str]
    finais: list[str]


@dataclass(frozen=True)
class PipelineTexto:
    remover_stopwords: bool = True
    aplicar_stemming: bool = True

    @staticmethod
    def tokenizar(texto: str) -> list[str]:
        return re.findall(r"\w+", texto)

    @staticmethod
    def normalizar(token: str) -> str:
        decomposto = unicodedata.normalize("NFD", token.lower())
        return "".join(ch for ch in decomposto if not unicodedata.combining(ch))

    @staticmethod
    def radical(palavra: str) -> str:
        for _, sufixos in ESTAGIOS_STEMMER:
            for suf in sufixos:
                if palavra.endswith(suf) and len(palavra) - len(suf) >= 3:
                    palavra = palavra[: -len(suf)]
                    break
        return palavra

    def etapas(self, texto: str) -> Etapas:
        tokens = self.tokenizar(texto)
        normalizados = [self.normalizar(t) for t in tokens]
        filtrados = [t for t in normalizados if t not in PALAVRAS_VAZIAS] if self.remover_stopwords else normalizados
        finais = [self.radical(t) for t in filtrados] if self.aplicar_stemming else filtrados
        return Etapas(tokens, normalizados, filtrados, finais)

    def __call__(self, texto: str) -> list[str]:
        return self.etapas(texto).finais


class IndiceInvertido:
    """Mapa termo -> lista ordenada de documentos que contêm o termo."""

    def __init__(self, colecao: dict[str, list[str]]):
        self.colecao = colecao
        self.postings: dict[str, list[str]] = {}
        for doc_id, termos in colecao.items():
            for termo in dict.fromkeys(termos):  # cada documento entra uma vez por termo
                self.postings.setdefault(termo, []).append(doc_id)
        self.postings = dict(sorted(self.postings.items()))

    @property
    def n_docs(self) -> int:
        return len(self.colecao)

    def df(self, termo: str) -> int:
        return len(self.postings.get(termo, ()))

    def idf(self, termo: str) -> float:
        """IDF(t) = log10(N / df(t)); termo fora do vocabulário vale 0."""
        df = self.df(termo)
        return math.log10(self.n_docs / df) if df else 0.0

    def como_tabela(self) -> pd.DataFrame:
        return pd.DataFrame(
            [{"termo": t, "df": len(d), "IDF": round(self.idf(t), 4), "postings": " → ".join(d)}
             for t, d in self.postings.items()]
        ).sort_values(["df", "termo"], ascending=[False, True], ignore_index=True)


class RanqueadorTFIDF:
    def __init__(self, indice: IndiceInvertido):
        self.indice = indice

    @staticmethod
    def tf(termo: str, termos_doc: list[str]) -> float:
        """TF(t, d) = f(t, d) / |d|."""
        return termos_doc.count(termo) / len(termos_doc) if termos_doc else 0.0

    def peso(self, termo: str, doc_id: str) -> float:
        return self.tf(termo, self.indice.colecao[doc_id]) * self.indice.idf(termo)

    def memoria_de_calculo(self, consulta: list[str]) -> pd.DataFrame:
        linhas = []
        for doc_id, termos_doc in self.indice.colecao.items():
            for t in dict.fromkeys(consulta):
                tf = self.tf(t, termos_doc)
                linhas.append({"doc": doc_id, "termo": t, "f(t,d)": termos_doc.count(t), "|d|": len(termos_doc),
                               "TF": tf, "df": self.indice.df(t), "IDF": self.indice.idf(t),
                               "TF×IDF": tf * self.indice.idf(t)})
        return pd.DataFrame(linhas)

    def ranking(self, consulta: list[str]) -> pd.DataFrame:
        termos = list(dict.fromkeys(consulta))
        placar = {d: sum(self.peso(t, d) for t in termos) for d in self.indice.colecao}
        tabela = pd.DataFrame({"doc": placar.keys(), "tfidf_acumulado": placar.values()})
        tabela = tabela.sort_values("tfidf_acumulado", ascending=False, ignore_index=True)
        tabela.insert(0, "posição", range(1, len(tabela) + 1))
        return tabela

    # ---- bônus ----
    def vetor(self, termos: list[str]) -> list[float]:
        return [self.tf(t, termos) * self.indice.idf(t) for t in self.indice.postings]

    def cosseno(self, consulta: list[str], doc_id: str) -> float:
        q, d = self.vetor(consulta), self.vetor(self.indice.colecao[doc_id])
        produto = sum(a * b for a, b in zip(q, d))
        normas = math.sqrt(sum(a * a for a in q)) * math.sqrt(sum(b * b for b in d))
        return produto / normas if normas else 0.0


# ═════════════════════════════════════════ tela ═════════════════════════════════════════
st.set_page_config(page_title="AgroSearch", page_icon="🌾", layout="wide")

with st.sidebar:
    st.subheader("Pré-processamento")
    usar_stop = st.checkbox("Remover stopwords", value=True)
    usar_stem = st.checkbox("Aplicar stemming", value=True)
    st.caption("Ligue e desligue para ver o vocabulário mudar.")
    st.divider()
    st.subheader("Manuais indexados")
    editados = st.data_editor(pd.DataFrame({"trecho": MANUAIS}), num_rows="dynamic",
                              use_container_width=True, hide_index=True, key="manuais")

trechos = [str(t).strip() for t in editados["trecho"].dropna() if str(t).strip()]
if not trechos:
    st.warning("Nenhum trecho na base: adicione pelo menos uma linha na tabela da barra lateral.")
    st.stop()

pipeline = PipelineTexto(usar_stop, usar_stem)
textos = {f"D{i}": t for i, t in enumerate(trechos, start=1)}
indice = IndiceInvertido({d: pipeline(t) for d, t in textos.items()})
ranqueador = RanqueadorTFIDF(indice)

st.title("🌾 AgroSearch")
cab1, cab2 = st.columns([3, 2], vertical_alignment="bottom")
consulta_txt = cab1.text_input("O que você procura nos manuais?", value="lagartas na soja")
cab2.markdown(f"**{indice.n_docs}** trechos · **{len(indice.postings)}** termos no vocabulário · "
              f"stopwords {'✂️' if usar_stop else '—'} · stemming {'🌱' if usar_stem else '—'}")

consulta = pipeline(consulta_txt)
if not consulta:
    st.info("Depois da limpeza a consulta ficou vazia. Tente outras palavras.")
    st.stop()

ranking = ranqueador.ranking(consulta)
vencedor = ranking.iloc[0]
desconhecidos = [t for t in consulta if t not in indice.postings]

res, lado = st.columns([3, 2])
with res:
    if vencedor.tfidf_acumulado > 0:
        st.success(f"**{vencedor.doc}** é o trecho mais relevante (TF-IDF = {vencedor.tfidf_acumulado:.4f})\n\n"
                   f"> {textos[vencedor.doc]}")
    else:
        st.error("Nenhum trecho contém os termos da consulta.")
    tabela = ranking.assign(trecho=ranking["doc"].map(textos))
    st.dataframe(
        tabela.style.format({"tfidf_acumulado": "{:.4f}"}).apply(
            lambda linha: ["background-color: rgba(76,175,80,.28); font-weight: 600" if linha.name == 0
                           and linha.tfidf_acumulado > 0 else ""] * len(linha), axis=1),
        use_container_width=True, hide_index=True,
        column_config={"tfidf_acumulado": "TF-IDF acumulado", "trecho": st.column_config.TextColumn(width="large")},
    )
with lado:
    st.markdown("**Consulta depois do pipeline**")
    st.code(" · ".join(consulta), language=None)
    if desconhecidos:
        st.warning(f"Fora do vocabulário (IDF = 0): {', '.join(desconhecidos)}")
    st.bar_chart(ranking.set_index("doc")["tfidf_acumulado"], horizontal=True, height=220)

st.divider()
st.subheader("Por dentro do motor")
visao = st.radio("Ver:", ["Pipeline de pré-processamento", "Índice invertido", "Memória de cálculo TF-IDF",
                          "Bônus · cosseno"], horizontal=True, label_visibility="collapsed")

if visao == "Pipeline de pré-processamento":
    escolhido = st.selectbox("Trecho:", list(textos), format_func=lambda d: f"{d} · {textos[d][:60]}…")
    e = pipeline.etapas(textos[escolhido])
    passos = [("1 · Tokenização", e.tokens), ("2 · Normalização", e.normalizados),
              ("3 · Stopwords" + ("" if usar_stop else " (desligado)"), e.sem_stopwords),
              ("4 · Stemming" + ("" if usar_stem else " (desligado)"), e.finais)]
    for coluna, (nome, tokens) in zip(st.columns(4), passos):
        coluna.markdown(f"**{nome}**  \n{len(tokens)} tokens")
        coluna.write(tokens)

elif visao == "Índice invertido":
    st.caption("Termo → documentos em que aparece (postings list), montado com os tokens já processados.")
    st.dataframe(indice.como_tabela(), use_container_width=True, hide_index=True)
    with st.expander("Mesmo índice em JSON"):
        st.json(indice.postings)

elif visao == "Memória de cálculo TF-IDF":
    st.latex(r"TF(t,d)=\frac{f(t,d)}{|d|}\qquad IDF(t)=\log_{10}\frac{N}{df(t)}\qquad "
             r"\text{score}(d)=\sum_{t\in q}TF(t,d)\cdot IDF(t)")
    memoria = ranqueador.memoria_de_calculo(consulta)
    st.dataframe(memoria.style.format({"TF": "{:.4f}", "IDF": "{:.4f}", "TF×IDF": "{:.4f}"}),
                 use_container_width=True, hide_index=True)
    st.caption(f"N = {indice.n_docs}. A soma da coluna TF×IDF por documento dá o ranking acima.")

else:
    st.markdown("A soma de TF-IDF premia quem tem um termo raro. O cosseno compara o vetor inteiro da "
                "consulta com o de cada trecho, o que funciona melhor em consultas de várias palavras.")
    st.latex(r"\cos(\vec q,\vec d)=\frac{\vec q\cdot\vec d}{\lVert\vec q\rVert\,\lVert\vec d\rVert}")
    comparacao = ranking.assign(cosseno=[ranqueador.cosseno(consulta, d) for d in ranking["doc"]])
    comparacao = comparacao.sort_values("cosseno", ascending=False, ignore_index=True)
    st.dataframe(comparacao.style.format({"tfidf_acumulado": "{:.4f}", "cosseno": "{:.4f}"}),
                 use_container_width=True, hide_index=True)
    st.caption(f"Vetores com {len(indice.postings)} dimensões (uma por termo do vocabulário).")
