"""
Núcleo da Ouvidoria Inteligente: tudo o que o app e os notebooks compartilham.

    carregar_manifestacoes()  -> DataFrame com as 40 manifestações
    Vetorizador               -> embeddings normalizados (produto escalar = cosseno)
    detectar_duplicatas()     -> pares acima de um limiar de similaridade
    fatiar()                  -> chunking com RecursiveCharacterTextSplitter ou tamanho fixo
    faixa_de_cor()            -> 🟢 > 0,7 · 🟡 > 0,5 · 🔴 demais
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")  # silencia aviso do joblib no Windows

PASTA_DADOS = Path(__file__).parent / "data"

MODELO_PADRAO = "paraphrase-multilingual-MiniLM-L12-v2"
MODELOS = {
    MODELO_PADRAO: "multilíngue, recomendado para PT-BR",
    "sentence-transformers/all-MiniLM-L6-v2": "inglês, para comparação",
}
CORES_CATEGORIA = {
    "infraestrutura": "#4C72B0", "saúde": "#C44E52", "segurança": "#8172B2",
    "educação": "#DD8452", "meio ambiente": "#55A868",
}
LIMIAR_DUPLICATA = 0.65  # melhor F1 na Entrega 2 e ≈ percentil 99 da distribuição
SEPARADORES = ["\n\n", "\n", ". ", ", ", " ", ""]


def carregar_manifestacoes() -> pd.DataFrame:
    with open(PASTA_DADOS / "manifestacoes.json", encoding="utf-8") as f:
        return pd.DataFrame(json.load(f))


def carregar_gabarito() -> set[tuple[str, str]]:
    with open(PASTA_DADOS / "duplicatas_gabarito.json", encoding="utf-8") as f:
        return {tuple(sorted(par)) for par in json.load(f)}


class Vetorizador:
    """Envolve um SentenceTransformer e sempre devolve vetores de norma 1."""

    def __init__(self, nome: str = MODELO_PADRAO):
        from sentence_transformers import SentenceTransformer
        self.nome = nome
        self.modelo = SentenceTransformer(nome)

    def __call__(self, textos) -> np.ndarray:
        return self.modelo.encode(list(textos), normalize_embeddings=True)

    def um(self, texto: str) -> np.ndarray:
        return self([texto])[0]


def similaridade(a: np.ndarray, b: np.ndarray | None = None) -> np.ndarray:
    """Cosseno entre vetores já normalizados (é só o produto escalar)."""
    return a @ (a if b is None else b).T


def detectar_duplicatas(textos, limiar: float = 0.85, vetorizador: Vetorizador | None = None,
                        ids=None) -> list[tuple]:
    """Todos os pares (id_a, id_b, cosseno) com similaridade >= limiar, do mais parecido ao menos."""
    vetorizador = vetorizador or Vetorizador()
    ids = list(ids) if ids is not None else list(range(len(textos)))
    S = similaridade(vetorizador(textos))
    i, j = np.triu_indices(len(ids), k=1)
    acima = S[i, j] >= limiar
    pares = [(ids[a], ids[b], round(float(s), 4)) for a, b, s in zip(i[acima], j[acima], S[i, j][acima])]
    return sorted(pares, key=lambda p: -p[2])


def fatiar(texto: str, tamanho: int, sobreposicao: int, estrategia: str = "recursiva") -> list[str]:
    from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
    if estrategia == "recursiva":
        splitter = RecursiveCharacterTextSplitter(chunk_size=tamanho, chunk_overlap=sobreposicao,
                                                  separators=SEPARADORES, keep_separator="end")
    else:
        splitter = CharacterTextSplitter(chunk_size=tamanho, chunk_overlap=sobreposicao, separator=" ")
    return splitter.split_text(texto)


def faixa_de_cor(score: float) -> str:
    return "🟢" if score > 0.7 else "🟡" if score > 0.5 else "🔴"
