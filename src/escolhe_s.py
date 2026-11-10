"""
Escolha do numero de fatores S por validacao cruzada interna.

Usa SO as celulas observadas da referencia (mascara de train.txt). As celulas
ocultas (teste do protocolo TSTR x TRTR) nao entram em nenhuma etapa.

PROCEDIMENTO
------------
As celulas observadas de cada participante sao divididas ao acaso em D dobras
(cada participante contribui para todas as dobras, de modo que sempre tem
historico). Para cada dobra e cada S candidato:

    1. ajuste nas demais dobras, com a mesma estimacao do gerador
       (gerador._cov_pares): perfil p por objeto e covariancia residual C por
       pares; B e lam dos S primeiros autovetores/autovalores de C (negativos
       truncados em zero); psi = diag(C) - parte explicada pelos S fatores
       (piso 1e-3). Covariancia do modelo:

           Sigma_S = B^T diag(lam^2) B + diag(psi)

    2. predicao das celulas da dobra pela esperanca condicional do modelo
       gaussiano de fatores, a partir das demais celulas observadas do mesmo
       participante (o):

           x_h = p_h + Sigma_S[h, o] Sigma_S[o, o]^{-1} (x_o - p_o)

       Com S = 0, Sigma e diagonal e a predicao e o perfil p (so popularidade).

    3. erro: RMSE sobre as celulas da dobra.

O procedimento e repetido com R particoes aleatorias.

REGRA DE ESCOLHA (fixada antes de rodar)
----------------------------------------
    principal:  S* = argmin do RMSE medio de validacao cruzada
    secundaria: regra de um erro-padrao (menor S cujo RMSE medio nao excede o
                minimo mais um erro-padrao; Hastie, Tibshirani e Friedman,
                The Elements of Statistical Learning, 2009, sec. 7.10)

O erro-padrao de cada repeticao e o desvio-padrao dos RMSE das D dobras
dividido por raiz de D; o relatorio usa a media sobre as R repeticoes.

USO
---
    python3 src/escolhe_s.py
    python3 src/escolhe_s.py --dobras 5 --repeticoes 20 --s-max 8

Saidas: resultados/relatorio_escolha_s.txt (o mesmo texto da tela) e
resultados/escolha_s.csv.
"""

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gerador import S_FATORES, _cov_pares, carrega_referencia

PISO_PSI = 1e-3


# --------------------------------------------------------------------------- #
class Relatorio:
    """Escreve na tela e num arquivo ao mesmo tempo (dispensa o tee)."""

    def __init__(self, caminho):
        self.tela, self.arquivo = sys.stdout, open(caminho, "w", encoding="utf-8")

    def write(self, texto):
        self.tela.write(texto)
        self.arquivo.write(texto)

    def flush(self):
        self.tela.flush()
        self.arquivo.flush()

    def fecha(self):
        sys.stdout = self.tela
        self.arquivo.close()


# --------------------------------------------------------------------------- #
def modelo_fatores(M, mascara, S):
    """Perfil e covariancia do modelo com S fatores, so com as celulas de `mascara`.

    Mesma estimacao de gerador.ajusta_atencao (sem a calibracao de a e sigma,
    que nao entra na covariancia usada para predizer)."""
    p, C = _cov_pares(M, mascara)
    w, V = np.linalg.eigh(C)
    w, V = w[::-1], V[:, ::-1]
    lam2 = np.clip(w[:S], 0, None)
    B = V[:, :S].T                                   # S x n
    explicada = lam2 @ (B ** 2) if S > 0 else np.zeros(C.shape[0])
    psi = np.clip(np.diag(C) - explicada, PISO_PSI, None)
    Sigma = (B.T * lam2) @ B + np.diag(psi) if S > 0 else np.diag(psi)
    return p, Sigma


def prediz_condicional(M, p, Sigma, obs, alvo):
    """Esperanca condicional das celulas `alvo` dado as celulas `obs`, por usuario."""
    pred = np.full(M.shape, np.nan)
    for k in range(M.shape[0]):
        o, h = np.nonzero(obs[k])[0], np.nonzero(alvo[k])[0]
        if len(h) == 0:
            continue
        if len(o) == 0:
            pred[k, h] = p[h]
            continue
        r = M[k, o] - p[o]
        coef = np.linalg.solve(Sigma[np.ix_(o, o)], r)
        pred[k, h] = p[h] + Sigma[np.ix_(h, o)] @ coef
    return pred


def particao_por_usuario(mascara, dobras, rng):
    """Rotulo de dobra (0..D-1) para cada celula observada; -1 nas ocultas.
    Cada participante tem suas celulas distribuidas por todas as dobras."""
    rot = np.full(mascara.shape, -1, dtype=int)
    for k in range(mascara.shape[0]):
        idx = rng.permutation(np.nonzero(mascara[k])[0])
        rot[k, idx] = np.arange(len(idx)) % dobras
    return rot


def validacao_cruzada(M, mascara, S_vals, dobras, repeticoes, semente):
    """RMSE por repeticao, dobra e S: array (R, D, len(S_vals))."""
    rng = np.random.default_rng(semente)
    rmse = np.zeros((repeticoes, dobras, len(S_vals)))
    for r in range(repeticoes):
        rot = particao_por_usuario(mascara, dobras, rng)
        for d in range(dobras):
            teste = rot == d
            treino = mascara & ~teste
            for j, S in enumerate(S_vals):
                p, Sigma = modelo_fatores(M, treino, S)
                pred = prediz_condicional(M, p, Sigma, treino, teste)
                rmse[r, d, j] = np.sqrt(np.mean((pred[teste] - M[teste]) ** 2))
        print(f"    repeticao {r + 1}/{repeticoes}", flush=True)
    return rmse


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Escolhe S por validacao cruzada interna.")
    ap.add_argument("--dados", default="dados")
    ap.add_argument("--resultados", default="resultados")
    ap.add_argument("--dobras", type=int, default=5)
    ap.add_argument("--repeticoes", type=int, default=20)
    ap.add_argument("--s-max", type=int, default=8)
    ap.add_argument("--semente", type=int, default=2026)
    args = ap.parse_args()

    os.makedirs(args.resultados, exist_ok=True)
    caminho_relatorio = os.path.join(args.resultados, "relatorio_escolha_s.txt")
    sys.stdout = Relatorio(caminho_relatorio)

    gd, tr, _ = carrega_referencia(args.dados)
    mascara = tr >= 0
    S_vals = list(range(0, args.s_max + 1))

    print("=" * 72)
    print("ESCOLHA DE S POR VALIDACAO CRUZADA INTERNA (so celulas observadas)")
    print(f"referencia: {gd.shape[0]} x {gd.shape[1]} | observadas {mascara.sum()}"
          f" | ocultas {(~mascara).sum()} (nao usadas)")
    print(f"{args.dobras} dobras por participante x {args.repeticoes} repeticoes"
          f" | S de 0 a {args.s_max} | semente {args.semente}")
    print(f"valor usado no artigo submetido: S = {S_FATORES}")
    print("=" * 72)

    rmse = validacao_cruzada(gd, mascara, S_vals, args.dobras, args.repeticoes,
                             args.semente)
    media = rmse.mean(axis=(0, 1))
    ep = (rmse.std(axis=1, ddof=1) / np.sqrt(args.dobras)).mean(axis=0)

    j_min = int(np.argmin(media))
    s_min = S_vals[j_min]
    s_1ep = next(S for S, m in zip(S_vals, media) if m <= media[j_min] + ep[j_min])
    # fracao das repeticoes em que cada S e o minimo (estabilidade da escolha)
    venc = np.bincount(np.argmin(rmse.mean(axis=1), axis=1), minlength=len(S_vals))

    print(f"\n    {'S':>3s}{'RMSE medio':>13s}{'EP':>9s}{'vs S=0':>10s}{'minimo em':>12s}")
    for j, S in enumerate(S_vals):
        marca = "  <- minimo" if S == s_min else ""
        marca += "  <- 1 EP" if S == s_1ep else ""
        print(f"    {S:3d}{media[j]:13.4f}{ep[j]:9.4f}{100 * (media[j] / media[0] - 1):+9.2f}%"
              f"{100 * venc[j] / args.repeticoes:11.0f}%{marca}")

    print(f"\n    S* (minimo do RMSE de validacao cruzada): {s_min}")
    print(f"    S pela regra de um erro-padrao:           {s_1ep}")
    if s_min == S_FATORES:
        print(f"    coincide com o S = {S_FATORES} do artigo submetido: as tabelas nao mudam")
    else:
        print(f"    difere do S = {S_FATORES} do artigo submetido: rodar a validacao com"
              f" python3 src/validacao.py --fatores {s_min}")

    c = os.path.join(args.resultados, "escolha_s.csv")
    with open(c, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["S", "rmse_medio", "erro_padrao", "fracao_repeticoes_minimo"])
        for j, S in enumerate(S_vals):
            w.writerow([S, f"{media[j]:.6f}", f"{ep[j]:.6f}", f"{venc[j] / args.repeticoes:.4f}"])
    print(f"\nresultados: {c}")
    print(f"relatorio:  {caminho_relatorio}")


if __name__ == "__main__":
    main()
    if isinstance(sys.stdout, Relatorio):
        sys.stdout.fecha()
