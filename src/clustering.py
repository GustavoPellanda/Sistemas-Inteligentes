"""Funções da Tarefa 3: inferência, K-Means, métricas, figuras e exportação."""

from pathlib import Path
import re
import warnings

import joblib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.text import Text
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.metrics import silhouette_samples
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


SEED = 42
FEATURES = ["idade", "fc", "fr", "pas", "spo2", "temp", "pr", "sg", "fx", "queim"]
ESPACIAIS = ["x_rel", "y_rel"]
PRINCIPAIS = ["x_rel", "y_rel", "sobr"]
CORES_TRI = ["#008000", "#f1c40f", "#df2020", "#101010"]
NOMES_TRI = ["verde", "amarela", "vermelha", "preta"]
CORES_CLUSTERS = list(plt.get_cmap("tab10").colors)


def salvar_figura(fig, caminho, dpi):
    """Confere a fonte e registra a escala física usada ao inserir a figura no PDF."""
    fig.canvas.draw()
    assert np.isclose(fig.get_size_inches()[0], 7.2)
    for texto in fig.findobj(Text):
        if texto.get_visible() and texto.get_text():
            assert texto.get_fontsize() == 11, "As figuras do relatório devem usar fonte 11."
    fig.savefig(caminho, dpi=dpi, metadata={"FontSizePt": "11", "WidthInches": "7.2"})


def carregar_mapa(caminho):
    """Lê o cenário fornecido e separa as vítimas; id=0 também é válido."""
    mapa = pd.read_csv(caminho)
    colunas = ESPACIAIS + ["obst", "id"] + FEATURES + ["gcs", "avpu", "tri", "sobr"]
    assert mapa.columns.tolist() == colunas, "Formato inesperado de map.csv."
    assert np.isfinite(mapa.to_numpy(dtype=float)).all()
    for coluna in ESPACIAIS + ["id"]:
        assert (mapa[coluna] == mapa[coluna].astype(int)).all()
    assert len(mapa) == 75 * 75
    assert not mapa.duplicated(ESPACIAIS).any()
    assert set(mapa.x_rel) == set(range(-30, 45))
    assert set(mapa.y_rel) == set(range(-40, 35))
    assert (mapa.id >= -1).all()
    assert ((mapa.obst.between(0, 3, inclusive="right")) | (mapa.obst == 100)).all()
    vitimas = mapa.loc[mapa.id >= 0].sort_values("id").reset_index(drop=True).copy()
    assert len(vitimas) == 410 and vitimas.id.is_unique
    assert set(vitimas.id) == set(range(410))
    assert (vitimas.obst != 100).all(), "Há vítima dentro de parede."
    assert not (vitimas[FEATURES] == -1).any().any()
    assert vitimas.idade.between(1, 90).all()
    for coluna in ["pr", "fx"]:
        assert vitimas[coluna].isin([0, 1]).all()
    for coluna in ["sg", "queim"]:
        assert vitimas[coluna].isin([0, 1, 2, 3]).all()
    assert (vitimas[["tri", "sobr"]] == -1).all().all()
    return mapa, vitimas


def carregar_modelos(base):
    """O classificador já inclui scaler; o regressor usa um scaler separado."""
    base = Path(base)
    with warnings.catch_warnings():
        warnings.simplefilter("error", InconsistentVersionWarning)
        classificador = joblib.load(base / "modelos/classificador/melhor_rn.joblib")
        regressor = joblib.load(base / "modelos/regressor/melhor_rn.joblib")
        scaler = joblib.load(base / "modelos/regressor/scaler.joblib")
    assert classificador.feature_names_in_.tolist() == FEATURES
    assert scaler.feature_names_in_.tolist() == FEATURES
    assert classificador.n_features_in_ == regressor.n_features_in_ == len(FEATURES)
    assert np.array_equal(classificador.classes_, np.arange(4))
    assert np.all(np.asarray(scaler.n_samples_seen_) == 10_000)
    assert np.all(np.asarray(classificador.named_steps["scaler"].n_samples_seen_) == 10_000)
    return classificador, regressor, scaler


def inferir(vitimas, classificador, regressor, scaler):
    """Usa apenas predict/transform; não ajusta os modelos supervisionados."""
    vitimas = vitimas.copy()
    entradas = vitimas[FEATURES]
    assert entradas.columns.tolist() == FEATURES
    with threadpool_limits(limits=1):
        vitimas["tri"] = classificador.predict(entradas).astype(int)
        vitimas["sobr"] = regressor.predict(scaler.transform(entradas))
    assert vitimas.tri.isin(range(4)).all()
    assert np.isfinite(vitimas.sobr).all()
    assert vitimas.sobr.between(0, 1).all(), (
        "A MLP previu sobr fora de [0, 1]. Investigue antes de agrupar; não há clipping."
    )
    assert not (vitimas[["tri", "sobr"]] == -1).any().any()
    return vitimas


def padronizar(vitimas, colunas):
    """Ajusta a escala do espaço de clustering uma vez para todos os K."""
    assert set(colunas).isdisjoint({"gcs", "avpu"})
    scaler = StandardScaler()
    entradas = scaler.fit_transform(vitimas[colunas])
    assert np.isfinite(entradas).all()
    np.testing.assert_allclose(entradas.mean(axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(entradas.std(axis=0, ddof=0), 1, atol=1e-12)
    return entradas, scaler


def avaliar_k(entradas, valores_k=range(2, 13)):
    """SSE do enunciado = sqrt(inércia); Sg = média das silhuetas individuais."""
    registros, modelos, silhuetas = [], {}, {}
    with threadpool_limits(limits=1):
        for k in valores_k:
            modelo = KMeans(
                n_clusters=int(k), init="k-means++", n_init=30,
                random_state=SEED, algorithm="lloyd", max_iter=500, tol=1e-4,
            ).fit(entradas)
            assert len(np.unique(modelo.labels_)) == k
            assert modelo.n_iter_ < modelo.max_iter
            residuos = entradas - modelo.cluster_centers_[modelo.labels_]
            inercia = float(np.sum(residuos ** 2))
            np.testing.assert_allclose(inercia, modelo.inertia_, rtol=1e-12)
            pontos = silhouette_samples(entradas, modelo.labels_, metric="euclidean")
            cobertura = float(np.mean(modelo.labels_ >= 0))
            sg = float(pontos.mean())
            registros.append({
                "K": int(k), "SSE": np.sqrt(inercia), "inercia": inercia,
                "RMSE": np.sqrt(inercia / len(entradas)), "silhueta": sg,
                "cobertura": cobertura, "score_final": sg * cobertura,
                "silhuetas_negativas": int((pontos < 0).sum()),
                "menor_cluster": int(np.bincount(modelo.labels_).min()),
                "maior_cluster": int(np.bincount(modelo.labels_).max()),
            })
            modelos[int(k)], silhuetas[int(k)] = modelo, pontos
    return pd.DataFrame(registros), modelos, silhuetas


def localizar_cotovelo(tabela):
    """Maior distância à reta entre os extremos da curva K x sqrt(inércia)."""
    x = tabela.K.to_numpy(dtype=float)
    y = tabela.SSE.to_numpy(dtype=float)
    x = (x - x.min()) / (x.max() - x.min())
    y = (y - y.min()) / (y.max() - y.min())
    distancias = (1 - x - y) / np.sqrt(2)
    return int(tabela.iloc[np.argmax(distancias)].K), distancias


def comparar_parametrizacoes(tabela, cotovelo):
    """Compara três K contíguos na região do cotovelo; maior score vence."""
    inicio = max(int(tabela.K.min()), min(cotovelo - 1, int(tabela.K.max()) - 2))
    candidatos = list(range(inicio, inicio + 3))
    comparacao = tabela.loc[tabela.K.isin(candidatos)].copy().reset_index(drop=True)
    comparacao.insert(0, "PAR", np.arange(1, 4))
    melhor_k = int(comparacao.sort_values(
        ["score_final", "K"], ascending=[False, True]
    ).iloc[0].K)
    return comparacao, melhor_k


def numerar_clusters(modelo):
    """Numera de 1 a K pela ordem dos centroides (x, y, sobr), sem significado de rota."""
    centros = modelo.cluster_centers_
    ordem = np.lexsort(tuple(centros[:, i] for i in reversed(range(centros.shape[1]))))
    # np.lexsort usa a última chave como principal: x, depois y e sobr.
    traducao = np.empty(modelo.n_clusters, dtype=int)
    traducao[ordem] = np.arange(1, modelo.n_clusters + 1)
    return traducao[modelo.labels_]


def resumir_clusters(vitimas):
    resumo = vitimas.groupby("cluster").agg(
        vitimas=("id", "size"), ate_12=("idade", lambda s: int((s <= 12).sum())),
        sobr_media=("sobr", "mean"), sobr_min=("sobr", "min"), sobr_max=("sobr", "max"),
        x_medio=("x_rel", "mean"), y_medio=("y_rel", "mean"),
        silhueta_media=("silhueta", "mean"),
    )
    for tri in range(4):
        resumo[f"tri_{tri}"] = vitimas.groupby("cluster")["tri"].apply(
            lambda s: int((s == tri).sum())
        )
    return resumo.reset_index()


def analisar_geometria(vitimas):
    """Resume a distribuição espacial e sua relação com a sobrevivência."""
    blocos, _, _ = np.histogram2d(
        vitimas.x_rel, vitimas.y_rel,
        bins=[np.arange(-30, 46, 15), np.arange(-40, 36, 15)],
    )
    assert blocos.sum() == len(vitimas)
    vizinhos = NearestNeighbors(n_neighbors=2).fit(vitimas[ESPACIAIS])
    distancias, _ = vizinhos.kneighbors(vitimas[ESPACIAIS])
    return {
        "bloco_15_min": int(blocos.min()), "bloco_15_max": int(blocos.max()),
        "blocos_vazios": int((blocos == 0).sum()),
        "vizinho_mais_proximo_mediana": float(np.median(distancias[:, 1])),
        "vizinho_mais_proximo_p90": float(np.quantile(distancias[:, 1], 0.9)),
        "correlacao_x_sobr": float(vitimas.x_rel.corr(vitimas.sobr)),
        "correlacao_y_sobr": float(vitimas.y_rel.corr(vitimas.sobr)),
    }


def plotar_avaliacao(tabela, caminho, escolhido, candidatos=()):
    fig, eixos = plt.subplots(1, 2, figsize=(7.2, 2.5), layout="constrained")
    for ax, coluna, titulo, ylabel in zip(
        eixos, ["SSE", "silhueta"], ["(a) K × SSE", "(b) K × silhueta"],
        ["SSE = √inércia", "Silhueta média"],
    ):
        ax.plot(tabela.K, tabela[coluna], "o-", color="#245b87", ms=4)
        if candidatos:
            ax.axvspan(min(candidatos) - 0.15, max(candidatos) + 0.15, color="#c7dcea", alpha=0.5)
            ax.axvline(5, color="#777777", ls=":", lw=1)
        ax.axvline(escolhido, color="#a3332d", ls="--", lw=1.2)
        ax.set(title=titulo, xlabel="K", ylabel=ylabel, xticks=list(range(2, 13, 2)))
        ax.grid(alpha=0.2)
    salvar_figura(fig, caminho, dpi=220)
    return fig


def plotar_espaco(vitimas, caminho):
    fig = plt.figure(figsize=(7.2, 3.2))
    ax = fig.add_axes([0.0, 0.12, 0.57, 0.85], projection="3d")
    ax.scatter(vitimas.x_rel, vitimas.y_rel, vitimas.sobr,
               c=vitimas.sobr, cmap="viridis", s=9, alpha=0.8, depthshade=False)
    ax.set(xlabel="x_rel", ylabel="y_rel", zlabel="sobr", zlim=(0, 1), zticks=[0, 0.5, 1])
    ax.set_xticks([-20, 0, 20, 40])
    ax.set_yticks([-40, -20, 0, 20])
    ax.view_init(elev=24, azim=-57)
    ax.set_box_aspect((1, 1, 0.85))
    ax.tick_params(pad=1)
    hist = fig.add_axes([0.71, 0.18, 0.27, 0.72])
    hist.hist(vitimas.sobr, bins=np.linspace(0, 1, 11), color="#42789e", edgecolor="white")
    hist.set(xlabel="sobr estimada", ylabel="Vítimas", xlim=(0, 1), xticks=[0, 0.5, 1])
    hist.grid(axis="y", alpha=0.2)
    salvar_figura(fig, caminho, dpi=220)
    return fig


def plotar_silhuetas(modelos, silhuetas, candidatos, caminho):
    fig, eixos = plt.subplots(1, 3, figsize=(7.2, 2.8), layout="constrained")
    for ax, k in zip(eixos, candidatos):
        labels = numerar_clusters(modelos[k])
        inicio, centros = 0, []
        for cluster in range(1, k + 1):
            valores = np.sort(silhuetas[k][labels == cluster])
            fim = inicio + len(valores)
            ax.fill_betweenx(np.arange(inicio, fim), 0, valores,
                             color=CORES_CLUSTERS[(cluster - 1) % 10], alpha=0.8)
            centros.append((inicio + fim) / 2)
            inicio = fim + 10
        ax.axvline(silhuetas[k].mean(), color="black", ls="--", lw=1)
        ax.set(title=f"K = {k}", xlabel="Silhueta por vítima", xlim=(-0.2, 0.8),
               yticks=centros, yticklabels=range(1, k + 1), ylabel="Cluster")
        ax.grid(axis="x", alpha=0.2)
    salvar_figura(fig, caminho, dpi=220)
    return fig


def plotar_grid(mapa, vitimas, caminho):
    """Mesmo referencial do VictSim: coluna à direita, linha para baixo; base=(0,0)."""
    terreno = mapa.pivot(index="y_rel", columns="x_rel", values="obst").sort_index()
    obst = terreno.to_numpy()
    cinza = np.where(obst == 100, 0.32, 1 - 0.15 * obst / 3)
    fig = plt.figure(figsize=(7.2, 5.8))
    ax = fig.add_axes([0.075, 0.135, 0.66, 0.82])
    ax.imshow(cinza, cmap="gray", vmin=0, vmax=1, origin="upper", interpolation="nearest",
              extent=(-30.5, 44.5, 34.5, -40.5))
    ax.set(xticks=range(-30, 45, 10), yticks=range(-40, 35, 10),
           xlabel="x_rel (colunas)")
    fig.text(0.015, 0.545, "y_rel (linhas)", rotation=90, va="center", ha="center", fontsize=11)
    ax.set_xticks(np.arange(-30.5, 45, 1), minor=True)
    ax.set_yticks(np.arange(-40.5, 35, 1), minor=True)
    ax.grid(which="minor", color="#bbbbbb", linewidth=0.12, alpha=0.7)
    ax.tick_params(which="minor", length=0)
    for v in vitimas.itertuples():
        ax.add_patch(Rectangle(
            (v.x_rel - 0.45, v.y_rel - 0.45), 0.9, 0.9, fill=False,
            edgecolor=CORES_CLUSTERS[(v.cluster - 1) % 10], linewidth=0.85, zorder=3,
        ))
    for tri, cor in enumerate(CORES_TRI):
        grupo = vitimas.loc[vitimas.tri == tri]
        ax.scatter(grupo.x_rel, grupo.y_rel, c=cor, s=5, edgecolors="#555555",
                   linewidths=0.15, zorder=4)
    ax.scatter([0], [0], marker="*", c="#00bccc", s=55, edgecolors="#004f61", linewidths=0.6, zorder=5)
    k = int(vitimas.cluster.nunique())
    clusters = [Patch(facecolor="white", edgecolor=CORES_CLUSTERS[i - 1], linewidth=1.6,
                      label=f"Cluster {i}") for i in range(1, k + 1)]
    triagens = [Line2D([], [], marker="o", linestyle="", color=cor, markersize=5,
                       label=f"{i}: {NOMES_TRI[i]}") for i, cor in enumerate(CORES_TRI)]
    ambiente = [
        Patch(facecolor="#525252", label="Parede"),
        Patch(facecolor="#e3e3e3", edgecolor="#aaaaaa", label="Terreno¹"),
        Line2D([], [], marker="*", linestyle="", color="#00aabc", markersize=9, label="Base (0,0)"),
    ]
    for handles, titulo, y in [(clusters, "Contorno", 0.94), (triagens, "tri predita", 0.54),
                                (ambiente, "Ambiente", 0.27)]:
        fig.legend(handles=handles, title=titulo, loc="upper left", bbox_to_anchor=(0.76, y),
                   frameon=False, borderaxespad=0, handlelength=1, handletextpad=0.5, labelspacing=0.35)
    fig.text(0.075, 0.98, f"K = {k} | {len(vitimas)} vítimas | grid 75 × 75", va="top", fontsize=11)
    fig.text(0.075, 0.025, "¹ Cinza mais escuro: maior dificuldade. Base absoluta: (30,40).", fontsize=11)
    salvar_figura(fig, caminho, dpi=300)
    return fig


def exportar_clusters(vitimas, pasta, k):
    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    assert set(vitimas.cluster) == set(range(1, k + 1))
    assert vitimas.id.is_unique and (vitimas.id >= 0).all()
    esperados = {f"cluster_{i}.txt" for i in range(1, k + 1)}
    # Na reexecução, remove somente saídas antigas com o formato obrigatório.
    for caminho in pasta.glob("cluster_*.txt"):
        if re.fullmatch(r"cluster_\d+\.txt", caminho.name) and caminho.name not in esperados:
            caminho.unlink()
    for cluster in range(1, k + 1):
        ids = sorted(vitimas.loc[vitimas.cluster == cluster, "id"].astype(int))
        (pasta / f"cluster_{cluster}.txt").write_text(
            "".join(f"{vid}\n" for vid in ids), encoding="utf-8"
        )
    return validar_clusters(vitimas, pasta, k)


def validar_clusters(vitimas, pasta, k):
    pasta = Path(pasta)
    arquivos = list(pasta.glob("cluster_*.txt"))
    assert {p.name for p in arquivos} == {f"cluster_{i}.txt" for i in range(1, k + 1)}
    reunidos = []
    for cluster in range(1, k + 1):
        linhas = (pasta / f"cluster_{cluster}.txt").read_text().splitlines()
        assert linhas and all(re.fullmatch(r"[0-9]+", linha) for linha in linhas)
        ids = [int(linha) for linha in linhas]
        assert set(ids) == set(vitimas.loc[vitimas.cluster == cluster, "id"])
        reunidos.extend(ids)
    assert len(reunidos) == len(set(reunidos)) == len(vitimas) == 410
    assert set(reunidos) == set(vitimas.id) == set(range(410))
    return {"arquivos": len(arquivos), "vitimas": len(reunidos), "duplicatas": 0, "ids_invalidos": 0}
