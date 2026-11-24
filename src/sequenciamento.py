"""Atribuição territorial e sequenciamento de socorro no cenário 75x75_410v."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .clustering import ESPACIAIS, carregar_mapa, carregar_modelos, inferir, validar_clusters


AGENTES = range(1, 6)
VERDE, AMARELA, VERMELHA, PRETA = range(4)
PESOS_TRI = np.array([1., 2., 3., 0.])


def carregar_cenario(raiz):
    """Reconstrói as predições da T3 e associa os IDs aos arquivos publicados."""
    raiz = Path(raiz)
    _, vitimas = carregar_mapa(raiz / "dados/map.csv")
    vitimas = inferir(vitimas, *carregar_modelos(raiz))
    pasta = raiz / "resultados/clustering"
    arquivos = sorted(pasta.glob("cluster_*.txt"))
    pares = [(int(linha), int(arquivo.stem.split("_")[1]))
             for arquivo in arquivos for linha in arquivo.read_text().splitlines()]
    membros = pd.DataFrame(pares, columns=["id", "cluster"])
    vitimas = vitimas.merge(membros, on="id", how="left", validate="one_to_one")
    validar_clusters(vitimas, pasta, len(arquivos))
    return vitimas.set_index("id").sort_index()


@dataclass
class Geometria:
    """A matriz usa IDs 0..n-1 como índices e reserva o índice n para a base."""
    d: np.ndarray
    tri: np.ndarray

    @classmethod
    def do_quadro(cls, vitimas):
        if not np.array_equal(vitimas.index, np.arange(len(vitimas))):
            raise ValueError("Os IDs do cenário devem estar em ordem de 0 a n-1.")
        xy = np.vstack([vitimas[ESPACIAIS].to_numpy(dtype=float), [0., 0.]])
        diferencas = xy[:, None, :] - xy[None, :, :]
        return cls(np.linalg.norm(diferencas, axis=2), vitimas.tri.to_numpy(dtype=int))

    @property
    def base(self):
        return len(self.tri)

    def vizinho_proximo(self, ids, fases=False):
        """Desempata por menor ID. Nas rotas só de verdes, a ida pode começar longe."""
        ids = np.sort(np.asarray(ids, dtype=int))
        urgentes = self.tri[ids] != VERDE
        grupos = [ids[urgentes], ids[~urgentes]] if fases else [ids]
        posicao = self.base
        rota = []
        for grupo in grupos:
            while len(grupo):
                distancias = self.d[posicao, grupo]
                if fases and not rota and not urgentes.any():
                    indice = distancias.argmax()
                else:
                    indice = distancias.argmin()
                posicao = grupo[indice]
                rota.append(posicao)
                grupo = np.delete(grupo, indice)
        return np.asarray(rota, dtype=int)

    def trechos(self, rota):
        percurso = np.empty(len(rota) + 2, dtype=int)
        percurso[0] = percurso[-1] = self.base
        percurso[1:-1] = rota
        return self.d[percurso[:-1], percurso[1:]]


def agente_territorial(x, y, pediatrico):
    """Fronteiras em coordenadas absolutas (bordas das células, largura 75)."""
    coluna, linha = x + 30, y + 40
    if pediatrico:
        return 1 if coluna < 37 else 2
    if coluna >= 49.5:
        return 5
    return 3 if linha < 37 else 4


def atribuir(vitimas, geometria):
    """Distribui blocos de um cluster/faixa etária e avalia cooperação por carga."""
    ativas = vitimas.loc[vitimas.tri != PRETA].copy()
    ativas["pediatrico"] = ativas.idade <= 12
    blocos = []
    for (cluster, pediatrico), grupo in ativas.groupby(["cluster", "pediatrico"]):
        x, y = grupo[ESPACIAIS].mean()
        agente = agente_territorial(x, y, pediatrico)
        blocos.append(dict(cluster=cluster, pediatrico=pediatrico, ids=grupo.index.to_numpy(),
                           x=x, y=y, inicial=agente, agente=agente))

    def distribuir():
        return {a: np.array([vid for b in blocos if b["agente"] == a for vid in b["ids"]],
                            dtype=int) for a in AGENTES}

    def estimar(ids):
        rota = geometria.vizinho_proximo(ids)
        return geometria.trechos(rota).sum()

    inicial = distribuir()
    d_inicial = np.array([estimar(ids) for ids in inicial.values()])
    n_inicial = np.array([len(ids) for ids in inicial.values()])
    d_ref = max(d_inicial.mean(), 1.)
    n_ref = max(n_inicial.mean(), 1.)

    def quadro(distancias, quantidades):
        return pd.DataFrame({"agente": list(AGENTES), "vítimas": quantidades,
                             "distância NN": distancias,
                             "carga": distancias / d_ref + quantidades / n_ref})

    atual = inicial
    ds, ns = d_inicial.copy(), n_inicial.copy()
    while True:
        melhor = None
        limite_carga = (ds / d_ref + ns / n_ref).max()
        for i, bloco in enumerate(blocos):
            origem = bloco["agente"]
            if origem <= 2:
                continue
            destino = agente_territorial(bloco["x"], bloco["y"], True)
            doador = np.setdiff1d(atual[origem], bloco["ids"])
            receptor = np.r_[atual[destino], bloco["ids"]]
            novos_d, novos_n = ds.copy(), ns.copy()
            for agente, ids in [(origem, doador), (destino, receptor)]:
                novos_d[agente-1], novos_n[agente-1] = estimar(ids), len(ids)
            pico = (novos_d / d_ref + novos_n / n_ref).max()
            if pico >= limite_carga - 1e-10 or novos_d.sum() > 1.05 * d_inicial.sum():
                continue
            chave = (pico, novos_d.sum(), i)
            if melhor is None or chave < melhor[0]:
                melhor = chave, i, destino, novos_d, novos_n
        if melhor is None:
            break
        _, i, destino, ds, ns = melhor
        blocos[i]["agente"] = destino
        atual = distribuir()

    colunas = ["cluster", "pediatrico", "x", "y", "inicial", "agente", "n"]
    territorios = pd.DataFrame([{**b, "n": len(b["ids"])} for b in blocos], columns=colunas)
    cargas = pd.concat({"inicial": quadro(d_inicial, n_inicial), "final": quadro(ds, ns)},
                       names=["atribuição"]).droplevel(1)
    return atual, territorios, cargas


def componentes(geometria, rota, corte):
    """Distância fechada, latência ponderada média e inversões radiais (em células)."""
    if not len(rota):
        return np.zeros(3)
    trechos = geometria.trechos(rota)
    pesos = PESOS_TRI[geometria.tri[rota]]
    latencia = np.average(np.cumsum(trechos[:-1]), weights=pesos)
    raios = geometria.d[geometria.base, rota]
    # As arestas base→primeira e última→base nunca contribuem para a penalidade.
    variacao = np.diff(raios)
    fronteira = max(corte - 1, 0)
    radial = -variacao[:fronteira].clip(max=0).sum() + variacao[fronteira:].clip(min=0).sum()
    return np.array([trechos.sum(), latencia, radial])


@dataclass(frozen=True)
class Politica:
    fases: bool
    prioridade: float
    radial: float


def variantes(lambda_p=0.5, lambda_r=0.5):
    if not np.isfinite([lambda_p, lambda_r]).all() or min(lambda_p, lambda_r) < 0:
        raise ValueError("Os coeficientes devem ser finitos e não negativos.")
    return {"A": Politica(False, 0., 0.), "B": Politica(False, lambda_p, 0.),
            "C": Politica(True, lambda_p, 0.), "D": Politica(True, lambda_p, lambda_r)}


def vizinha(rota, intervalos, rng):
    candidata = rota.copy()
    inicio, fim = intervalos[rng.integers(len(intervalos))]
    i, j = rng.choice(fim - inicio, size=2, replace=False) + inicio
    operador = rng.integers(3)
    if operador == 0:
        candidata[i], candidata[j] = rota[j], rota[i]
    elif operador == 1:
        # Insere o elemento i na posição final j, deslocando os vizinhos.
        if i < j:
            candidata[i:j] = rota[i+1:j+1]
        else:
            candidata[j+1:i+1] = rota[j:i]
        candidata[j] = rota[i]
    else:
        a, b = sorted((i, j))
        candidata[a:b+1] = rota[a:b+1][::-1]
    return candidata


def tempera(geometria, inicial, politica, referencia, semente, orcamento):
    """Calibra T0 e guarda a melhor permutação; orçamento inclui a calibração."""
    if orcamento < 2:
        raise ValueError("São necessárias ao menos duas avaliações de vizinhos.")
    rng = np.random.default_rng(semente)
    n = len(inicial)
    corte = int((geometria.tri[inicial] != VERDE).sum()) if politica.fases else n
    intervalos = [(0, corte), (corte, n)] if politica.fases else [(0, n)]
    intervalos = [(a, b) for a, b in intervalos if b - a >= 2]
    pesos = np.array([1., politica.prioridade, politica.radial])

    def custo(rota):
        if not politica.prioridade and not politica.radial:
            return geometria.trechos(rota).sum() / referencia
        return (componentes(geometria, rota, corte) @ pesos) / referencia

    corrente = melhor = inicial
    j_corrente = j_melhor = j_inicial = custo(inicial)
    if not intervalos:
        return melhor, j_inicial, j_melhor, 0, np.nan
    amostras = min(48, max(1, orcamento // 10))
    pioras = []
    for _ in range(amostras):
        candidata = vizinha(inicial, intervalos, rng)
        j = custo(candidata)
        if j > j_inicial:
            pioras.append(j - j_inicial)
        if j < j_melhor:
            melhor, j_melhor = candidata, j
    t0 = -np.median(pioras) / np.log(0.8) if pioras else 0.01
    restantes = orcamento - amostras
    por_nivel = max(20, n)
    niveis = int(np.ceil(restantes / por_nivel))
    for nivel in range(niveis):
        temperatura = t0 * 0.001 ** (nivel / max(1, niveis - 1))
        for _ in range(min(por_nivel, restantes - nivel * por_nivel)):
            candidata = vizinha(corrente, intervalos, rng)
            j = custo(candidata)
            delta = j - j_corrente
            if delta <= 0 or rng.random() < np.exp(-delta / temperatura):
                corrente, j_corrente = candidata, j
                if j < j_melhor:
                    melhor, j_melhor = candidata, j
    return melhor, j_inicial, j_melhor, orcamento, t0


def validar_rotas(vitimas, atribuicao, rotas, fases=True):
    if set(rotas) != set(AGENTES) or set(atribuicao) != set(AGENTES):
        raise ValueError("O plano deve identificar os cinco agentes.")
    vistos = set()
    for agente, rota in rotas.items():
        ids = set(rota)
        if len(rota) != len(ids) or ids != set(atribuicao[agente]) or ids & vistos:
            raise ValueError(f"Duplicação ou mudança de responsabilidade no agente {agente}.")
        grupo = vitimas.loc[rota]
        if (grupo.tri == PRETA).any() or (agente > 2 and (grupo.idade <= 12).any()):
            raise ValueError(f"Vítima inelegível na rota do agente {agente}.")
        if fases and np.any(np.diff((grupo.tri == VERDE).to_numpy(dtype=int)) < 0):
            raise ValueError(f"Urgente na fase de retorno do agente {agente}.")
        vistos.update(ids)
    if vistos != set(vitimas.index[vitimas.tri != PRETA]):
        raise ValueError("As rotas devem cobrir todas as vítimas ativas.")


def avaliar(geometria, rota, fases):
    tri = geometria.tri[rota]
    chegadas = np.cumsum(geometria.trechos(rota)[:-1])
    # O primeiro raio máximo define uma convenção descritiva comum às variantes.
    apice = int(np.argmax(geometria.d[geometria.base, rota])) + 1 if len(rota) else 0
    distancia, latencia, radial_comum = componentes(geometria, rota, apice)
    urgentes = np.flatnonzero(tri != VERDE)
    verdes = np.flatnonzero(tri == VERDE)
    radial_fases = componentes(geometria, rota, len(urgentes))[2] if fases else np.nan
    return {"D": distancia, "L": latencia, "R_ápice": radial_comum, "R_fases": radial_fases,
            "verdes_após_urgentes": (verdes > urgentes[-1]).mean()
            if len(urgentes) and len(verdes) else np.nan,
            **{f"L_{cor}": chegadas[tri == i].mean() if (tri == i).any() else np.nan
               for i, cor in enumerate(["VD", "AM", "VM"])}}


def experimentar(vitimas, geometria, atribuicao, sementes, orcamento, lambda_p=0.5, lambda_r=0.5):
    """Compara referências e ablações sob atribuição fixa; seleciona uma frota D."""
    sementes = tuple(sementes)
    if not sementes or len(set(sementes)) != len(sementes):
        raise ValueError("Forneça uma ou mais sementes distintas.")
    politicas = variantes(lambda_p, lambda_r)
    registros, planos_d = [], {s: {} for s in sementes}
    for agente, ids in atribuicao.items():
        livre = geometria.vizinho_proximo(ids)
        bifasica = geometria.vizinho_proximo(ids, fases=True)
        referencia = max(geometria.trechos(livre).sum(), 1.)
        for nome, rota, fases in [("NN", livre, False), ("NN-fases", bifasica, True)]:
            registros.append(dict(agente=agente, metodo=nome, semente=None,
                                   **avaliar(geometria, rota, fases)))
        for nome, politica in politicas.items():
            inicio = bifasica if politica.fases else livre
            for s in sementes:
                rota, j0, j, avaliacoes, t0 = tempera(
                    geometria, inicio, politica, referencia, [s, agente], orcamento)
                registros.append(dict(agente=agente, metodo=nome, semente=s, J_inicial=j0, J=j,
                                       lambda_p=politica.prioridade, lambda_r=politica.radial,
                                       avaliacoes=avaliacoes, T0=t0,
                                       **avaliar(geometria, rota, politica.fases)))
                if nome == "D":
                    planos_d[s][agente] = rota
    dados = pd.DataFrame(registros)
    dados["semente"] = dados.semente.astype("Int64")
    custos = dados.loc[dados.metodo == "D"].groupby("semente").J.sum()
    escolhida = int(custos.idxmin())
    dados["selecionada"] = (dados.metodo.eq("D") & dados.semente.eq(escolhida)).fillna(False)
    plano = planos_d[escolhida]
    validar_rotas(vitimas, atribuicao, plano)
    return dados, plano, escolhida


def tabelas_plano(vitimas, geometria, atribuicao, rotas):
    """Uma linha por sequência e uma linha de rastreabilidade por vítima do mapa."""
    validar_rotas(vitimas, atribuicao, rotas)
    linhas = []
    rastro = vitimas[["cluster", "tri", "idade", "sobr"]].copy()
    for coluna in ["sequência", "agente", "ordem"]:
        rastro[coluna] = pd.Series(pd.NA, index=rastro.index, dtype="Int64")
    rastro["situação"] = "excluída por triagem preta"
    for agente, rota in sorted(rotas.items()):
        grupo = vitimas.loc[rota]
        contagens = grupo.tri.value_counts()
        linhas.append({"Seq.": agente, "Agente": agente,
                       **{cor: int(contagens.get(i, 0)) for i, cor in enumerate(["VD", "AM", "VM", "PR"])},
                       "Total": len(rota), "Distância": geometria.trechos(rota).sum(),
                       "μ sobr": grupo.sobr.mean(), "σ sobr": grupo.sobr.std(ddof=0)})
        rastro.loc[rota, ["sequência", "agente"]] = agente
        rastro.loc[rota, "ordem"] = np.arange(1, len(rota) + 1)
        rastro.loc[rota, "situação"] = np.where(grupo.tri == VERDE, "retorno", "ida")
    return pd.DataFrame(linhas), rastro


def exportar(pasta, tabela, rastro, experimentos):
    """Agrupa as cinco sequências executáveis; o CSV preserva os clusters originais."""
    pasta = Path(pasta)
    sequencias = pasta / "sequencias_por_agente"
    sequencias.mkdir(parents=True, exist_ok=True)

    def gravar_ids(arquivo, ids):
        arquivo.write_text("".join(f"{vid}\n" for vid in ids), encoding="utf-8")

    for agente in AGENTES:
        grupo = rastro.loc[rastro.agente == agente].sort_values("ordem")
        destino = sequencias / f"agente_{agente}"
        destino.mkdir(exist_ok=True)

        gravar_ids(destino / "rota.txt", grupo.index)
        esperados = set()
        for cluster, membros in grupo.groupby("cluster"):
            arquivo = destino / f"cluster_{cluster}.txt"
            gravar_ids(arquivo, membros.index)
            esperados.add(arquivo)
        for antigo in destino.glob("cluster_*.txt"):
            if antigo.stem.removeprefix("cluster_").isdigit() and antigo not in esperados:
                antigo.unlink()
    tabela.to_csv(pasta / "sequencias.csv", index=False)
    rastro.to_csv(pasta / "atribuicao.csv", index_label="id")
    experimentos.to_csv(pasta / "experimentos.csv", index=False)
