import numpy as np

def build_absorbing_block_mask(km: int, pre_block: int, post_block: int,
                                bridge: tuple[int, int] | None = None) -> np.ndarray:
    """
    Constrói máscara booleana (km x km) para um bloco absorvente pós-quebra.
    - pre_block + post_block deve == km.
    - Dentro de cada bloco, todas as transições permitidas (livres).
    - Transições do bloco pós-quebra para o pré-quebra são proibidas.
    - Transições do pré para o pós são proibidas, exceto (opcional) uma ponte 'bridge'=(i_pre, j_post).
    Ex.: km=4, pre=2, post=2, bridge=(1,2) -> permite 1->2 como "quebra" única.
    """
    if not (isinstance(km, int) and km > 0):
        raise ValueError("km deve ser inteiro > 0")
    if not (isinstance(pre_block, int) and isinstance(post_block, int)):
        raise ValueError("pre_block e post_block devem ser inteiros")
    if pre_block <= 0 or post_block <= 0 or (pre_block + post_block) != km:
        raise ValueError("pre_block + post_block deve ser igual a km e ambos > 0")
    mask = np.zeros((km, km), dtype=bool)

    pre = range(0, pre_block)
    post = range(pre_block, km)

    # livre dentro de cada bloco
    for i in pre:
        for j in pre:
            mask[i, j] = True
    for i in post:
        for j in post:
            mask[i, j] = True

    # proibições cruzadas
    for i in post:
        for j in pre:
            mask[i, j] = False  # pós -> pré (proibido)
    for i in pre:
        for j in post:
            mask[i, j] = False  # pré -> pós (proibido)

    # ponte opcional pré->pós
    if bridge is not None:
        i_pre, j_post = bridge
        if (0 <= i_pre < pre_block) and (pre_block <= j_post < km):
            mask[i_pre, j_post] = True

    # sanity-check: nenhuma linha pode ficar sem transições permitidas
    if not np.all(mask.any(axis=1)):
        bad = np.where(~mask.any(axis=1))[0]
        raise ValueError(f"build_absorbing_block_mask: linhas sem transições permitidas: {bad.tolist()}")

    return mask

def _row_softmax(row: np.ndarray) -> np.ndarray:
    r = row - np.nanmax(row)
    ex = np.exp(np.nan_to_num(r, nan=-1e6))
    s = ex.sum()
    if not np.isfinite(s) or s <= 0:
        # fallback: uniforme nos finitos
        m = np.isfinite(ex) & (ex > 0)
        cnt = int(m.sum())
        if cnt == 0:
            return np.full_like(row, 1.0 / row.size)
        out = np.zeros_like(row, dtype=float)
        out[m] = 1.0 / cnt
        return out
    return ex / s

def build_transition(theta, mask=None, absorbing_last=False, triangular=False):
    """
    Constrói matriz de transição row-stochastic na convenção interna ("from_to"):
        P[i, j] = Pr(S_t = j | S_{t-1} = i), com somas por linha = 1.

    Observação de convenção:
        A notação usual em papers (Doornik / Marçal) usa p_{i|j} = Pr(S_t = i | S_{t-1} = j), com somas por coluna = 1.
        As duas convenções se relacionam por transposição: P_pij = P.T.

    Parâmetros:
        - theta: logits (k x k) em vetor, reorganizados por linha (i).
        - mask: máscara booleana (k x k) na convenção interna; entradas False geram probabilidade 0 exata.
        - absorbing_last: se True, força última linha absorvente.
        - triangular: se True, restringe a j>=i na convenção interna.
    """
    theta = np.asarray(theta, dtype=float)
    k = int(np.sqrt(theta.size))
    if k <= 0 or (k * k) != theta.size:
        raise ValueError("build_transition: theta deve ter k*k elementos (matriz quadrada em forma vetorizada)")
    theta = theta.reshape(k, k)

    P = np.zeros((k, k), dtype=float)
    triu = np.triu(np.ones((k, k), dtype=bool), 0) if triangular else None

    for i in range(k):
        row = theta[i].copy()

        # conjunto permitido (convenção interna: linha i -> coluna j)
        allowed = np.ones((k,), dtype=bool)
        if mask is not None:
            mrow = np.asarray(mask[i], dtype=bool)
            if mrow.shape != (k,):
                raise ValueError("mask: dimensão incompatível")
            allowed &= mrow
        if triu is not None:
            allowed &= triu[i]

        if absorbing_last and i == k - 1:
            P[i] = np.eye(k)[-1]          # linha absorvente exata
            continue

        if not allowed.any():
            raise ValueError(
                f"build_transition: linha {i} sem transições permitidas após restrições "
                f"(mask/triangular/absorbing_last). Verifique a máscara efetiva e as flags; "
                f"cada estado deve ter ao menos 1 transição permitida."
            )

        # saneamento numérico (apenas para logits permitidos)
        row = np.nan_to_num(row, nan=0.0, posinf=30.0, neginf=-30.0)
        row = np.clip(row, -30.0, 30.0)

        # softmax somente nas entradas permitidas -> zeros exatos nas proibidas
        logits = row[allowed]
        m = np.max(logits)
        ex = np.exp(logits - m)
        s = ex.sum()
        probs_allowed = ex / s if (np.isfinite(s) and s > 0) else np.full_like(logits, 1.0 / logits.size)

        P[i, :] = 0.0
        P[i, allowed] = probs_allowed

    # normalização final por segurança (mantém zeros estruturais)
    rs = P.sum(axis=1, keepdims=True)
    if not np.all(rs > 0):
        raise ValueError("build_transition: linha com soma zero após restrições")
    P = P / rs
    # pós-condições: probabilidades válidas e normalizadas
    if np.nanmin(P) < -1e-12:
        raise ValueError("build_transition: probabilidade negativa detectada após normalização")
    if not np.allclose(P.sum(axis=1), 1.0, atol=1e-10):
        raise ValueError("build_transition: linhas não somam 1 após normalização")
    return P

def logsumexp(logw):
    # robusto a -inf e NaN
    logw = np.asarray(logw, dtype=float)
    if not np.isfinite(logw).any():
        return -np.inf
    m = np.nanmax(logw)
    s = np.nansum(np.exp(logw - m))
    if s <= 0 or not np.isfinite(s):
        return -np.inf
    return m + np.log(s)

def kron(Pv, Pm):
    return np.kron(Pv, Pm)

def stable_cholesky(var):
    # for scalar variance; ensure positivity
    return np.sqrt(max(var, 1e-12))

def to_positive(x):
    # softplus for strictly positive parameters (variance)
    return np.log1p(np.exp(x)) + 1e-6