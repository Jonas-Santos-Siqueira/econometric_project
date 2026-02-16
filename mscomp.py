import numpy as np
from dataclasses import dataclass
from scipy.optimize import minimize
from typing import Optional, Tuple, Dict, Any
from scipy.special import logsumexp as sp_logsumexp
from utils import build_transition, build_absorbing_block_mask, kron, logsumexp

def _assert_transition_matrix(P: np.ndarray, *, name: str, convention: str, atol: float = 1e-8) -> None:
    """Sanity-check transition matrices to prevent silent convention mismatches.
    convention:
        - 'from_to'   : internal row-stochastic, P[i,j] = Pr(s_t=j | s_{t-1}=i), rows sum to 1
        - 'paper_pij' : paper/Doornik notation p_{i|j} = Pr(s_t=i | s_{t-1}=j), columns sum to 1
    """
    P = np.asarray(P, dtype=float)
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError(f"{name}: expected square 2D array, got shape {P.shape}")
    if not np.isfinite(P).all():
        raise ValueError(f"{name}: contains non-finite entries")
    if np.nanmin(P) < -atol:
        mn = float(np.nanmin(P))
        raise ValueError(f"{name}: has negative probability ({mn:.3e}) below tolerance -{atol:g}")
    if convention == "from_to":
        s = P.sum(axis=1)
        err = float(np.max(np.abs(s - 1.0)))
        if not np.allclose(s, 1.0, atol=atol):
            raise ValueError(f"{name}: rows must sum to 1 in 'from_to' convention; max|row_sum-1|={err:.3e}")
    elif convention == "paper_pij":
        s = P.sum(axis=0)
        err = float(np.max(np.abs(s - 1.0)))
        if not np.allclose(s, 1.0, atol=atol):
            raise ValueError(f"{name}: columns must sum to 1 in 'paper_pij' convention; max|col_sum-1|={err:.3e}")
    else:
        raise ValueError(f"{name}: unknown convention '{convention}'")

@dataclass
class MSCompResults:
    params: Dict[str, np.ndarray]
    Pm: np.ndarray
    Pv: np.ndarray
    P: np.ndarray    
    loglike: float
    filtered_probs: np.ndarray      # T x (km*kv)
    smoothed_probs: np.ndarray      # T x (km*kv)
    mean_regime_probs: np.ndarray   # T x km (smoothed marginal)
    var_regime_probs: np.ndarray    # T x kv (smoothed marginal)
    fitted: np.ndarray              # in-sample one-step-ahead predictions
    residuals: np.ndarray
    ar_order: int
    k_mean: int
    k_var: int

    k_mean_chain: Optional[int] = None               # = km se não expandido; = km^p se expandido
    use_expanded_mean: bool = False

    # [DR]
    q_exog: int = 0
    exog_lags: int = 0

    # [MA]
    ma_order: int = 0

    # Transition matrices Pm/Pv/P are stored in internal 'from_to' convention (row-stochastic):
    #   Pm[i,j] = Pr(S^m_t=j | S^m_{t-1}=i)   (rows sum to 1)
    # Paper/Doornik notation p_{i|j} = Pr(S_t=i | S_{t-1}=j) is available as the transposed views below.
    transition_convention: str = "from_to"

    Pm_pij: Optional[np.ndarray] = None
    Pv_pij: Optional[np.ndarray] = None
    P_pij: Optional[np.ndarray] = None

    smoothed_pair_probs: Optional[np.ndarray] = None

class MSComp:
    """
    --- MSMVC (Markov-Switching Mean–Variance Component) implementation ---
    
    Markov-switching mean–variance component model (Gaussian), univariate:
        y_t = a_{S^m_t} + sum_{l=1..p} phi_l y_{t-l} + eps_t,   eps_t ~ N(0, sigma^2_{S^v_t})
    with independent Markov chains for mean (S^m) and variance (S^v),
    transition matrices Pm and Pv in internal 'from_to' (row-stochastic) convention:
        Pm[i,j] = Pr(S^m_t = j | S^m_{t-1} = i),   with sum_j Pm[i,j] = 1,
    and analogously for Pv. The total transition is P = kron(Pv, Pm) in the same convention.
    For paper/Doornik notation p_{i|j} = Pr(S_t = i | S_{t-1} = j) (column-stochastic),
    use the transposed views exposed in results: Pm_pij = Pm.T, Pv_pij = Pv.T, P_pij = P.T.
    """

    def __init__(self,
                    k_mean: int,
                    k_var: int,
                    ar_order: int = 0,
                    ma_order: int = 0,
                    *,

                    # compartilhamento por regime
                    share_ar_across_regimes: bool = True,
                    share_ma_across_regimes: bool = True,

                    # DR (regressão dinâmica / exógenas)
                    D: int = 0,
                    exog_lags: int = 0,
                    share_beta_across_regimes: bool = True,

                    # restrições em P_m e P_v
                    mask_mean=None,
                    mask_var=None,
                    mask_convention: str = "from_to",  # 'from_to' (interno) | 'paper_pij' (p_{i|j})

                    triangular_Pm: bool = False,
                    triangular_Pv: bool = False,
                    absorbing_last_mean: bool = False,
                    absorbing_last_var: bool = False,

                    # quebra estrutural (conforme artigo do Marçal et. al, 2025)
                    structural_break: str | None = None,        # None | 'absorbing_last' | 'absorbing_block'
                    absorbing_block_spec: dict | None = None,   # {'pre': int, 'post': int, 'bridge': (i_pre, j_post)}

                    init_xi="auto",    # 'auto' | 'uniform' | "stationary"

                    # estabilidade/invertibilidade com margem
                    enforce_stationary_ar: bool = True,
                    enforce_invertible_ma: bool = True,
                    unit_root_margin: float = 0.03,

                    # média defasada exata (μ(S_{m,t-i}))
                    exact_mu_lags: bool = True,

                    # limite de expansão de estados quando exact_mu_lags=True (km^L); evita explosão combinatória
                    max_expanded_states: int = 20000,

                    # MS-ARMA: em coeficientes MA, regime 'current' (θ_j(S_t)) vs 'lagged' (θ_j(S_{t-j}))
                    ma_regime_timing: str = "lagged",  # 'current' | 'lagged'

                    _effective_mask_mean = None,
                    _effective_mask_var  = None,

                    # reprodutibilidade
                    random_state: int | None = None):

        # ------------ validações básicas ------------
        if not (isinstance(k_mean, int) and k_mean >= 1):
            raise ValueError("k_mean deve ser inteiro >= 1")
        if not (isinstance(k_var, int) and k_var >= 1):
            raise ValueError("k_var deve ser inteiro >= 1")
        if not (isinstance(ar_order, int) and ar_order >= 0):
            raise ValueError("ar_order deve ser inteiro >= 0")
        if not (isinstance(ma_order, int) and ma_order >= 0):
            raise ValueError("ma_order deve ser inteiro >= 0")
        if not (isinstance(D, int) and D >= 0):
            raise ValueError("D deve ser inteiro >= 0")
        if structural_break not in (None, 'absorbing_last', 'absorbing_block'):
            raise ValueError("structural_break deve ser None, 'absorbing_last' ou 'absorbing_block'")
        if unit_root_margin < 0:
            raise ValueError("unit_root_margin deve ser >= 0")
        if mask_convention not in ("from_to", "paper_pij"):
            raise ValueError("mask_convention deve ser 'from_to' ou 'paper_pij'")
        self.mask_convention = mask_convention

        # Convenções de matriz de transição usadas no código:
        # - Interna ("from_to"): P[i, j] = Pr(S_t = j | S_{t-1} = i).  -> linhas somam 1 (row-stochastic).
        # - Paper/Doornik ("paper_pij"): P_pij[i, j] = Pr(S_t = i | S_{t-1} = j). -> colunas somam 1 (col-stochastic).
        # Relação: P_pij == P.T.
        #
        # Máscaras de transição (mask_mean/mask_var) e restrições estruturais são APLICADAS internamente na convenção
        # "from_to" (linha i -> coluna j). Se o usuário fornecer máscaras na convenção "paper_pij", elas são transpostas
        # para a forma interna antes de qualquer combinação com triangularidade/absorvência/quebra estrutural.

        # ------------ atributos de dimensão ------------
        self.km = int(k_mean)
        self.kv = int(k_var)
        self.p = int(ar_order)
        self.q_ma = int(ma_order)
        self.D = int(D)
        self.Lx = int(exog_lags)  # número de lags das variáveis exógenas

        # ------------ compartilhamento ------------
        self.share_ar = bool(share_ar_across_regimes)
        self.share_ma = bool(share_ma_across_regimes)
        self.share_beta = bool(share_beta_across_regimes)

        # ------------ máscaras e restrições em P ------------
        def _to_bool_mask(M, k, name):
            if M is None:
                return None
            M = np.asarray(M, dtype=bool)
            if M.shape != (k, k):
                raise ValueError(f"{name} deve ter shape {(k, k)}")

            # Converte máscara do usuário para a convenção interna ("from_to") se ela vier no padrão do paper.
            M_int = M.T if self.mask_convention == "paper_pij" else M
            if np.any(M_int.sum(axis=1) == 0):
                raise ValueError(f"{name}: há linha sem transições permitidas ...")
            return M_int

        self.mask_mean = _to_bool_mask(mask_mean, self.km, "mask_mean")
        self.mask_var  = _to_bool_mask(mask_var,  self.kv, "mask_var")

        self._effective_mask_mean = _effective_mask_mean
        self._effective_mask_var  = _effective_mask_var

        self.triangular_Pm = bool(triangular_Pm)
        self.triangular_Pv = bool(triangular_Pv)
        self.absorbing_last_mean = bool(absorbing_last_mean)
        self.absorbing_last_var  = bool(absorbing_last_var)

        # ------------ quebra estrutural ------------
        self.structural_break = structural_break                     # None | 'absorbing_last' | 'absorbing_block'
        self.init_xi = init_xi
        self.absorbing_block_spec = dict(absorbing_block_spec or {})

        # se absorbing_block_spec existir, faça checagem leve:
        if self.structural_break == 'absorbing_block':
            pre = int(self.absorbing_block_spec.get('pre', max(1, self.km // 2)))
            post = int(self.absorbing_block_spec.get('post', self.km - pre))
            if pre <= 0 or post <= 0 or (pre + post) != self.km:
                raise ValueError("absorbing_block_spec inválido: 'pre'+'post' deve == k_mean e ser >0")
            bridge = self.absorbing_block_spec.get('bridge', None)
            if bridge is not None:
                i_pre, j_post = bridge
                if not (0 <= i_pre < pre and pre <= j_post < self.km):
                    raise ValueError("absorbing_block_spec['bridge'] deve ser (i_pre, j_post) com i_pre<pre e j_post>=pre")

        # ------------ estabilidade/invertibilidade ------------
        self.enforce_stationary_ar = bool(enforce_stationary_ar)
        self.enforce_invertible_ma = bool(enforce_invertible_ma)
        self.unit_root_margin = float(unit_root_margin)

        # ------------ opção de média defasada exata ------------
        self.exact_mu_lags = bool(exact_mu_lags)

        # ------------ timing de regime nos termos MA ------------
        if ma_regime_timing not in ("lagged", "current"):
            raise ValueError("ma_regime_timing deve ser 'lagged' ou 'current'")
        self.ma_regime_timing = ma_regime_timing

        # ------------ RNG / estado interno ------------
        self.random_state = random_state
        self.rng_ = np.random.default_rng(random_state)

        # buffers internos (preenchidos em fit)
        self._Z = None                     # matriz de exógenas (se D>0)
        self._use_expanded_mean = False    # liga cadeia expandida para μ(S_{m,t-i})
        self._mean_decoder = None          # mapeia estado expandido -> (s_t, s_{t-1}, ...)
        self._rescue_count = 0             # contagem de "rescates" no filtro
        self.diagnostics_ = {}

        # checagens finais úteis
        if max(self.p, self.q_ma) == 0 and self.exact_mu_lags:
            self.exact_mu_lags = False

        if self.km == 1:
            # estrutura MS-Comp na média degenera; desliga opções relacionadas
            self.structural_break = None
            self.triangular_Pm = False
            self.absorbing_last_mean = False
        if self.kv == 1:
            self.triangular_Pv = False
            self.absorbing_last_var = False

        self._prepare_effective_masks()

    # ==================================
    # [MSMVC] AR estável por construção
    # ==================================
    def _unconstrained_to_pacf(self, theta: np.ndarray) -> np.ndarray:
        # R -> (-1,1) com tanh(theta/2)
        return np.tanh(theta / 2.0)

    def _pacf_to_ar(self, pacf: np.ndarray) -> np.ndarray: #
        """
        Durbin-Levinson: mapeia PACF (em (-1,1)) para coeficientes AR estáveis.
        pacf: array shape (p,)
        retorna: phi shape (p,)
        """
        pacf = np.asarray(pacf, float).ravel()
        p = pacf.size
        if p == 0:
            return np.zeros(0, dtype=float)
        phi = np.zeros((p, p), dtype=float)
        phi[0, 0] = pacf[0]
        for k in range(1, p):
            phi[k, k] = pacf[k]
            for j in range(k):
                phi[k, j] = phi[k-1, j] - pacf[k] * phi[k-1, k-1-j]
        return phi[p-1, :]

    def _enforce_root_margin(self, coefs, kind: str, max_iter: int = 80, shrink: float = 0.98):
        """
        Ajusta coeficientes (AR ou MA) para que min |root| >= root_target_effective.
        Funciona para qualquer ordem p/q. Para AR/MA(0), retorna como está.
        """
        c = np.asarray(coefs, float).copy()
        if c.size == 0:
            return c

        target = self._root_target()  # 1 + max(margin, eps)
        for _ in range(max_iter):
            if kind == "ar":
                r = self._ar_roots(c)
            elif kind == "ma":
                r = self._ma_roots(c)
            else:
                raise ValueError("kind must be 'ar' or 'ma'")

            if r.size == 0:
                return c

            min_abs = float(np.min(np.abs(r)))
            if min_abs >= target:
                return c

            # Encolhe coeficientes em direção a zero: isso empurra raízes para fora.
            c *= shrink
        return c

    def _enforce_root_margin_from_pacf(self, pacf: np.ndarray, kind: str,
                                    shrink: float = 0.97, max_iter: int = 80) -> np.ndarray:
        """
        Garante min|root| >= target (target = self._root_target()) reduzindo a magnitude do PACF.

        Observação importante sobre convenções:
        - Para AR, testamos raízes de A(z)=1 - sum phi_i z^i via _ar_roots(phi).
        - Para MA, a invertibilidade é definida para B(z)=1 + sum theta_i z^i via _ma_roots(theta).
            Aqui usamos o mesmo mapeamento PACF->coeficientes (via _pacf_to_ar) para obter um vetor
            'psi' que torna A(z)=1 - sum psi_i z^i estável, e então definimos theta = -psi para que
            B(z)=1 + sum theta_i z^i = 1 - sum psi_i z^i herde a mesma propriedade (raízes fora do
            círculo unitário), alinhando sinal e convenção.

        kind: 'ar' ou 'ma'
        Retorna coeficientes (phi ou theta) já com margem.
        """
        pacf = np.asarray(pacf, float).ravel()
        target = self._root_target()

        if pacf.size == 0:
            return np.zeros(0, dtype=float)

        pacf_s = pacf.copy()
        psi = self._pacf_to_ar(pacf_s)

        for _ in range(max_iter):
            if kind == "ar":
                coefs = psi
                r = self._ar_roots(coefs)
            elif kind == "ma":
                coefs = -psi  # B(z)=1 + theta z + ... com theta = -psi
                r = self._ma_roots(coefs)
            else:
                raise ValueError("kind must be 'ar' or 'ma'")

            min_abs = np.inf if r.size == 0 else float(np.min(np.abs(r)))
            if (not np.isfinite(min_abs)) or (min_abs >= target):
                return coefs

            # se está perto demais do círculo unitário, encolhe o PACF e recalcula
            pacf_s *= shrink
            psi = self._pacf_to_ar(pacf_s)

        # fallback: devolve melhor estimativa (com convenção correta)
        return psi if kind == "ar" else -psi
    def _stabilize_ar_coefs(self, raw_phi: np.ndarray) -> np.ndarray: #
        """
        Parâmetros 'crus' (sem restrição) -> PACF via tanh -> AR estável, com margem.
        Convensão do AR: phi define o polinômio 1 - sum phi_i z^i.
        """
        raw_phi = np.asarray(raw_phi, float).ravel()
        p = raw_phi.size
        if p == 0:
            return np.zeros(0, dtype=float)

        pacf = np.tanh(raw_phi / 2.0)  # mapeia para (-1,1)
        phi = self._pacf_to_ar(pacf)

        # aplica margem nas raízes do polinômio AR
        if getattr(self, "enforce_stationary_ar", True):
            phi = self._enforce_root_margin_from_pacf(pacf, kind="ar")
        return phi

    def _ar_from_unconstrained(self, theta: np.ndarray, p: int) -> np.ndarray:
        if p == 0:
            return np.array([], dtype=float)
        pacf = self._unconstrained_to_pacf(theta[:p])
        return self._pacf_to_ar(pacf)

    def _stack_phi(self, params_phi: np.ndarray, k_mean: int, p: int, share: bool) -> np.ndarray:  #???
        """
        Constrói Φ (k_mean x p):
            - share=True : único vetor AR(p) compartilhado;
            - share=False: um AR(p) por regime de média.
        """
        if p == 0:
            return np.zeros((k_mean, 0), dtype=float)
        if share:
            ar = self._ar_from_unconstrained(params_phi, p)
            Phi = np.tile(ar, (k_mean, 1))
        else:
            Phi = np.zeros((k_mean, p), dtype=float)
            for i in range(k_mean):
                sl = slice(i * p, (i + 1) * p)
                Phi[i, :] = self._ar_from_unconstrained(params_phi[sl], p)
        return Phi

    # [MA] estabiliza coeficientes MA: 1 + θ1 z + ... + θq z^q com raízes fora do círculo unitário
    def _stabilize_ma_coefs(self, raw_theta: np.ndarray) -> np.ndarray:    #
        """
        Parâmetros 'crus' (sem restrição) -> coeficientes MA com invertibilidade e margem.

        Convenção do MA neste código:
            B(z) = 1 + θ1 z + ... + θq z^q

        Estratégia (padrão/robusta): gera um theta inicial suave (tanh) e impõe invertibilidade
        refletindo raízes que estejam dentro do círculo de raio target = self._root_target().
        """
        raw_theta = np.asarray(raw_theta, float).ravel()
        q = raw_theta.size
        if q == 0:
            return np.zeros(0, dtype=float)

        # theta inicial suave para evitar explosões numéricas na otimização
        theta = np.tanh(raw_theta / 2.0)

        if getattr(self, "enforce_invertible_ma", True):
            theta = self._enforce_invertible_ma_by_roots(theta)

        return theta

    def _expand_mean_chain(self, Pm: np.ndarray):
        """
        Constrói cadeia expandida para carregar (S_{m,t}, S_{m,t-1}, ..., S_{m,t-L+1}),
        onde L = max(p, q_ma) + 1, de modo a suportar:
        - μ(S_{m,t-i}) nos termos AR centrados
        - θ_i(S_{m,t-i}) nos termos MA (quando ma_regime_timing='lagged')

        Observação: o número de estados cresce como km^L. Para evitar explosão combinatória,
        esta rotina impõe um limite via `self.max_expanded_states` (default=20000).
        """
        L = max(int(self.p), int(self.q_ma)) + 1
        if L <= 1:
            return np.asarray(Pm, float), [(i,) for i in range(self.km)]

        km = int(self.km)
        K = km ** L
        max_states = int(getattr(self, "max_expanded_states", 20000))
        if K > max_states:
            raise ValueError(
                "exact_mu_lags requer cadeia expandida com km^L estados; "
                f"aqui km={km}, L={L} => {K} estados, acima do limite max_expanded_states={max_states}. "
                "Reduza (p,q_ma), use exact_mu_lags=False, ou aumente max_expanded_states conscientemente."
            )

        Pm = np.asarray(Pm, float)
        if Pm.shape != (km, km):
            raise ValueError(f"Pm deve ter shape {(km, km)}, mas veio {Pm.shape}")

        # Cache: depende apenas de (km, L). Evita recriar `decoder` e o mapeamento de deslocamento a cada chamada.
        if not hasattr(self, "_mean_chain_cache") or self._mean_chain_cache is None:
            self._mean_chain_cache = {}
        key = (km, L)
        cached = self._mean_chain_cache.get(key)

        if cached is None:
            from itertools import product
            decoder = list(product(range(km), repeat=L))  # (s_t, s_{t-1}, ..., s_{t-L+1})
            idx = {state: i for i, state in enumerate(decoder)}
            next_index = np.empty((K, km), dtype=np.int64)
            for i, st in enumerate(decoder):
                tail = st[:-1]
                for s_next in range(km):
                    next_index[i, s_next] = idx[(s_next,) + tail]
            self._mean_chain_cache[key] = (decoder, next_index)
        else:
            decoder, next_index = cached

        Pexp = np.zeros((K, K), dtype=float)
        # Cada linha tem exatamente km não-zeros: Pr(s_next | s_t) com deslocamento do registrador de regimes.
        for i, st in enumerate(decoder):
            s_t = st[0]
            Pexp[i, next_index[i, :]] = Pm[s_t, :]

        return Pexp, decoder

    def _needs_expanded_mean_chain(self) -> bool:
        """
        Indica se precisamos expandir a cadeia de média para carregar histórico de regimes
        (S_{m,t}, S_{m,t-1}, ..., S_{m,t-L+1}).

        Isso é necessário quando:
        - exact_mu_lags=True e p>0 (μ(S_{m,t-i}) nos termos AR centrados), ou
        - q_ma>0 e ma_regime_timing='lagged' (θ_i(S_{m,t-i}) nos termos MA; cf. Marçal et al., 2025, Eq. (3)).
        """
        if bool(getattr(self, "exact_mu_lags", False)) and int(getattr(self, "p", 0)) > 0:
            return True
        q_ma = int(getattr(self, "q_ma", 0))
        if q_ma > 0 and getattr(self, "ma_regime_timing", "lagged") == "lagged":
            return True
        return False

    def _build_seed_P_from_mask(self,
                                k: int,
                                mask: np.ndarray | None,
                                diag_p: float = 0.85,
                                bridge: tuple[int, int] | None = None,
                                bridge_weight: float = 0.08) -> np.ndarray:
        """
        Constrói uma matriz P0 (k x k) a partir de uma máscara booleana (True=permitido).
        - Coloca 'diag_p' na diagonal (se permitido) e distribui o restante igualmente
        entre os destinos permitidos da linha.
        - Se 'bridge=(i,j)' for fornecido e j for permitido na linha i, transfere uma
        fração 'bridge_weight' da massa remanescente para a coluna j.
        - Se uma linha não tiver nenhum destino permitido, ajusta para identidade.
        """
        if mask is None:
            mask = np.ones((k, k), dtype=bool)
        else:
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != (k, k):
                raise ValueError("mask deve ser k x k")

        P = np.zeros((k, k), dtype=float)
        for i in range(k):
            allowed = np.where(mask[i])[0]
            if allowed.size == 0:
                P[i, i] = 1.0
                continue

            # base na diagonal (se permitido)
            if i in allowed:
                P[i, i] = min(max(diag_p, 0.0), 1.0)
            rem = max(0.0, 1.0 - P[i, i])

            # distribui restante entre destinos permitidos (exceto diagonal)
            others = [j for j in allowed if j != i]
            if len(others) > 0 and rem > 0:
                base = rem / len(others)
                for j in others:
                    P[i, j] = base
            else:
                # só diagonal é permitida ou sem resto: identidade
                P[i, i] = 1.0

            # aplica "ponte" i->j se solicitado e viável
            if bridge is not None and i == bridge[0] and bridge[1] in allowed and bridge[1] != i:
                # desloca 'bump' da massa fora da diagonal proporcionalmente
                bump = min(bridge_weight, rem * 0.8)  # evita distorção grande
                total_off = P[i, allowed].sum() - P[i, i]
                if total_off > 0 and bump > 0:
                    for j in others:
                        dec = bump * (P[i, j] / total_off)
                        P[i, j] -= dec
                    P[i, bridge[1]] += bump

            # renormaliza linha nos permitidos
            s = P[i, allowed].sum()
            if s <= 0:
                P[i, i] = 1.0
                s = 1.0
            P[i, allowed] /= s
        return P

    def _seed_transition_logits_from_P(self,
                                    P_desired: np.ndarray,
                                    mask: np.ndarray | None,
                                    clip_low: float = -4.0,
                                    clip_high: float = 4.0,
                                    eps: float = 1e-9) -> np.ndarray:
        """
        Converte uma matriz de probabilidades-alvo P_desired (k x k) em logits 'theta' (flatten),
        respeitando a máscara (True=permitido). Entradas proibidas recebem 'clip_low'.
        """
        P_desired = np.asarray(P_desired, float)
        k = P_desired.shape[0]
        if P_desired.shape != (k, k):
            raise ValueError("P_desired deve ser k x k")

        if mask is None:
            mask = np.ones_like(P_desired, dtype=bool)
        else:
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != (k, k):
                raise ValueError("mask deve ser k x k")

        logits = np.full_like(P_desired, fill_value=clip_low, dtype=float)
        allowed = mask & (P_desired > 0)
        logits[allowed] = np.log(P_desired[allowed] + eps)
        logits = np.clip(logits, clip_low, clip_high)
        return logits.ravel()

    def _make_effective_mask(self, k: int,
                            user_mask: np.ndarray | None,
                            triangular: bool,
                            absorbing_last: bool,
                            structural_break: str | None = None,
                            block_spec: dict | None = None) -> np.ndarray:
        """
        Constrói máscara booleana (k x k) de transições permitidas na convenção interna "from_to".

        Convenção interna ("from_to"):
        M[i, j] = True  <=>  transição permitida i -> j, isto é, Pr(S_t = j | S_{t-1} = i).
        Portanto, quando essa máscara é usada em build_transition / filtro, espera-se matriz row-stochastic (linhas somam 1).

        Notação do paper/Doornik ("paper_pij"):
        p_{i|j} = Pr(S_t = i | S_{t-1} = j) (colunas somam 1).
        A máscara equivalente é M_pij = M.T.

        Triangularidade (triangular=True):
        Implementa monotonicidade não-decrescente do índice do regime ao longo do tempo:
        permite i -> j apenas se j >= i  (na convenção "from_to").
        Em termos de p_{i|j}, isso corresponde a permitir apenas i >= j (matriz "lower-triangular" no paper).

        Absorvência no último estado (absorbing_last=True):
        Força o último estado a ser absorvente na convenção "from_to": última linha = [0, ..., 0, 1].
        Em p_{i|j}, isso aparece como última coluna = [0, ..., 0, 1]^T.
        """
        """
        Constrói a máscara EFETIVA k×k combinando:
        - máscara explícita do usuário (user_mask), SE houver;
        - OU, se user_mask=None e structural_break='absorbing_block', deriva do spec;
        - aplica triangularidade (zera j<i) se triangular=True;
        - aplica 'último absorvente' se absorbing_last=True (linha final só coluna final).
        """
        # 1) ponto de partida
        if user_mask is None:
            if structural_break == "absorbing_block":
                block_spec = block_spec or {}
                pre   = int(block_spec.get("pre", max(1, k // 2)))
                post  = max(1, k - pre)
                bridge = block_spec.get("bridge", None)
                M = build_absorbing_block_mask(k, pre, post, bridge=bridge)
            else:
                M = np.ones((k, k), dtype=bool)
        else:
            M = np.asarray(user_mask, dtype=bool).copy()
            if M.shape != (k, k):
                raise ValueError(f"Máscara deve ser {k}x{k} (recebido {M.shape}).")

        # 2) triangularidade: proíbe retornos "para trás"
        if triangular:
            i, j = np.indices((k, k))
            # Na convenção interna ("from_to"), j >= i impede retornos para regimes de índice menor; no paper (p_{i|j}) isso é i >= j.
            M &= (j >= i)

        # 3) último absorvente: só permanece no último
        if absorbing_last:
            # Absorvente no último estado na convenção interna ("from_to"): última linha = e_k.
            M[-1, :]  = False
            M[-1, -1] = True
        return M

    def _prepare_effective_masks(self) -> None:
        km = getattr(self, "km", None)
        kv = getattr(self, "kv", None)
        if km is None or kv is None:
            raise RuntimeError("_prepare_effective_masks chamado antes de km/kv serem inicializados.")

        structural_break = getattr(self, "structural_break", None)

        abs_last_mean = bool(getattr(self, "absorbing_last_mean", False)) or (structural_break == "absorbing_last")
        abs_last_var  = bool(getattr(self, "absorbing_last_var",  False))

        self._effective_mask_mean = self._make_effective_mask(
            k=km,
            user_mask=getattr(self, "mask_mean", None),
            triangular=bool(getattr(self, "triangular_Pm", False)),
            absorbing_last=abs_last_mean,
            structural_break=structural_break,
            block_spec=getattr(self, "absorbing_block_spec", None),
        )

        self._effective_mask_var = self._make_effective_mask(
            k=kv,
            user_mask=getattr(self, "mask_var", None),
            triangular=bool(getattr(self, "triangular_Pv", False)),
            absorbing_last=abs_last_var,
            structural_break=None,
            block_spec=None,
        )

    # [DR] utilitário: empilha X, X_{-1}, ..., X_{-L}:
    def _stack_exog_lags(self, X: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if X is None:
            self.q = 0
            self.D = 0
            return None
        X = np.asarray(X, dtype=float)
        T, q = X.shape
        self.q = q
        L = max(0, self.Lx)
        blocks = []
        for ell in range(L + 1):
            # desloca para trás; preenche início com zeros
            pad = np.zeros((ell, q))
            blk = np.vstack([pad, X[:T-ell, :]])
            blocks.append(blk)
        Z = np.hstack(blocks)      # T x D, D = q*(L+1)
        self.D = q * (L + 1)
        return Z

    # ---------- Parameter packing / unpacking ----------
    def _pack(self, a, phi_u, theta_ma_u, beta, log_sigmasq, theta_m, theta_v):
        parts = [a.ravel(), phi_u.ravel()]
        if theta_ma_u is not None and theta_ma_u.size:
            parts.append(theta_ma_u.ravel())
        if beta is not None and beta.size:
            parts.append(beta.ravel())
        parts.extend([log_sigmasq.ravel(), theta_m.ravel(), theta_v.ravel()])
        return np.concatenate(parts)

    def _unpack(self, params):
        """
        Converte vetor de parâmetros livres em objetos do modelo,
        aplicando estabilizações e mapeamentos.
        """
        params = np.asarray(params, float).ravel()
        idx = 0

        # interceptos (a_i)
        a = params[idx: idx + self.km]; idx += self.km

        # AR
        if self.p > 0:
            if self.share_ar:
                raw = params[idx: idx + self.p]; idx += self.p
                phi = self._stabilize_ar_coefs(raw)
                phi = self._enforce_root_margin(phi, kind="ar", shrink=0.95, max_iter=60)
                Phi = np.tile(phi, (self.km, 1))
            else:
                raw = params[idx: idx + self.km * self.p]; idx += self.km * self.p
                Phi = np.zeros((self.km, self.p))
                for m in range(self.km):
                    phi_m = self._stabilize_ar_coefs(raw[m*self.p:(m+1)*self.p])
                    Phi[m] = self._enforce_root_margin(phi_m, kind="ar")
        else:
            Phi = np.zeros((self.km, 0))

        # MA
        if self.q_ma > 0:
            if self.share_ma:
                raw = params[idx: idx + self.q_ma]; idx += self.q_ma
                th = self._stabilize_ma_coefs(raw)
                th = self._enforce_root_margin(th, kind="ma", shrink=0.95, max_iter=60)
                Theta = np.tile(th, (self.km, 1))
            else:
                raw = params[idx: idx + self.km * self.q_ma]; idx += self.km * self.q_ma
                Theta = np.zeros((self.km, self.q_ma))
                for m in range(self.km):
                    th_m = self._stabilize_ma_coefs(raw[m*self.q_ma:(m+1)*self.q_ma])
                    Theta[m] = self._enforce_root_margin(th_m, kind="ma")
        else:
            Theta = np.zeros((self.km, 0))

        # DR exógeno (se houver)
        Beta = np.zeros((self.km, 0))
        if self.D > 0:
            if self.share_beta:
                raw = params[idx: idx + self.D]; idx += self.D
                Beta = np.tile(raw, (self.km, 1))
            else:
                raw = params[idx: idx + self.km * self.D]; idx += self.km * self.D
                Beta = raw.reshape(self.km, self.D)

        # variâncias (log)
        log_sigmasq = params[idx: idx + self.kv]; idx += self.kv

        # logits Pm, Pv
        theta_m = params[idx: idx + self.km * self.km]; idx += self.km * self.km
        theta_v = params[idx: idx + self.kv * self.kv]; idx += self.kv * self.kv

        # saneamento leve
        a = np.nan_to_num(a, 0.0)
        Phi = np.nan_to_num(Phi, 0.0)
        Theta = np.nan_to_num(Theta, 0.0)
        Beta = np.nan_to_num(Beta, 0.0)
        log_sigmasq = np.clip(np.nan_to_num(log_sigmasq, 0.0), -10.0, 10.0)
        return a, Phi, Theta, Beta, log_sigmasq, theta_m, theta_v

    def _init_params(self, y, Z=None):
        y = np.asarray(y, float).ravel()
        km, kv, p, q = self.km, self.kv, self.p, self.q_ma

        # --- Interceptos: espalha pelos quantis ---
        qs = np.linspace(0.1, 0.9, km)
        qy = np.quantile(y, qs) if y.size >= 5 else np.linspace(np.min(y), np.max(y), km)
        a0 = np.linspace(qy[0], qy[-1], km).astype(float)

        # --- AR/MA crus (tamanho depende de 'share_*') ---
        phi_u0 = np.zeros(p if self.share_ar else km * p, dtype=float)
        th0    = (np.zeros(q if self.share_ma else km * q, dtype=float) if q > 0
                else np.array([], dtype=float))

        # --- DR (se houver) ---
        if Z is None or self.D == 0:
            beta0 = np.array([], dtype=float)
        else:
            beta0 = np.zeros(self.D if self.share_beta else km * self.D, dtype=float)

        # --- Variâncias (log sigma^2) ---
        s2 = float(np.var(y - np.mean(y))) if y.size > 1 else 1.0
        s2 = max(s2, 1e-6)
        sig0 = np.log(np.linspace(0.7 * s2, 1.3 * s2, kv)).astype(float)

        # --- Máscaras efetivas (se não vierem, usa 'tudo permitido') ---
        mask_m = self.mask_mean
        mask_v = self.mask_var

        # Se o usuário optou por 'absorbing_block' e não passou máscara explícita, derive-a:
        if mask_m is None and getattr(self, "structural_break", None) == "absorbing_block":
            spec = getattr(self, "absorbing_block_spec", {}) or {}
            pre = int(spec.get("pre", max(1, km // 2)))
            post = max(1, km - pre)
            bridge = spec.get("bridge", None)
            mask_m = build_absorbing_block_mask(km, pre, post, bridge=bridge)

        # --- Ponte opcional (apenas para o chute inicial; a máscara continua mandando) ---
        bridge = None
        if hasattr(self, "absorbing_block_spec") and isinstance(self.absorbing_block_spec, dict):
            bridge = self.absorbing_block_spec.get("bridge", None)

        # --- Pm0 e Pv0 genéricos, coerentes com as máscaras e sem absorções ---
        Pm0 = self._build_seed_P_from_mask(km, mask_m, diag_p=0.85, bridge=bridge, bridge_weight=0.08)
        Pv0 = self._build_seed_P_from_mask(kv, mask_v, diag_p=0.90, bridge=None)

        # --- Converte para logits iniciais (respeitando bounds de theta_m/theta_v) ---
        theta_m0 = self._seed_transition_logits_from_P(Pm0, mask_m, clip_low=-4.0, clip_high=4.0)
        theta_v0 = self._seed_transition_logits_from_P(Pv0, mask_v,  clip_low=-4.0, clip_high=4.0)

        # --- Empacota no formato esperado ---
        return self._pack(a0, phi_u0, th0, beta0, sig0, theta_m0, theta_v0)

    # =======
    # Emissão
    # =======
    def _emission_ll(self, y_t, y_hist, eps_hist, x_tL, a, Phi, Theta, Beta, sigmasq):
        """
        log f(y_t | S^m=i, S^v=j):
        mean: a_i + (Phi_i @ y_hist) + (Theta_i @ eps_hist) + (Beta_i @ x_tL)
        var : sigma^2_j
        """
        km, kv = self.km, self.kv
        mu_by_mean = a + (Phi @ y_hist if self.p > 0 else 0.0)
        if self.q_ma > 0:
            mu_by_mean = mu_by_mean + (Theta @ eps_hist)
        if self.D > 0:
            mu_by_mean = mu_by_mean + (Beta @ x_tL)

        ll = np.empty(km * kv)
        idx = 0
        for iv in range(kv):
            var = float(max(sigmasq[iv], 1e-10))
            inv = 1.0 / var
            c = -0.5*np.log(2*np.pi*var)
            for im in range(km):
                err = y_t - mu_by_mean[im]
                ll[idx] = c - 0.5*err*err*inv
                idx += 1
        return ll, mu_by_mean

    # ==================
    # Filtro de Hamilton
    # ==================
    def _stationary(self, P: np.ndarray) -> np.ndarray:
        # distribuição estacionária por power-iteration
        k = P.shape[0]
        w = np.ones(k) / k
        for _ in range(512):
            w_next = w @ P
            if np.max(np.abs(w_next - w)) < 1e-12:
                break
            w = w_next
        w = np.maximum(w, 1e-15)
        return w / w.sum()

    def _arma_state_dims(self) -> Tuple[int, int, int]:
        """
        Estado exato para ARMA(p,q) com MA por regime.
        alpha_t = [y_t, y_{t-1}, ..., y_{t-p+1}, u_t, u_{t-1}, ..., u_{t-q+1}]
        onde u_t é a inovação gaussiana (com variância dependente do regime de variância).
        """
        p_y = max(int(self.p), 1)          # garante ao menos y_t no estado
        q_u = max(int(self.q_ma), 1)       # garante ao menos u_t no estado
        return p_y, q_u, p_y + q_u

    def _arma_transition_mats(self, m: int, Phi: np.ndarray, Theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Constrói (T_m, R) tal que:
        alpha_t = T_m alpha_{t-1} + d_t(m) + R u_t,   u_t ~ N(0, sigma^2_{v_t})

        alpha_t = [y_t, y_{t-1},..., y_{t-p_y+1}, u_t, u_{t-1},..., u_{t-q_u+1}]
        """
        p_y, q_u, n = self._arma_state_dims()
        p, q = int(self.p), int(self.q_ma)

        Tm = np.zeros((n, n), dtype=float)

        # y_t depende de y_{t-1..t-p} e u_{t-1..t-q} + u_t
        if p > 0:
            Tm[0, 0:p] = Phi[m]
        if q > 0:
            # em alpha_{t-1}, o bloco u começa em p_y e contém u_{t-1}, u_{t-2}, ...
            Tm[0, p_y:p_y+q] = Theta[m]

        # shift do bloco de y: y_{t-1} <- y_{t-1} etc.
        for i in range(1, p_y):
            Tm[i, i-1] = 1.0

        # shift do bloco u (exceto o primeiro, que é u_t novo)
        for j in range(1, q_u):
            Tm[p_y + j, p_y + j - 1] = 1.0

        # u_t entra em y_t (coef 1) e é armazenado no estado em alpha_t[p_y]
        R = np.zeros((n, 1), dtype=float)
        R[0, 0] = 1.0
        R[p_y, 0] = 1.0
        return Tm, R

    def _effective_theta_lags(self, m_tuple, Theta) -> np.ndarray:
        """
        Retorna vetor [θ1(m_{t-1}), θ2(m_{t-2}), ..., θq(m_{t-q})] para MA por regime defasado.

        Referência: em Marçal et al. (2025), Eq. (3), os termos MA são do tipo θ1(S_{m,t-1}) ε_{t-1}
        e θ2(S_{m,t-2}) ε_{t-2}.

        Observação: este helper deve ser usado apenas quando ma_regime_timing == 'lagged' e a cadeia de
        média estiver expandida (mean_decoder ativo). No caso padrão (ma_regime_timing == 'current'),
        os coeficientes MA já estão embutidos em Tm_list[m0] via _arma_transition_mats.
        """
        q = int(getattr(self, "q_ma", 0))
        if q <= 0:
            return np.zeros(0, dtype=float)

        m0 = int(m_tuple[0])
        out = np.empty(q, dtype=float)
        for j in range(1, q + 1):
            mj = int(m_tuple[j]) if j < len(m_tuple) else m0  # fallback seguro
            out[j - 1] = float(Theta[mj, j - 1])
        return out

    def _effective_d0(self, m_tuple, a, Phi, Beta=None, x_tL=None) -> float:
        """
        Implementa o centramento:
        y_t = a[m0] + sum_l Phi[m0,l-1]*(y_{t-l} - a[m_l]) + ... (sem a inovação)
        Logo, o termo determinístico em y_t é:
        d0 = a[m0] - sum_l Phi[m0,l-1]*a[m_l] + Beta[m0]@x_tL
        onde m_tuple = (m_t, m_{t-1}, ..., m_{t-L+1}).
        """
        m0 = int(m_tuple[0])
        d0 = float(a[m0])

        p = int(getattr(self, "p", 0))
        if p > 0 and getattr(self, "exact_mu_lags", False):
            for ell in range(1, p + 1):
                m_ell = int(m_tuple[ell]) if ell < len(m_tuple) else m0  # fallback seguro
                d0 -= float(Phi[m0, ell - 1]) * float(a[m_ell])

        if x_tL is not None and getattr(self, "D", 0) > 0 and Beta is not None and Beta.size:
            d0 += float(Beta[m0] @ x_tL)
        return d0

    def _normalize_mix_weights(self, alpha_prev: np.ndarray, P_col: np.ndarray) -> np.ndarray:
        """
        Retorna w_i ∝ alpha_prev[i] * P[i, s_cur], normalizado, de forma robusta.
        Preserva zeros estruturais: P_col == 0 -> peso 0.
        """
        # log(alpha_prev) assume alpha_prev > 0 (se você já aplica floor). Caso contrário:
        S = alpha_prev.shape[0]

        def _fallback_weights() -> np.ndarray:
            """
            Fallback conservador: evita introduzir massa em transições proibidas.
            Prioriza o suporte de (alpha_prev * P_col). Se degenerar, usa apenas o suporte permitido por P_col.
            """
            # 1) domínio linear preservando suporte (alpha_prev * P_col)
            a = np.nan_to_num(alpha_prev, nan=0.0, posinf=0.0, neginf=0.0)
            p = np.nan_to_num(P_col,      nan=0.0, posinf=0.0, neginf=0.0)
            w_lin = np.maximum(a, 0.0) * np.maximum(p, 0.0)
            s_lin = float(w_lin.sum())
            if np.isfinite(s_lin) and s_lin > 0.0:
                return w_lin / s_lin

            # 2) uniforme SOMENTE no suporte permitido por P_col e alpha_prev>0
            m = (P_col > 0.0) & np.isfinite(alpha_prev) & (alpha_prev > 0.0)
            cnt = int(m.sum())
            if cnt > 0:
                out = np.zeros(S, dtype=float)
                out[m] = 1.0 / cnt
                return out

            # 3) normaliza alpha_prev (sem criar massa onde alpha_prev==0)
            a2 = np.nan_to_num(np.maximum(alpha_prev, 0.0), nan=0.0, posinf=0.0, neginf=0.0)
            s2 = float(a2.sum())
            if np.isfinite(s2) and s2 > 0.0:
                return a2 / s2

            # 4) último recurso: uniforme global
            return np.full(S, 1.0 / S, dtype=float)

        log_alpha = np.full_like(alpha_prev, -np.inf, dtype=float)
        mA = alpha_prev > 0.0
        log_alpha[mA] = np.log(alpha_prev[mA])

        logPcol = np.full_like(P_col, -np.inf, dtype=float)
        mP = P_col > 0.0
        logPcol[mP] = np.log(P_col[mP])

        logw = log_alpha + logPcol
        c = logsumexp(logw)

        if not np.isfinite(c):
            return _fallback_weights()

        w = np.exp(logw - c)
        s = float(w.sum())
        if (not np.isfinite(s)) or (s <= 0.0):
            return _fallback_weights()
        return w / s

    def _init_alpha_prev(self, Pm, Pv, S, mean_decoder=None):
        """
        Inicialização da distribuição discreta α_{t-1}.
        - Sem quebra estrutural: 'auto' usa estacionária; 'uniform' usa uniforme.
        - Com quebra estrutural (absorvente): 'auto' e 'uniform' concentram massa no suporte pré-quebra
            (sem massa inicial em estados pós-quebra), evitando viés imediato no filtro.
        """
        method = getattr(self, "init_xi", "auto")
        structural_break = getattr(self, "structural_break", None)

        km_chain = int(np.asarray(Pm).shape[0])
        kv = int(np.asarray(Pv).shape[0])
        if S != km_chain * kv:
            # mantém robustez se S vier de fora com inconsistência
            S = km_chain * kv

        # 1) modo estritamente estacionário (sem restrições de quebra)
        if method == "stationary":
            pi_m = self._stationary(Pm).reshape(-1)
            pi_v = self._stationary(Pv).reshape(-1)
            alpha0 = np.kron(pi_v.reshape(1, -1), pi_m.reshape(1, -1)).ravel()
            alpha0 = np.maximum(alpha0, 1e-300)
            return alpha0 / alpha0.sum()

        # 2) sem quebra estrutural: comportamento padrão
        if structural_break is None:
            if method == "uniform":
                alpha0 = np.full(S, 1.0 / S, dtype=float)
            else:  # 'auto'
                pi_m = self._stationary(Pm).reshape(-1)
                pi_v = self._stationary(Pv).reshape(-1)
                alpha0 = np.kron(pi_v.reshape(1, -1), pi_m.reshape(1, -1)).ravel()
            alpha0 = np.maximum(alpha0, 1e-300)
            return alpha0 / alpha0.sum()

        # 3) com quebra estrutural: restringe suporte do bloco de média ao pré-quebra
        allowed_m = None
        if structural_break == "absorbing_block":
            spec = getattr(self, "absorbing_block_spec", None) or {}
            pre = int(spec.get("pre", max(1, int(getattr(self, "km", km_chain)) // 2)))
            if mean_decoder is None:
                allowed_m = np.arange(min(pre, km_chain), dtype=int)
            else:
                # preferencial: todos os componentes de m_tuple devem estar no bloco pré-quebra
                tmp = []
                for idx, tup in enumerate(mean_decoder):
                    try:
                        comps = [int(x) for x in tup]
                    except Exception:
                        comps = [int(tup[0])]
                    if all(c < pre for c in comps):
                        tmp.append(idx)
                if len(tmp) == 0:
                    # fallback: pelo menos o estado corrente (m_t) no pré-quebra
                    for idx, tup in enumerate(mean_decoder):
                        if int(tup[0]) < pre:
                            tmp.append(idx)
                allowed_m = np.asarray(tmp, dtype=int)

        elif structural_break == "absorbing_last":
            last_base = int(getattr(self, "km", km_chain)) - 1
            if mean_decoder is None:
                allowed_m = np.arange(max(km_chain - 1, 1), dtype=int)
            else:
                tmp = []
                for idx, tup in enumerate(mean_decoder):
                    try:
                        comps = [int(x) for x in tup]
                    except Exception:
                        comps = [int(tup[0])]
                    if all(c != last_base for c in comps):
                        tmp.append(idx)
                if len(tmp) == 0:
                    for idx, tup in enumerate(mean_decoder):
                        if int(tup[0]) != last_base:
                            tmp.append(idx)
                allowed_m = np.asarray(tmp, dtype=int)

        # se a quebra não for reconhecida, volta ao padrão 'auto' (estacionária) / 'uniform'
        if allowed_m is None or allowed_m.size == 0:
            if method == "uniform":
                alpha0 = np.full(S, 1.0 / S, dtype=float)
            else:
                pi_m = self._stationary(Pm).reshape(-1)
                pi_v = self._stationary(Pv).reshape(-1)
                alpha0 = np.kron(pi_v.reshape(1, -1), pi_m.reshape(1, -1)).ravel()
            alpha0 = np.maximum(alpha0, 1e-300)
            return alpha0 / alpha0.sum()

        # monta π_m restrita ao suporte permitido; π_v segue (auto: estacionária; uniform: uniforme)
        pi_m = np.zeros(km_chain, dtype=float)
        pi_m[allowed_m] = 1.0 / float(allowed_m.size)

        if method == "uniform":
            pi_v = np.full(kv, 1.0 / kv, dtype=float)
        else:  # 'auto'
            pi_v = self._stationary(Pv).reshape(-1)

        alpha0 = np.kron(pi_v.reshape(1, -1), pi_m.reshape(1, -1)).ravel()
        # IMPORTANT (ID 6): preserve exact zeros outside the allowed support.
        pos = alpha0 > 0.0
        if np.any(pos):
            alpha0[pos] = np.maximum(alpha0[pos], 1e-300)
        s = float(alpha0.sum())
        if (not np.isfinite(s)) or s <= 0.0:
            raise ValueError("_init_alpha_prev: alpha0 inválida após restrição de suporte")
        return alpha0 / s

    def _filter(self, y, Pm, Pv, a, Phi, Theta, Beta, sigmasq, Z, mean_decoder=None):
        """
        Hamilton (discreto) + Kalman (contínuo) no esquema de Kim (1994),
        com representação exata do ARMA(p,q) no estado.
        ---
        Discreto: s_t = (m_t, v_t), transição P = kron(Pv, Pm)
        Contínuo: alpha_t contém y/lags e u/lags (u_t é a inovação).
        """
        y = np.asarray(y, float).ravel()
        Tobs = len(y)
        km_base = self.km
        kv = self.kv
        km_chain = int(Pm.shape[0])
        S = km_chain * kv
        P = kron(Pv, Pm)

        # log(P) pré-computado para pesos de mistura via log-sum-exp (evita underflow em α_{t-1}·P)
        logP = np.full_like(P, -np.inf, dtype=float)
        mP = P > 0.0
        logP[mP] = np.log(P[mP])

        # Pré-cálculo opcional para colapso esparso (reduz custo quando P tem muitos zeros)
        # Obs.: se P for densa, o caminho vetorizado (Wn + einsum) tende a ser mais rápido.
        _use_sparse_collapse = False
        _P_support = None
        _nnzP = int(mP.sum())
        if _nnzP < int(0.65 * P.size):
            _use_sparse_collapse = True
            _P_support = [np.flatnonzero(mP[:, j]) for j in range(S)]

        # inicial discreto (α_{t-1}): ver _init_alpha_prev (inclui suporte pré-quebra quando aplicável)
        alpha_prev = self._init_alpha_prev(Pm, Pv, S, mean_decoder=mean_decoder)

        # estado ARMA exato
        p_y, q_u, n = self._arma_state_dims()

        # inicialização difusa do estado contínuo (por estado discreto)
        yvar = float(np.nanvar(y)) if np.isfinite(np.nanvar(y)) else 1.0
        P0 = np.eye(n) * max(10.0 * yvar, 1.0)
        a_prev_cont = np.zeros((S, n), dtype=float)
        P_prev_cont = np.repeat(P0[None, :, :], S, axis=0)

        # pré-computa T_m e RR' (para Q_v = sigma2_v * RR')
        Tm_list = []
        R0 = None
        for m in range(km_base):
            Tm, R = self._arma_transition_mats(m, Phi, Theta)
            Tm_list.append(Tm)
            if R0 is None:
                R0 = R
        RR = R0 @ R0.T

        # Pré-computa Q_v = sigma2_v * RR (evita multiplicação de matriz por escalar dentro do loop)
        sig2 = np.maximum(np.asarray(sigmasq, dtype=float).ravel(), 1e-12)
        Qv_list = [float(sig2[v]) * RR for v in range(kv)]

        # parâmetros fixos (para evitar getattr repetido em loops internos)
        q_ma = int(getattr(self, "q_ma", 0))
        use_lagged_ma = (q_ma > 0) and (mean_decoder is not None) and (getattr(self, "ma_regime_timing", "current") == "lagged")


        filtered = np.zeros((Tobs, S), dtype=float)
        fitted = np.full(Tobs, np.nan, dtype=float)

        loglike = 0.0
        self._rescue_count = 0
        # diagnóstico do "resgate" (capado): lista de pequenos snapshots por iteração do filtro
        self._rescue_info = []

        # Buffers pré-alocados (evita alocações por t). São sobrescritos integralmente a cada iteração.
        a_pred_cont = np.empty_like(a_prev_cont)
        P_pred_cont = np.empty_like(P_prev_cont)
        a_filt_cont = np.empty_like(a_prev_cont)
        P_filt_cont = np.empty_like(P_prev_cont)
        y_pred = np.empty(S, dtype=float)
        ll = np.empty(S, dtype=float)

        for t in range(Tobs):
            y_t = y[t]

            x_tL = None
            if Z is not None and getattr(self, "D", 0) > 0:
                x_tL = Z[t]

            # predição discreta
            alpha_pred = P.T @ alpha_prev
            # Preservar zeros estruturais (estados inalcançáveis devem permanecer com prob.=0).
            # Evitar floor global que injeta massa em estados proibidos pela máscara/transições.
            alpha_pred = np.clip(alpha_pred, 0.0, None)
            s = float(alpha_pred.sum())
            if (not np.isfinite(s)) or (s <= 0.0):
                # Massa inválida na predição (sugere P/máscara inconsistente). Não injeta uniforme.
                # Fallback conservador: mantém suporte do passo anterior.
                self._rescue_count += 1
                alpha_pred = alpha_prev.copy()
            else:
                alpha_pred = alpha_pred / s

            # buffers do passo t: pré-alocados fora do loop (a_pred_cont, P_pred_cont, a_filt_cont, P_filt_cont, y_pred, ll)

            # --- Pré-cálculo do colapso (moment matching) para todos os s_cur ---
            # w_i(s_cur) ∝ α_{t-1}(i) · P(i, s_cur), normalizado via log-sum-exp (estável)
            log_alpha = np.full_like(alpha_prev, -np.inf, dtype=float)
            mA = alpha_prev > 0.0
            log_alpha[mA] = np.log(alpha_prev[mA])

            if not _use_sparse_collapse:
                # Caminho vetorizado (denso): O(S^2) por t, mas geralmente mais rápido quando P é densa.
                logw = log_alpha[:, None] + logP
                c = sp_logsumexp(logw, axis=0)  # (S,)
                Wn = np.zeros_like(P, dtype=float)
                good = np.isfinite(c)
                if np.any(good):
                    Wn[:, good] = np.exp(logw[:, good] - c[good])
                    ss = Wn[:, good].sum(axis=0)
                    ss = np.maximum(ss, 1e-300)
                    Wn[:, good] /= ss

                # fallback conservador para colunas degeneradas
                if not np.all(good):
                    for j in np.where(~good)[0]:
                        Wn[:, j] = self._normalize_mix_weights(alpha_prev, P[:, j])

                mean_prev_all = Wn.T @ a_prev_cont  # (S, n)

                E_P_all = np.einsum('ij,iab->jab', Wn, P_prev_cont, optimize=True)  # (S,n,n)
                E_aa_all = np.einsum('ij,ia,ib->jab', Wn, a_prev_cont, a_prev_cont, optimize=True)  # (S,n,n)

            else:
                # Caminho esparso: O(nnz(P)) por t (em termos de pesos), útil quando P tem muitos zeros.
                mean_prev_all = np.empty((S, n), dtype=float)
                E_P_all = np.empty((S, n, n), dtype=float)
                E_aa_all = np.empty((S, n, n), dtype=float)

                for j in range(S):
                    idx = _P_support[j]
                    if idx.size == 0:
                        w_full = self._normalize_mix_weights(alpha_prev, P[:, j])
                        mean_prev_all[j] = w_full @ a_prev_cont
                        E_P_all[j] = np.einsum('i,iab->ab', w_full, P_prev_cont, optimize=True)
                        E_aa_all[j] = np.einsum('i,ia,ib->ab', w_full, a_prev_cont, a_prev_cont, optimize=True)
                        continue

                    logw_j = log_alpha[idx] + logP[idx, j]
                    c_j = sp_logsumexp(logw_j)
                    if not np.isfinite(c_j):
                        w_full = self._normalize_mix_weights(alpha_prev, P[:, j])
                        mean_prev_all[j] = w_full @ a_prev_cont
                        E_P_all[j] = np.einsum('i,iab->ab', w_full, P_prev_cont, optimize=True)
                        E_aa_all[j] = np.einsum('i,ia,ib->ab', w_full, a_prev_cont, a_prev_cont, optimize=True)
                        continue

                    w = np.exp(logw_j - c_j)
                    ss = float(w.sum())
                    ss = max(ss, 1e-300)
                    w /= ss

                    a_sel = a_prev_cont[idx]
                    mean_prev_all[j] = w @ a_sel
                    E_P_all[j] = np.einsum('i,iab->ab', w, P_prev_cont[idx], optimize=True)
                    E_aa_all[j] = np.einsum('i,ia,ib->ab', w, a_sel, a_sel, optimize=True)

            cov_prev_all = E_P_all + E_aa_all - (mean_prev_all[:, :, None] * mean_prev_all[:, None, :])
            cov_prev_all = 0.5 * (cov_prev_all + cov_prev_all.transpose(0, 2, 1))

            for s_cur in range(S):
                m_state  = s_cur % km_chain
                v = s_cur // km_chain

                if mean_decoder is None:
                    m_tuple = (int(m_state),)
                else:
                    m_tuple = mean_decoder[int(m_state)]

                m0 = int(m_tuple[0])
                Tm_base = Tm_list[m0]

                # Coeficientes MA:
                # - 'current': θ_j(S_{m,t}) já embutido em Tm_list[m0] via _arma_transition_mats.
                # - 'lagged' : θ_j(S_{m,t-j}) (Marçal et al., 2025, Eq. (3)); requer mean_decoder (cadeia expandida).
                #
                # Nota: para 'lagged', NÃO podemos mutar Tm_list in-place, pois θ depende de (m_{t-1},...,m_{t-q}).
                if use_lagged_ma:
                    p_y, q_u, n = self._arma_state_dims()
                    Tm = Tm_base.copy()
                    theta_lags = self._effective_theta_lags(m_tuple, Theta)  # [θ1(m_{t-1}), θ2(m_{t-2}), ...]
                    Tm[0, p_y:p_y+q_ma] = theta_lags
                else:
                    Tm = Tm_base

                # termo determinístico entra em y_t (1º componente do estado)
                d0 = self._effective_d0(m_tuple, a, Phi, Beta=Beta, x_tL=x_tL)

                Q = Qv_list[v]

                # colapso em t-1 (moment matching) pré-computado para todos os s_cur neste t
                mean_prev = mean_prev_all[s_cur]   # (n,)
                cov_prev = cov_prev_all[s_cur]     # (n,n)

                # predição contínua sob (m,v)
                mean_pred = Tm @ mean_prev
                mean_pred[0] += float(d0)
                cov_pred = Tm @ cov_prev @ Tm.T + Q
                cov_pred = 0.5 * (cov_pred + cov_pred.T)
                cov_pred.flat[:: n + 1] = np.maximum(cov_pred.diagonal(), 1e-12)

                a_pred_cont[s_cur] = mean_pred
                P_pred_cont[s_cur] = cov_pred

                yhat = float(mean_pred[0])
                y_pred[s_cur] = yhat

                F = float(cov_pred[0, 0])
                F = max(F, max(1e-8, 1e-8 * yvar))

                if not np.isfinite(y_t):
                    ll[s_cur] = 0.0
                    a_filt_cont[s_cur] = mean_pred
                    P_filt_cont[s_cur] = cov_pred
                else:
                    v_t = float(y_t - yhat)
                    ll[s_cur] = -0.5 * (np.log(2.0*np.pi) + np.log(F) + (v_t*v_t)/F)

                    # update Kalman (H=0, observação é o 1º componente)
                    K = cov_pred[:, 0] / F
                    mean_upd = mean_pred + K * v_t
                    cov_upd = cov_pred - np.outer(K, K) * F
                    cov_upd = 0.5 * (cov_upd + cov_upd.T)
                    cov_upd.flat[:: n + 1] = np.maximum(cov_upd.diagonal(), 1e-12)

                    a_filt_cont[s_cur] = mean_upd
                    P_filt_cont[s_cur] = cov_upd

            # one-step-ahead fitted (antes de ver y_t)
            fitted[t] = float(alpha_pred @ y_pred)

            # se y_t faltante, não atualiza discreto
            if not np.isfinite(y_t):
                filtered[t] = alpha_pred
                alpha_prev = alpha_pred
                # swap: próxima iteração usa a predição como "prev"
                a_prev_cont, a_pred_cont = a_pred_cont, a_prev_cont
                P_prev_cont, P_pred_cont = P_pred_cont, P_prev_cont
                continue

            # update discreto
            # log(alpha_pred) respeitando zeros estruturais (evita warning e mantém -inf onde alpha_pred==0)
            log_alpha_pred = np.full_like(alpha_pred, -np.inf, dtype=float)
            mAp = alpha_pred > 0.0
            log_alpha_pred[mAp] = np.log(alpha_pred[mAp])
            logw = log_alpha_pred + ll
            c = logsumexp(logw)
            if not np.isfinite(c):
                self._rescue_count += 1
                # guarda um snapshot leve para diagnóstico (no máximo 20 ocorrências)
                if len(self._rescue_info) < 20:
                    self._rescue_info.append({
                        "t": int(t),
                        "y_t": float(y_t),
                        "finite_logw": int(np.isfinite(logw).sum()),
                        "finite_ll": int(np.isfinite(ll).sum()),
                        "alpha_pred_nonzero": int((alpha_pred > 0.0).sum()),
                    })                
                filtered[t] = alpha_pred
                alpha_prev = alpha_pred
                # swap: mantém coerência sem alocações novas
                a_prev_cont, a_pred_cont = a_pred_cont, a_prev_cont
                P_prev_cont, P_pred_cont = P_pred_cont, P_prev_cont
                # penalização finita "principiada": aproxima colapso (densidade ~ 0) por piso numérico
                # escala por Tobs para manter magnitude comparável ao total da log-verossimilhança
                loglike += float(Tobs) * float(np.log(np.finfo(float).tiny))
                continue

            alpha = np.exp(logw - c)
            # Não aplicar floor global: preserva zeros estruturais (estados/transições impossíveis)
            s_alpha = float(alpha.sum())
            if not (np.isfinite(s_alpha) and s_alpha > 0.0):
                self._rescue_count += 1
                alpha = alpha_pred
            else:
                alpha = alpha / s_alpha

            filtered[t] = alpha
            alpha_prev = alpha
            # swap: estado filtrado vira "prev" na próxima iteração
            a_prev_cont, a_filt_cont = a_filt_cont, a_prev_cont
            P_prev_cont, P_filt_cont = P_filt_cont, P_prev_cont
            loglike += float(c)
        return float(loglike), filtered, fitted

    # =================
    # Suavização de Kim
    # =================
    def _smooth(self, filtered, Pm, Pv, return_pair: bool = False):
        T, S = filtered.shape
        P = kron(Pv, Pm)

        smoothed = np.zeros_like(filtered)
        smoothed[-1] = filtered[-1]

        pair = None
        if return_pair:
            pair = np.zeros((T-1, S, S), dtype=float)

        for t in range(T-2, -1, -1):
            # ξ_{t+1|t} = P' ξ_{t|t} na convenção do seu filtro
            denom = P.T @ filtered[t]
            # denom[j] = Pr(S_{t+1}=j | F_t). Se denom[j]==0, o estado j é inalcançável em t+1 dado F_t.
            # O Kim smoother exige razão apenas no suporte alcançável para preservar zeros estruturais.
            valid = np.isfinite(denom) & (denom > 0.0)
            if not np.any(valid):
                # fallback numérico: mantém a distribuição filtrada se nada é alcançável
                smoothed[t] = filtered[t]
                if return_pair:
                    pair[t] = 0.0
                continue

            sm_next = smoothed[t+1]
            # resgate: se houver massa suavizada em estados inalcançáveis, zere e renormalize no suporte válido
            bad_mass = float(np.sum(sm_next[~valid]))
            if bad_mass > 0.0:
                self._rescue_count += 1
                sm_next = sm_next.copy()
                sm_next[~valid] = 0.0
                s_next = float(np.sum(sm_next))
                if np.isfinite(s_next) and s_next > 0.0:
                    sm_next = sm_next / s_next
                else:
                    # fallback: usa a previsão one-step-ahead como distribuição em t+1
                    sm_next = denom / float(np.sum(denom))

            ratio = np.zeros_like(denom)
            ratio[valid] = sm_next[valid] / denom[valid]          # elementwise (0 onde denom==0)
            beta  = P @ ratio                              # vetor S

            st = filtered[t] * beta
            s = float(np.sum(st))

            if return_pair:
                # pair[t,i,j] = ξ_{t|t}(i) * P[i,j] * ratio[j]
                # Em aritmética exata: sum(M)=sum(st)=1.0. Diferenças são puramente numéricas.
                M = (filtered[t][:, None] * P) * (ratio[None, :])
                M = np.where(np.isfinite(M) & (M > 0.0), M, 0.0)
                Ms = float(np.sum(M))

                # Normalizador único para manter coerência entre smoothed[t] e pair[t]
                norm = Ms if (np.isfinite(Ms) and Ms > 0.0) else s
                if (not np.isfinite(norm)) or (norm <= 0.0):
                    smoothed[t] = filtered[t]
                    pair[t] = 0.0
                else:
                    smoothed[t] = st / norm
                    # Evite renormalização desnecessária; normalização escalar não altera suporte
                    pair[t] = M if abs(Ms - 1.0) <= 1e-12 else (M / norm)
            else:
                smoothed[t] = filtered[t] if s <= 0.0 else (st / s)

        return (smoothed, pair) if return_pair else smoothed

    def _build_Pm(self, theta_m):
        if self._effective_mask_mean is None:
            self._prepare_effective_masks()
        mask_eff = self._effective_mask_mean
        return build_transition(theta_m, mask=mask_eff,
                                absorbing_last=False,
                                triangular=False)

    # fazer um _build_Pv análogo aqui depois se necessário
    def _build_Pv(self, theta_v):
        if self._effective_mask_var is None:
            self._prepare_effective_masks()
        mask_eff = self._effective_mask_var
        return build_transition(theta_v, mask=mask_eff,
                                absorbing_last=False,
                                triangular=False)

    # --- Penalização suave perto da borda da estacionariedade/invertibilidade --- 
    def _ar_roots(self, phi_row):
        """Roots of A(z)=1 - phi1 z - ... - phip z^p."""
        phi_row = np.asarray(phi_row, float).ravel()
        if phi_row.size == 0:
            return np.array([np.inf])
        # A(z)=1 - phi1 z - ... - phip z^p  => coefficients (highest power first):
        # [-phip, ..., -phi1, 1]
        coefs = np.r_[-phi_row[::-1], 1.0]
        return np.roots(coefs)

    def _ma_roots(self, th_row):
        """Roots of B(z)=1 + theta1 z + ... + thetaq z^q."""
        th_row = np.asarray(th_row, float).ravel()
        if th_row.size == 0:
            return np.array([np.inf])
        # B(z)=1 + theta1 z + ... + thetaq z^q => [thetaq, ..., theta1, 1]
        coefs = np.r_[th_row[::-1], 1.0]
        return np.roots(coefs)

    def _enforce_invertible_ma_by_roots(self, theta: np.ndarray) -> np.ndarray:
        """Enforce invertibility of MA polynomial B(z)=1+θ1 z+...+θq z^q by root reflection.

        This follows the convention used in `_ma_roots`. Any root with |root| < target
        (target = self._root_target() = 1+margin) is reflected outside the circle of radius `target`:
            r_new = target^2 / conj(r)

        The polynomial is then rescaled to keep the constant term equal to 1 (B(0)=1), and the
        MA coefficients θ are extracted from the ascending representation.

        Note: reflection is not perfectly smooth at the boundary, but it is robust and widely used.
        """
        theta = np.asarray(theta, float).ravel()
        if theta.size == 0:
            return np.zeros(0, dtype=float)

        # Degenerate constant MA polynomial: if all MA coefficients are ~0 then B(z)=1
        # has no roots; numpy returns roots=[] and poly([]) is scalar. Nothing to enforce.
        if np.max(np.abs(theta)) < 1e-14:
            return theta

        target = float(self._root_target())
        roots = self._ma_roots(theta)
        roots = np.asarray(roots, complex)

        if roots.size == 0:
            return theta

        absr = np.abs(roots)
        bad = np.isfinite(absr) & (absr < target)
        if np.any(bad):
            roots = roots.copy()
            roots[bad] = (target ** 2) / np.conj(roots[bad])

        # Rebuild polynomial from (possibly reflected) roots.
        # np.poly gives monic coefficients for z^q + ... + c0. We rescale to enforce c0=1.
        poly_desc = np.poly(roots)  # descending, leading coefficient 1
        c0 = poly_desc[-1]
        if (not np.isfinite(c0)) or (np.abs(c0) < 1e-14):
            # fallback conservador: mantém theta original
            return theta

        poly_asc = poly_desc[::-1] / c0
        theta_new = poly_asc[1:]

        # Keep the original MA order even if numpy trimmed leading zero coefficients.
        q = theta.size
        if theta_new.size != q:
            if theta_new.size != q:
                theta_new = np.r_[theta_new, np.zeros(q - theta_new.size)]
            else:
                theta_new = theta_new[:q]

        theta_new = np.real_if_close(theta_new, tol=1000)
        return np.asarray(theta_new, float)

    def _root_target(self) -> float:
        """Alvo efetivo para estabilidade/invertibilidade: 1 + max(margin, eps)."""
        margin = float(getattr(self, "unit_root_margin", 0.0))
        eps    = float(getattr(self, "unit_root_eps", 1e-6))
        return 1.0 + max(margin, eps)

    def _penalty_near_unit_roots(self, Phi, Theta, w_ar=5e2, w_ma=5e3):
        pen = 0.0
        # target = 1.0 + float(getattr(self, "unit_root_margin", 0.0), 1e-6)
        target = self._root_target()

        for m in range(self.km):
            if self.p > 0:
                r = self._ar_roots(Phi[m])
                min_abs = np.inf if r.size == 0 else float(np.min(np.abs(r)))
                if min_abs <= target:
                    pen += w_ar * (target - min_abs)**2

            if self.q_ma > 0:
                r = self._ma_roots(Theta[m])
                min_abs = np.inf if r.size == 0 else float(np.min(np.abs(r)))
                if min_abs <= target:
                    pen += w_ma * (target - min_abs)**2
        return pen

    def fit(self, y, X: Optional[np.ndarray] = None,
            maxiter=500, tol=1e-6, method="L-BFGS-B",
            verbose=False, n_starts: int = 5, random_state: Optional[int] = None) -> MSCompResults:
        """
        Estima o MSMVC por Máxima Verossimilhança (Hamilton + Kim), agora robusto a NaN/Inf:
        - trata res.fun não finito,
        - aplica fallback para garantir best['x'] válido,
        - valida máscaras de transição antes da otimização.
        """

        y = np.asarray(y, float).ravel()

        # --- DR exógeno (se houver): empilha lags de X; se X=None => desliga DR (D=0) ---
        self._Z = self._stack_exog_lags(X)
        #self._Z = X

        self._prepare_effective_masks()

        # Vetor de bounds:
        def _bounds_vector():
            B = []
            # a (interceptos por regime de média)
            B += [(-10.0, 10.0)] * self.km

            # AR (parâmetros "crus" que serão estabilizados via PACF)
            if self.p > 0:
                if self.share_ar:
                    B += [(-3.5, 3.5)] * self.p
                else:
                    B += [(-3.5, 3.5)] * (self.km * self.p)

            # MA (parâmetros "crus" que serão estabilizados via PACF)
            if self.q_ma > 0:
                if self.share_ma:
                    B += [(-3.5, 3.5)] * self.q_ma
                else:
                    B += [(-3.5, 3.5)] * (self.km * self.q_ma)

            # DR (se houver)
            if self.D > 0:
                if self.share_beta:
                    B += [(-5.0, 5.0)] * self.D
                else:
                    B += [(-5.0, 5.0)] * (self.km * self.D)

            # log variâncias (sempre positivas após exp) — limitar escala numérica
            B += [(-10.0, 10.0)] * self.kv

            # logits de Pm e Pv (theta_m, theta_v) — limitar range numérico do softmax
            B += [(-4.0, 4.0)] * (self.km * self.km)
            B += [(-4.0, 4.0)] * (self.kv * self.kv)
            return B

        bounds = _bounds_vector()

        # utilitários para respeitar bounds nos chutes
        def _clip_to_bounds(x):
            lo = np.array([b[0] for b in bounds], float)
            hi = np.array([b[1] for b in bounds], float)
            return np.minimum(np.maximum(x, lo), hi)

        rng = np.random.default_rng(self.random_state if random_state is None else random_state)

        def _jitter(x, scale=0.15):
            lo = np.array([b[0] for b in bounds], float)
            hi = np.array([b[1] for b in bounds], float)
            span = hi - lo
            z = x + rng.normal(scale=scale * span, size=x.shape)
            return _clip_to_bounds(z)

        def _penalty_transitions(P, mask=None, lam=50.0, eps=1e-6):
            if mask is None:
                mask = np.ones_like(P, dtype=bool)
            Pm = np.clip(P[mask], eps, 1.0 - eps)
            # CE simétrica: -log(P) - log(1-P) evita colar em 0 ou 1
            return lam * ( -np.log(Pm).sum() - np.log(1.0 - Pm).sum() )

        # --- Função objetivo: -logverossimilhança penalizada ---
        def _negloglike(th):
            a, Phi, Theta, Beta, log_sigmasq, theta_m, theta_v = self._unpack(th)
            sigmasq = np.exp(log_sigmasq)

            Pm = self._build_Pm(theta_m)
            Pv = build_transition(theta_v, mask=self.mask_var,
                                absorbing_last=self.absorbing_last_var,
                                triangular=self.triangular_Pv)

            if self._needs_expanded_mean_chain():
                Pm_eff, dec = self._expand_mean_chain(Pm)
                # P = kron(Pv, Pm_exp)
            else:
                Pm_eff, dec = Pm, None
            ll, _, _ = self._filter(y, Pm_eff, Pv, a, Phi, Theta, Beta, sigmasq, self._Z, mean_decoder=dec)

            # penalização suave perto da borda
            pen_roots = self._penalty_near_unit_roots(Phi, Theta, w_ar=40, w_ma=200)

            pen_Pm = 0
            pen_Pv = 0
            return -ll + pen_roots + pen_Pm + pen_Pv

        # --- Validação das máscaras (evita "linhas mortas" em Pm/Pv) ---
        def _validate_mask(mask, name, k):
            if mask is None:
                return
            m = np.asarray(mask, dtype=bool)
            if m.shape != (k, k):
                raise ValueError(f"{name} deve ser {k}x{k} (recebido {m.shape}).")
            row_ok = m.sum(axis=1) >= 1
            if not np.all(row_ok):
                bad = np.where(~row_ok)[0].tolist()
                raise ValueError(f"{name}: linha(s) sem transições permitidas: {bad}.")
        _validate_mask(self.mask_mean, "mask_mean", self.km)
        _validate_mask(self.mask_var,  "mask_var",  self.kv)

        # --- Otimização com fallback robusto ---
        best     = dict(fun=np.inf, x=None, res=None)
        last_res = None
        last_x0  = None

        # chute base + jitters
        x0_base = self._init_params(y, Z=self._Z)
        x0_base = _clip_to_bounds(x0_base)                     # ### CHANGED
        starts = [x0_base] + [_jitter(x0_base, scale=0.10) for _ in range(max(0, n_starts - 1))]

        for x0 in starts:
            # (opcional) jitter leve extra nas transições pode permanecer, mas já temos jitter global
            try:
                km2, kv2 = self.km*self.km, self.kv*self.kv
                x0[-kv2:] += rng.normal(scale=0.10, size=kv2)           # θ_v
                x0[-(km2+kv2):-kv2] += rng.normal(scale=0.10, size=km2) # θ_m
                x0 = _clip_to_bounds(x0)                                # ### CHANGED
            except Exception:
                pass

            last_x0 = x0
            res = minimize(_negloglike, x0, method=method,               # ### CHANGED
                        bounds=bounds,                                # passa BOUNDS
                        options={"maxiter": maxiter, "disp": verbose, "gtol": tol})
            last_res = res

            f = float(res.fun) if np.isfinite(res.fun) else np.inf
            if f < best["fun"]:
                best = dict(fun=f, x=res.x.copy(), res=res)

        # --- Fallback garantido: best['x'] nunca fica None ---
        if best["x"] is None:
            fallback = (getattr(last_res, "x", None) if last_res is not None else None)
            if fallback is None:
                fallback = last_x0
            if fallback is None:
                fallback = self._init_params(y, Z=self._Z)
            best = dict(fun=np.inf, x=fallback, res=last_res)

        # --- Reconstrói parâmetros e matrizes de transição ---
        a, Phi, Theta, Beta, log_sigmasq, theta_m, theta_v = self._unpack(best["x"])
        sigmasq = np.exp(log_sigmasq)

        Pm = self._build_Pm(theta_m)
        Pv = build_transition(theta_v, mask=self._effective_mask_var,
                            absorbing_last=self.absorbing_last_var,
                            triangular=self.triangular_Pv)

        if self._needs_expanded_mean_chain():
            Pm_exp, self._mean_decoder = self._expand_mean_chain(Pm)
            P = kron(Pv, Pm_exp)
            self._use_expanded_mean = True
        else:
            P = kron(Pv, Pm)
            self._mean_decoder = [(m,) for m in range(self.km)]
            self._use_expanded_mean = False

        # --- Filtro de Hamilton + previsão one-step (com ARMA/DR) ---
        if self._use_expanded_mean:
            Pm_eff = Pm_exp
            dec = self._mean_decoder
        else:
            Pm_eff = Pm
            dec = None

        ll, filtered, fitted = self._filter(y, Pm_eff, Pv, a, Phi, Theta, Beta, sigmasq, self._Z, mean_decoder=dec)

        y_arr = np.asarray(y, dtype=float)
        resid = y_arr - np.asarray(fitted, dtype=float)

        # --- Suavização de Kim ---
        smooth, pair = self._smooth(filtered, Pm_eff, Pv, return_pair=True)

        # --- Probabilidades marginais de regimes de média e variância ---
        T = len(y)
        km_chain = int(Pm_eff.shape[0])
        mean_probs = np.zeros((T, self.km))   # base regimes (0..km-1)
        var_probs  = np.zeros((T, self.kv))

        # mapeamento de estado expandido -> regime corrente base
        m0_of_chain = None
        if self._use_expanded_mean:
            m0_of_chain = np.asarray([st[0] for st in self._mean_decoder], dtype=int)

        for t in range(T):
            # vetor de probabilidades está ordenado como s = v*km_chain + m_state
            M = smooth[t].reshape(self.kv, km_chain)   # linhas: v, colunas: m_state

            var_probs[t] = M.sum(axis=1)              # kv
            mean_chain = M.sum(axis=0)                # km_chain

            if not self._use_expanded_mean:
                # km_chain == km
                mean_probs[t] = mean_chain
            else:
                # agrega km_chain -> km pelos regimes correntes
                mean_probs[t].fill(0.0)
                np.add.at(mean_probs[t], m0_of_chain, mean_chain)

        # --- Empacota resultados ---
        params = {
            "a": a,
            "phi": (Phi[0] if self.share_ar else Phi.reshape(-1)),
            "theta_ma": (Theta[0] if (self.q_ma > 0 and self.share_ma)
                        else Theta.reshape(-1) if self.q_ma > 0
                        else np.array([])),
            "beta": (Beta[0] if (self.D > 0 and self.share_beta)
                    else Beta.reshape(-1) if self.D > 0
                    else np.array([])),
            "sigma2": sigmasq,
            "theta_m": theta_m,
            "theta_v": theta_v,
        }

        # diagnósticos (raízes AR/MA, máscara, fração de resgates se você acumula em _filter)
        def _roots(poly):
            try:
                r = np.roots(poly)
                return np.sort(np.abs(r))
            except Exception:
                return np.array([np.nan])

        diag = {"ar_roots_min": [], "ma_roots_min": [], "rescue_frac": None, "mask_ok": None}

        def _minabs(roots):
            roots = np.asarray(roots)
            return np.inf if roots.size == 0 else float(np.min(np.abs(roots)))

        for m in range(self.km):
            if self.p > 0:
                diag["ar_roots_min"].append(_minabs(self._ar_roots(Phi[m])))
            if self.q_ma > 0:
                diag["ma_roots_min"].append(_minabs(self._ma_roots(Theta[m])))

        diag["rescue_frac"] = getattr(self, "_rescue_count", 0) / max(1, len(y))

        if self._use_expanded_mean:
            P_result = kron(Pv, Pm_exp)
        else:
            P_result = kron(Pv, Pm)

        # --- Diagnósticos finais (máscaras/estrutura em Pm) ---
        eff = getattr(self, "_effective_mask_mean", None)
        if eff is None:
            diag["mask_effective"] = False
            diag["mask_ok"] = True
            diag["mask_violations"] = []
        else:
            eff = np.asarray(eff, dtype=bool)
            diag["mask_effective"] = True
            diag["mask_ok"] = bool(np.allclose(Pm[~eff], 0.0, atol=1e-8))
            diag["mask_violations"] = [tuple(map(int, ij)) for ij in np.argwhere((~eff) & (np.abs(Pm) > 1e-8))]
            diag["mask_effective_allowed"] = int(eff.sum())
            diag["mask_effective_total"] = int(eff.size)

        diag["mask_user_provided"] = bool(getattr(self, "mask_mean", None) is not None)
        diag["mask_convention_user"] = str(getattr(self, "mask_convention", "from_to"))
        diag["mask_transformations"] = (
            f"triangular_Pm={bool(getattr(self,'triangular_Pm',False))}; "
            f"absorbing_last_mean={bool(getattr(self,'absorbing_last_mean',False))}; "
            f"structural_break={getattr(self,'structural_break',None)}")
        self.diagnostics_ = diag

        # --- Armazena resultados --- #
        # Convenção (crítica): internamente usamos 'from_to' (row-stochastic):
        #   Pm[i,j] = Pr(S^m_t=j | S^m_{t-1}=i) e sum_j Pm[i,j]=1 (idem para Pv).
        #   No filtro: alpha_pred = P.T @ alpha_prev equivale a alpha_prev^T P (P row-stochastic).
        _assert_transition_matrix(Pm, name="Pm", convention="from_to")
        _assert_transition_matrix(Pv, name="Pv", convention="from_to")
        _assert_transition_matrix(P_result, name="P", convention="from_to")

        # Convenção do paper/Doornik p_{i|j} = Pr(S_t=i | S_{t-1}=j) (colunas somam 1):
        Pm_pij = Pm.T
        Pv_pij = Pv.T
        P_pij  = P_result.T
        _assert_transition_matrix(Pm_pij, name="Pm_pij", convention="paper_pij")
        _assert_transition_matrix(Pv_pij, name="Pv_pij", convention="paper_pij")
        _assert_transition_matrix(P_pij,  name="P_pij",  convention="paper_pij")

        self.fitted_ = MSCompResults(
            params=params,
            transition_convention="from_to",
            # ---
            Pm=Pm, Pv=Pv, P=P_result,
            Pm_pij=Pm_pij, Pv_pij=Pv_pij, P_pij=P_pij,
            # ---
            loglike=ll,
            filtered_probs=filtered,
            smoothed_probs=smooth,
            mean_regime_probs=mean_probs,
            var_regime_probs=var_probs,
            fitted=fitted,
            residuals=resid,
            ar_order=self.p,
            k_mean=self.km, k_var=self.kv,
            
            k_mean_chain=int(Pm_eff.shape[0]),
            use_expanded_mean=bool(self._use_expanded_mean),
            
            q_exog=getattr(self, "q", 0),
            exog_lags=getattr(self, "Lx", 0),
            ma_order=self.q_ma,
            smoothed_pair_probs=pair,   # <- novo
        )
        return self.fitted_

    def predict_in_sample(self):
        if self.fitted_ is None:
            raise RuntimeError("Call fit() first.")
        return self.fitted_.fitted

    def summary(self) -> str:
        if getattr(self, "fitted_", None) is None:
            return "MSComp(not fitted)."

        r = self.fitted_
        L = []
        L.append("=== MSComp Results ===")
        L.append(f"log-likelihood: {r.loglike:.3f}")
        L.append(f"Mean regimes: {r.k_mean}, Variance regimes: {r.k_var}, AR order: {r.ar_order}, MA order: {r.ma_order}")

        # parâmetros
        L.append("Intercepts (a_i): " + ", ".join(f"{x:.4f}" for x in r.params["a"]))

        if r.ar_order > 0:
            ph = np.atleast_1d(r.params["phi"])
            if getattr(self, "share_ar", True):
                L.append("Shared AR coefficients (phi): " + ", ".join(f"{x:.4f}" for x in ph))
            else:
                L.append("AR coefficients by regime (flattened): " + ", ".join(f"{x:.4f}" for x in ph))

        if r.ma_order > 0:
            th = np.atleast_1d(r.params["theta_ma"])
            if getattr(self, "share_ma", True):
                L.append("Shared MA coefficients (theta): " + ", ".join(f"{x:.4f}" for x in th))
            else:
                L.append("MA coefficients by regime (flattened): " + ", ".join(f"{x:.4f}" for x in th))

        if r.q_exog > 0:
            bet = np.atleast_1d(r.params["beta"])
            L.append(f"DR (beta), q={r.q_exog}, Lx={r.exog_lags}: " + ", ".join(f"{x:.4f}" for x in bet))

        L.append("Sigma^2 (variance regimes): " + ", ".join(f"{x:.6f}" for x in r.params["sigma2"]))

        # matrizes de transição
        L.append("Pm (mean transitions):\n" + np.array2string(r.Pm, formatter={'float_kind': lambda x: f'{x: .4f}'}))
        L.append("Pv (variance transitions):\n" + np.array2string(r.Pv, formatter={'float_kind': lambda x: f'{x: .4f}'}))

        # ==== Diagnósticos adicionais (se disponíveis) ====
        diag = getattr(self, "diagnostics_", None) or {}

        def _fmt(v, nd=9):
            v = float(v)
            if not np.isfinite(v):
                return "inf"
            return f"{v:.{nd}f}"

        # alvo efetivo
        try:
            target = float(self._root_target())
        except Exception:
            # fallback: 1 + max(margin, eps)
            margin = float(getattr(self, "unit_root_margin", 0.0))
            eps    = float(getattr(self, "unit_root_eps", 1e-6))
            target = 1.0 + max(margin, eps)

        ar_roots = np.asarray(diag.get("ar_roots_min", []), dtype=float).ravel()
        ma_roots = np.asarray(diag.get("ma_roots_min", []), dtype=float).ravel()

        if ar_roots.size > 0:
            L.append("Min |root_AR| por regime: " + ", ".join(_fmt(v) for v in ar_roots))
            L.append("Gap_AR (min|root|-1): " + ", ".join(_fmt(v - 1.0) for v in ar_roots))
            L.append("Gap_AR_to_target (min|root|-target): " + ", ".join(_fmt(v - target) for v in ar_roots))

        if ma_roots.size > 0:
            L.append("Min |root_MA| por regime: " + ", ".join(_fmt(v) for v in ma_roots))
            L.append("Gap_MA (min|root|-1): " + ", ".join(_fmt(v - 1.0) for v in ma_roots))
            L.append("Gap_MA_to_target (min|root|-target): " + ", ".join(_fmt(v - target) for v in ma_roots))

        if hasattr(self, "unit_root_margin"):
            L.append(f"Root margin target: {float(getattr(self, 'unit_root_margin', 0.0)):.6f}")
        L.append(f"Root target effective: {target:.6f}")

        resc = diag.get("rescue_frac", None)
        if resc is not None:
            L.append(f"Rescue frac (filtro): {float(resc):.4f}")

        m_ok = diag.get("mask_ok", None)
        if m_ok is not None:
            L.append(f"Mask respected (Pm[~mask]=0): {bool(m_ok)}")

        eff = getattr(self, "_effective_mask_mean", None)
        if eff is not None:
            try:
                ok_eff = bool(np.allclose(r.Pm[~eff], 0.0, atol=1e-8))
                L.append(f"Mask respected (effective): {ok_eff}")
            except Exception:
                pass

        # checagem de P == kron(Pv, Pm/Pm_exp)
        try:
            if getattr(self, "_use_expanded_mean", False):
                Pm_exp, _ = self._expand_mean_chain(r.Pm)
                P_expected = kron(r.Pv, Pm_exp)
                tag = "_exp"
            else:
                P_expected = kron(r.Pv, r.Pm)
                tag = ""
            eqP = np.allclose(getattr(r, "P", P_expected), P_expected, atol=1e-8, rtol=1e-8)
            L.append(f"P == kron(Pv,Pm{tag})? {bool(eqP)}")
        except Exception:
            pass

        return "\n".join(L)