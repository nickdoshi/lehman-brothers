import numpy as np
from statsmodels.tsa.stattools import coint, adfuller
import statsmodels.api as sm
from main import COINT_PVAL_THRESH, ADF_PVAL_THRESH, BASE_CAP, MAX_CAP

def find_cointegrated_pairs(prcSoFar, lookback=120):
    (nins, nt) = prcSoFar.shape
    width = max(nt, lookback)
    width = min(width, 500)
    window = prcSoFar[:, -width-1:-1]
    
    pairs = []
    for i in range(nins):
        for j in range(i + 1, nins):
            S1, S2 = window[i, :], window[j, :]
            S2_const = sm.add_constant(S2)
            model = sm.OLS(S1, S2_const).fit()
            alpha, beta = model.params[0], model.params[1]
            
            spread = S1 - (alpha + beta * S2)
            if spread.std() < 1e-6:
                continue
            
            coint_t, coint_pval, _ = coint(S1, S2)
            if coint_pval > COINT_PVAL_THRESH:
                continue
                
            adf_stat, adf_pval, *_ = adfuller(spread)
            if adf_pval > ADF_PVAL_THRESH:
                continue
                
            # halflife = estimate_half_life(spread)
            # if halflife > 10:
            #     continue
            
            pairs.append((i, j, alpha, beta, spread.mean(), spread.std(), adf_pval))
    
    pairs.sort(key=sort_func)
    return pairs

def estimate_half_life(spread):
    spread_lag = spread[:-1]
    spread_ret = spread[1:] - spread[:-1]
    spread_lag_const = sm.add_constant(spread_lag)
    model = sm.OLS(spread_ret, spread_lag_const).fit()
    beta = model.params[1]
    if beta >= 0:
        return np.inf
    halflife = -np.log(2) / beta
    return max(halflife, 1)

def sort_func(p):
    # Prioritize strong ADF, low spread std, reasonable half-life
    return p[6] + 0.1 * p[5]

def priceFromZscore(z, pval_adf, entry_z):
    B = np.log(11) / 2
    A = 200 / (np.exp(B))
    cap_scale = 1 - pval_adf / ADF_PVAL_THRESH
    return A * np.exp(B * abs(z)) + 300 * cap_scale