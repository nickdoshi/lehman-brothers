import numpy as np
from statsmodels.tsa.stattools import coint, adfuller
import statsmodels.api as sm

def find_cointegrated_pairs(prcSoFar, lookback=120, coint_pval_thresh=0.05, adf_pval_thresh=0.05):
    window = prcSoFar[:, -lookback-1:-1]
    (nins, nt) = window.shape
    pairs = []
    for i in range(nins):
        for j in range(i+1, nins):
            S1 = window[i, :]
            S2 = window[j, :]

            # Hedge Ration: Beta = cov(S1, S2) / var(S2) 
            # cov = np.cov(np.stack((S1, S2), axis=0))[0][1]
            # var = np.var(S2)
            # if abs(var) < 1e-6: #unstable
            #     continue
            # beta = cov / var

            # Hedge Ratio Linear Regression:
            S2_const = sm.add_constant(S2)
            model = sm.OLS(S1, S2_const).fit()
            alpha = model.params[0]
            beta = model.params[1]

            # Spread and Coint P-value
            spread = S1 - (alpha + beta * S2)
            coint_t, pvalue, _ = coint(S1, S2)
            if pvalue > coint_pval_thresh:
                continue
            
            # ADF test for stationarity
            adf_stat, pval_adf, *_ = adfuller(spread)
            if pval_adf < adf_pval_thresh: 
                pairs.append((i, j, alpha, beta, spread.mean(), spread.std()))

    return pairs