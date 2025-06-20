
import numpy as np


import json

##### TODO #########################################
### IMPLEMENT 'getMyPosition' FUNCTION #############
### TO RUN, RUN 'eval.py' ##########################

# === GLOBAL STATE === 
nInst = 50
currentPos = np.zeros(nInst)
pairs = []
active_trades = []
closed_trades = []

# === PARAMETERS ===
COINT_PVAL_THRESH = 0.001
ADF_PVAL_THRESH = 0.001
EXPOSURE_LIMIT = 1e6
BASE_CAP = 500
MAX_CAP = 3000

# === UTILITY FUNCTIONS ===
from helper import find_cointegrated_pairs, priceFromZscore, estimate_half_life 

# Momentum - based strategy 
def getMyPosition(prcSoFar):
    global currentPos
    (nins, nt) = prcSoFar.shape
    if (nt < 2):
        return np.zeros(nins)
    
    lastRet = np.log(prcSoFar[:, -1] / prcSoFar[:, -2])
    lNorm = np.sqrt(lastRet.dot(lastRet))
    lastRet /= lNorm
    rpos = np.array([int(x) for x in 5000 * lastRet / prcSoFar[:, -1]])
    currentPos = np.array([int(x) for x in currentPos+rpos])
    return currentPos

# Mean - reversion 1 std
def getMyPosition2(prcSoFar):
    global currentPos
    (nins, nt) = prcSoFar.shape
    if (nt < 100):
        return np.zeros(nins)
    
    # Get strongly correlated stocks
    corr = np.corrcoef(prcSoFar[:, -100:])
    strong_pairs = []

    for i in range(nins):
        for j in range(i + 1, nins):
            if abs(corr[i, j]) > 0.90:
                spread_series = prcSoFar[i, -100:] - prcSoFar[j, -100:]
                mean = spread_series.mean()
                std = spread_series.std()
                if std < 1e-6:
                    print(f"skipping {i}, {j}")
                    continue
                strong_pairs.append((i, j, mean, std))

    strong_pairs.sort(key=lambda x: abs(x[3]), reverse=True) 
    strong_pairs = strong_pairs[:5]

    for stock1, stock2, mean, std in strong_pairs:
        price1 = prcSoFar[stock1, -1]
        print(price1)
        price2 = prcSoFar[stock2, -1]
        print(price2)
        spread = price1 - price2
        z = (spread - mean)/std
        stock_value = priceFromZscore(abs(z))
        if z > 1:
            currentPos[stock1] +=  -stock_value / price1
            currentPos[stock2] += stock_value / price2
        elif z < -1:
            currentPos[stock1] += stock_value / price1
            currentPos[stock2] += -stock_value / price2
    return currentPos


# Model 2: Co-integration + ADF test

def getMyPosition3(prcSoFar, lookback=120, entry_z_base=1.5, exit_z_multiplier=0.5):
    global currentPos, pairs, active_trades, closed_trades
    (nins, nt) = prcSoFar.shape
    latest = prcSoFar[:, -1]
    
    if nt < lookback:
        return np.zeros(nins)
    
    if nt % 25 == 0:
        pairs = find_cointegrated_pairs(prcSoFar, lookback)
    
    if not pairs:
        return currentPos

    baseline_std = np.median([p[5] for p in pairs if p[5] >= 1e-6]) or 1.0

    # Check for new trades
    for i, j, alpha, beta, mean_spread, std_spread, pval_adf in pairs[:5]:
        if std_spread < 1e-6:
            continue

        price_i, price_j = latest[i], latest[j]
        spread_now = price_i - (alpha + beta * price_j)
        z = (spread_now - mean_spread) / std_spread

        entry_z = max(1.5, entry_z_base * (1 - pval_adf / ADF_PVAL_THRESH))
        exit_z = entry_z * exit_z_multiplier
        cap_per_stock = priceFromZscore(abs(z), pval_adf, entry_z)

        if (i, j) not in active_trades:
            if z > entry_z and z < 4:
                # short spread
                pos_i = int(-cap_per_stock * (std_spread / baseline_std) / price_i)
                pos_j = int(cap_per_stock * (std_spread / baseline_std) * beta / price_j)
                currentPos[i] += pos_i
                currentPos[j] += pos_j
                active_trades.append(dict(i=i, j=j,
                                             direction='short', pos_i=pos_i, pos_j=pos_j,
                                             price_i=price_i, price_j=price_j,
                                             mean=mean_spread, std=std_spread, alpha=alpha, beta=beta)
                                             )
            elif z < -entry_z and z > -4:
                # long spread
                pos_i = int(cap_per_stock * (std_spread / baseline_std) / price_i)
                pos_j = int(-cap_per_stock * (std_spread / baseline_std) * beta / price_j)
                currentPos[i] += pos_i
                currentPos[j] += pos_j
                active_trades.append(dict(i=i, j=j,
                                             direction='long', pos_i=pos_i, pos_j=pos_j,
                                             price_i=price_i, price_j=price_j,
                                             mean=mean_spread, std=std_spread, alpha=alpha, beta=beta)
                                             )

    # Exit logic
    for idx, trade in enumerate(active_trades):

        price_i, price_j = latest[trade['i']], latest[trade['j']]
        spread_now = price_i - (trade['alpha'] + trade['beta'] * price_j)
        z = (spread_now - trade['mean']) / trade['std']

        profit = (price_i - trade['price_i']) * trade['pos_i'] + (price_j - trade['price_j']) * trade['pos_j']
        exposure = abs(trade['pos_i'] * trade['price_i']) + abs(trade['pos_j'] * trade['price_j'])
        profit_pct = profit / exposure if exposure > 0 else 0

        if abs(z) < exit_z or profit_pct < -0.1:
            currentPos[trade['i']] -= trade['pos_i']
            currentPos[trade['j']] -= trade['pos_j']
            closed_trades.append(trade)
            active_trades.pop(idx)
            print(f"Closed ({trade['i']},{trade['j']}) z={z:.2f} profit={profit:.2f} pct={profit_pct:.2%}")

    if (nt % 50 == 0):
        print(len(active_trades))
        print(len(closed_trades))
    return currentPos