
import numpy as np

from helper import find_cointegrated_pairs as getStockPairs 
import json

##### TODO #########################################
### IMPLEMENT 'getMyPosition' FUNCTION #############
### TO RUN, RUN 'eval.py' ##########################

nInst = 50
currentPos = np.zeros(nInst)

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


def priceFromZscore(z):

    # Points (1,500) and (3, 4500)
    # y = A e^Bx + 300
    B = np.log(11) / 2
    A = 200 / (np.exp(B))
    if z < 2:
        return 0
    if z > 3:
        return 2000
    # Expo: 
    return A * np.exp(B * abs(z)) + 300 

# Model 2: Co-integration + ADF test

def getMyPosition3(prcSoFar, 
                   lookback=200, 
                   entry_z=1.0, 
                   exit_z=0.5):
    global currentPos, pairs
    (nins, nt) = prcSoFar.shape

    # Ignore the first window width of days
    if (nt < lookback): 
        return np.zeros(nins)
    
    # Slide window every 25 time increments, more granular --> worst runtime
    if (nt % 50 == 0):
        pairs = getStockPairs(prcSoFar, lookback=lookback)

    
    latest = prcSoFar[:, -1]               # shape (nInst,)
    if (nt % 50 == 0):
        print(f"checking day {nt}")

    for i, j, alpha, beta, mean_spread, std_spread in pairs:
        if std_spread < 1e-6:
            continue

        # current spread
        price_i = latest[i]
        price_j = latest[j]
        spread_now = price_i - (alpha + beta*price_j)
        z = (spread_now - mean_spread)/std_spread
        cap_per_pair = priceFromZscore(abs(z))
        # entry / exit logic
        if z > entry_z:
            if (nt % 50 == 0):
                print(f"found pos spread {z}")
            # short the spread: i short, j long*β
            currentPos[i] += int(-cap_per_pair / price_i)
            currentPos[j] +=  int(cap_per_pair * beta / price_j)
        elif z < -entry_z:
            if (nt % 50 == 0):
                print(f"found neg spread {z}")
            # long the spread: i long, j short*β
            currentPos[i] +=  int(cap_per_pair / price_i)
            currentPos[j] += int(-cap_per_pair * beta / price_j)
        elif abs(z) < exit_z:
            currentPos[i] = 0
            currentPos[j] = 0
        # if |z| < exit_z, we do nothing (positions reset each call)
    if (nt % 50 == 0):
        print(currentPos)
    return currentPos