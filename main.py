
import numpy as np

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
    std_th = 1
    (nins, nt) = prcSoFar.shape
    if (nt < 2):
        return np.zeros(nins)
    
    # Get strongly correlated stocks
    corr = prcSoFar.corr()
    strong_pairs = list()

    for i in range(len(corr)):
        for j in range(len(corr[0])):
            if i < j and abs(corr[i][j]) > 0.9:
                mean = (prcSoFar[i] - prcSoFar[j]).mean()
                std = np.sqrt((prcSoFar[i] - prcSoFar[j]).var())
                strong_pairs.append((i,j, mean, std))
    trade_stocks = list()
    for stock1, stock2, mean, std in strong_pairs:
        price1 = prcSoFar[:-1, stock1]
        price2 = prcSoFar[:-1, stock2]

        z = (price1 - price2)/std

    

    