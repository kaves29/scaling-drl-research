"""
Sparse Evolutionary Training (SET) implementation.

Reference for the original SET algorithm:
Mocanu, D. C., Mocanu, E., Stone, P., Nguyen, P. H., Gibescu, M., & Liotta, A. (2018). 
"Scalable training of artificial neural networks with adaptive sparse connectivity 
inspired by network science." Nature Communications.

Code adapted from:
Sokar, G. (2019/2020). Dynamic Sparse Training for Deep Reinforcement Learning.
Repository: https://github.com/GhadaSokar/Dynamic-Sparse-Training-for-Deep-Reinforcement-Learning
"""

import numpy as np
import torch

def initializeEpsilonWeightsMask(text,epsilon, noRows, noCols):
    # generate an epsilon based Erdos Renyi sparse weights mask
    mask_weights = np.random.rand(noRows, noCols)
    prob = 1 - (epsilon * (noRows + noCols)) / (noRows * noCols)
    mask_weights = np.random.rand(noRows, noCols)
    mask_weights[mask_weights < prob] = 0
    mask_weights[mask_weights >= prob] = 1
    noParameters = np.sum(mask_weights)
    sparsity = 1-noParameters/(noRows * noCols)
    print("Epsilon Sparse Initialization ",text,": Epsilon ",epsilon,"; Sparsity ",sparsity,"; NoParameters ",noParameters,"; NoRows ",noRows,"; NoCols ",noCols,"; NoDenseParam ",noRows*noCols)
    print (" OutDegreeBottomNeurons %.2f ± %.2f; InDegreeTopNeurons %.2f ± %.2f" % (np.mean(mask_weights.sum(axis=1)),np.std(mask_weights.sum(axis=1)),np.mean(mask_weights.sum(axis=0)),np.std(mask_weights.sum(axis=0))))
    return [noParameters, mask_weights.transpose()]

def initializeSparsityLevelWeightMask(text,sparsityLevel,noRows, noCols):
    # generate an Erdos Renyi sparse weights mask
    prob=sparsityLevel
    mask_weights = np.random.rand(noRows, noCols)
    mask_weights[mask_weights < prob] = 0
    mask_weights[mask_weights >= prob] = 1
    noParameters = np.sum(mask_weights)
    sparsity = 1-noParameters/(noRows * noCols)
    epsilon = int((prob*(noRows * noCols)-1)/(noRows + noCols))
    print("Sparsity Level Initialization ",text,": Computed Epsilon ",epsilon,"; Sparsity ",sparsity,"; NoParameters ",noParameters,"; NoRows ",noRows,"; NoCols ",noCols,"; NoDenseParam ",noRows*noCols)
    print (" OutDegreeBottomNeurons %.2f ± %.2f; InDegreeTopNeurons %.2f ± %.2f" % (np.mean(mask_weights.sum(axis=1)),np.std(mask_weights.sum(axis=1)),np.mean(mask_weights.sum(axis=0)),np.std(mask_weights.sum(axis=0))))
    return [noParameters, mask_weights]

def find_first_pos(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx

def find_last_pos(array, value):
    idx = (np.abs(array - value))[::-1].argmin()
    return array.shape[0] - idx

def changeConnectivitySET(weights, noWeights, initMask, zeta, lastTopologyChange, iteration):
    # change Connectivity
    # remove zeta largest negative and smallest positive weights
    weights = weights * initMask
    values = np.sort(weights.ravel())
    firstZeroPos = find_first_pos(values, 0)
    lastZeroPos = find_last_pos(values, 0)
    largestNegative = values[int((1 - zeta) * firstZeroPos)]
    smallestPositive = values[int(min(values.shape[0] - 1, lastZeroPos + zeta * (values.shape[0] - lastZeroPos)))]
    rewiredWeights = weights.copy()
    rewiredWeights[rewiredWeights > smallestPositive] = 1
    rewiredWeights[rewiredWeights < largestNegative] = 1
    rewiredWeights[rewiredWeights != 1] = 0

    # add random weights
    nrAdd = 0
    if (lastTopologyChange==False):
        noRewires = noWeights - np.sum(rewiredWeights)
        while (nrAdd < noRewires):
            i = np.random.randint(0, rewiredWeights.shape[0])
            j = np.random.randint(0, rewiredWeights.shape[1])
            if (rewiredWeights[i, j] == 0):
                rewiredWeights[i, j] = 1
                nrAdd += 1

    ascStats=[iteration, nrAdd, noWeights, np.count_nonzero(rewiredWeights)]

    return [rewiredWeights,ascStats]

def changeConnectivityXReLU(weights, noWeights, initMask, lastTopologyChange, iteration):
    weights = weights * initMask

    weightspos = weights.copy()
    weightspos[weightspos < 0] = 0
    strengthpos = np.sum(weightspos, axis=0)

    weightsneg = weights.copy()
    weightsneg[weightsneg > 0] = 0
    strengthneg = np.sum(weightsneg, axis=0)

    for j in range(strengthpos.shape[0]):
        if (strengthpos[j] + strengthneg[j] < 0):
            difference = strengthpos[j]
            iis = np.nonzero(weightsneg[:, j])[0]
            ww = weightsneg[iis, j]
            iisort = np.argsort(ww)
            for i in iisort:
                if (difference > 0):
                    difference += weightsneg[iis[i], j]
                else:
                    weights[iis[i], j] = 0
    rewiredWeights = weights.copy()
    rewiredWeights[rewiredWeights != 0] = 1

    nrAdd = 0
    if (lastTopologyChange == False):
        noRewires = noWeights - np.sum(rewiredWeights)
        while (nrAdd < noRewires):
            i = np.random.randint(0, rewiredWeights.shape[0])
            j = np.random.randint(0, rewiredWeights.shape[1])
            if (rewiredWeights[i, j] == 0):
                rewiredWeights[i, j] = 1
                nrAdd += 1

    ascStats = [iteration, nrAdd, noWeights, np.count_nonzero(rewiredWeights)]
    return [rewiredWeights, ascStats]

class SparseManager:
    def __init__(self, agent, sparsity_level):
        self.masks = {}
        for name, param in agent.named_parameters():
            if name.endswith(".weight") and len(param.shape) == 2:
                _, mask = initializeSparsityLevelWeightMask(name, sparsity_level, *param.shape)
                self.masks[name] = torch.tensor(mask).to(param.device)

                print(f"SANITY CHECK -----------> Param.shape: {param.shape} | self.masks[name].shape: {self.masks[name].shape}")

    def apply_mask(self, agent):
        with torch.no_grad():
            for name, param in agent.named_parameters():
                if name in self.masks:
                    param.data *= self.masks[name]

    def evolve(self, agent, zeta):
        for name, param in agent.named_parameters():
            if name in self.masks:
                new_mask, stats = changeConnectivitySET(param.data.cpu().numpy(), np.count_nonzero(self.masks[name].cpu().numpy()), self.masks[name].cpu().numpy(), zeta, False, 0)
                self.masks[name] = torch.tensor(new_mask).to(param.device)
