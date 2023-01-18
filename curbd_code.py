# File to run the CURBD code for a particular sweep and parameter
import math
import random

import numpy as np
import numpy.random as npr
import numpy.linalg

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

activity = np.load("mouse599975_allprobes_sweep1_param30.npy")  #[:1, 85000:89000]        #[:, 85000:86000]  #[:1,85600:85650]   #numpy.array  N X T
print(activity.shape)
#activity = np.zeros((1, 50))
dtData=0.005                        #time step (in s) of the training data
dtFactor=5                          #number of interpolation steps for RNN -> used to calculate the time step of integration in the euqation
g=1.5  #0.8    #1.5                       #instability -> strength of the recurrent connections
tauRNN=0.0075                          #RNN units decay constant -> tau in the equation 
tauWN=0.1                           #decay constant on filtered white noise inputs            
ampInWN=0.001                        #input amplitude of filtered white noise
nRunTrain=2000      #2000                      #no of training runs
nRunFree=10                         #no of untrained runs at end
P0=1.0                              #learning rate
nonLinearity=np.tanh
nonLinearity_inv=np.arctanh
resetPoints=None                    #list of indeces into T. default to only set initial state at time 1.
plotStatus=False                    #whether to plot data fits during training 
verbose=True                        #whether to print status updates
regions=None

"""
    Trains a data-constrained multi-region RNN. The RNN can be used for,
    among other things, Current-Based Decomposition (CURBD).
    Parameters
    ----------
    activity: numpy.array
        N X T
    dtData: float
        time step (in s) of the training data
    dtFactor: float
        number of interpolation steps for RNN
    g: float
        instability (chaos); g<1=damped, g>1=chaotic
    tauRNN: float
        decay constant of RNN units
    tauWN: float
        decay constant on filtered white noise inputs
    ampInWN: float
        input amplitude of filtered white noise
    nRunTrain: int
        number of training runs
    nRunFree: int
        number of untrained runs at end
    P0: float
        learning rate
    nonLinearity: function
        inline function for nonLinearity
    resetPoints: list of int
        list of indeces into T. default to only set initial state at time 1.
    plotStatus: bool
        whether to plot data fits during training
    verbose: bool
        whether to print status updates
    regions: dict()
        keys are region names, values are np.array of indeces.
"""
if dtData is None:
    print('dtData not specified. Defaulting to 1.');
    dtData = 1;
if resetPoints is None:
    resetPoints = [0, ]
if regions is None:
    regions = {}
    
number_units = activity.shape[0]
number_learn = activity.shape[0]

dtRNN = dtData / float(dtFactor)     #time step of training data/no of interpolation steps
nRunTot = nRunTrain + nRunFree       #Total no of training epochs

# set up everything for training
learnList = npr.permutation(number_units)
iTarget = learnList[:number_learn]
iNonTarget = learnList[number_learn:]
tData = dtData*np.arange(activity.shape[1])
tRNN = np.arange(0, tData[-1] + dtRNN, dtRNN)

ampWN = math.sqrt(tauWN/dtRNN)
iWN = ampWN * npr.randn(number_units, len(tRNN))
inputWN = np.ones((number_units, len(tRNN)))
for tt in range(1, len(tRNN)):
    inputWN[:, tt] = iWN[:, tt] + (inputWN[:, tt - 1] - iWN[:, tt])*np.exp(- (dtRNN / tauWN))
inputWN = ampInWN * inputWN

# initialize directed interaction matrix J
J = g * npr.randn(number_units, number_units) / math.sqrt(number_units)
J0 = J.copy()

# set up target training data
Adata = activity.copy()
Adata = Adata/Adata.max()
Adata = np.minimum(Adata, 0.999)
Adata = np.maximum(Adata, -0.999)

# get standard deviation of entire data
stdData = np.std(Adata[iTarget, :])

# get indices for each sample of model data
iModelSample = numpy.zeros(len(tData), dtype=np.int32)
for i in range(len(tData)):
    iModelSample[i] = (np.abs(tRNN - tData[i])).argmin()
    
# initialize some others
RNN = np.zeros((number_units, len(tRNN)))
chi2s = []
pVars = []

# initialize learning update matrix (see Sussillo and Abbot, 2009)
PJ = P0*np.eye(number_learn)

if plotStatus is True:
    plt.rcParams.update({'font.size': 6})
    fig = plt.figure()
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    gs = GridSpec(nrows=2, ncols=4)
else:
    fig = None

#print(Adata[:, 0, np.newaxis])
# start training
# loop along training runs
for nRun in range(0, nRunTot):
    H = Adata[:, 0, np.newaxis]
    RNN[:, 0, np.newaxis] = nonLinearity(H)
    # variables to track when to update the J matrix since the RNN and
    # data can have different dt values
    tLearn = 0  # keeps track of current time
    iLearn = 0  # keeps track of last data point learned
    chi2 = 0.0

    for tt in range(1, len(tRNN)):
        # update current learning time
        tLearn += dtRNN
        # check if the current index is a reset point. Typically this won't
        # be used, but it's an option for concatenating multi-trial data
        if tt in resetPoints:
            timepoint = math.floor(tt / dtFactor)
            H = Adata[:, timepoint]
        # compute next RNN step
        RNN[:, tt, np.newaxis] = nonLinearity(H)
        JR = (J.dot(RNN[:, tt]).reshape((number_units, 1)) + inputWN[:, tt, np.newaxis])
        #print("#JR#")
        #print(JR)
        #print("H")
        #print(H)
        #print("tauRNN")
        #print(tauRNN)
        #print(dtRNN)
        H = H + dtRNN*(-H + JR)/tauRNN
        # check if the RNN time coincides with a data point to update J
        if tLearn >= dtData:
            tLearn = 0
            err = RNN[:, tt, np.newaxis] - Adata[:, iLearn, np.newaxis]
            iLearn = iLearn + 1
            # update chi2 using this error
            chi2 += np.mean(err ** 2)

            if nRun < nRunTrain:
                r_slice = RNN[iTarget, tt].reshape(number_learn, 1)
                k = PJ.dot(r_slice)
                rPr = (r_slice).T.dot(k)[0, 0]
                c = 1.0/(1.0 + rPr)
                PJ = PJ - c*(k.dot(k.T))
                J[:, iTarget.flatten()] = J[:, iTarget.reshape((number_units))] - c*np.outer(err.flatten(), k.flatten())

    rModelSample = RNN[iTarget, :][:, iModelSample]
    distance = np.linalg.norm(Adata[iTarget, :] - rModelSample)
    pVar = 1 - (distance / (math.sqrt(len(iTarget) * len(tData)) * stdData)) ** 2
    pVars.append(pVar)
    chi2s.append(chi2)
    #mouse599975_allprobes_sweep1_param70.npy
    np.save("mouse599975_sweep1_param30_chi2s_2000iters.npy", chi2s)
    np.save("mouse599975_sweep1_param30_pvars_2000iters.npy", pVars)
    np.save("mouse599975_sweep1_param30_J_2000iters.npy", J)
    if verbose:
        print('trial=%d pVar=%f chi2=%f' % (nRun, pVar, chi2))
    if fig:
        fig.clear()
        ax = fig.add_subplot(gs[0, 0])
        ax.axis('off')
        ax.imshow(Adata[iTarget, :])
        ax.set_title('real rates')

        ax = fig.add_subplot(gs[0, 1])
        ax.imshow(RNN, aspect='auto')
        ax.set_title('model rates')
        ax.axis('off')

        ax = fig.add_subplot(gs[1, 0])
        ax.plot(pVars)
        ax.set_ylabel('pVar')

        ax = fig.add_subplot(gs[1, 1])
        ax.plot(chi2s)
        ax.set_ylabel('chi2s')

        ax = fig.add_subplot(gs[:, 2:4])
        idx = npr.choice(range(len(iTarget)))
        ax.plot(tRNN, RNN[iTarget[idx], :])
        ax.plot(tData, Adata[iTarget[idx], :])
        ax.set_title(nRun)
        fig.show()
        plt.pause(0.05)
        
np.save("mouse599975_sweep1_param30_chi2s.npy", chi2s)
np.save("mouse599975_sweep1_param30_pvars.npy", pVars)
np.save("mouse599975_sweep1_param30_J.npy", J)
#chi2s = []
#pVars = []
        
out_params = {}
out_params['dtFactor'] = dtFactor
out_params['number_units'] = number_units
out_params['g'] = g
out_params['P0'] = P0
out_params['tauRNN'] = tauRNN
out_params['tauWN'] = tauWN
out_params['ampInWN'] = ampInWN
out_params['nRunTot'] = nRunTot
out_params['nRunTrain'] = nRunTrain
out_params['nRunFree'] = nRunFree
out_params['nonLinearity'] = nonLinearity
out_params['resetPoints'] = resetPoints

out = {}
out['regions'] = regions
out['RNN'] = RNN
out['tRNN'] = tRNN
out['dtRNN'] = dtRNN
out['Adata'] = Adata
out['tData'] = tData
out['dtData'] = dtData
out['J'] = J
out['J0'] = J0
out['chi2s'] = chi2s
out['pVars'] = pVars
out['stdData'] = stdData
out['inputWN'] = inputWN
out['iTarget'] = iTarget
out['iNonTarget'] = iNonTarget
out['params'] = out_params
