import numpy as np
from scipy.interpolate import CubicSpline  # for warping
from transforms3d.axangles import axangle2mat  # for rotation
import tensorflow as tf


def DA_Jitter(X, sigma=0.8):
    myNoise = np.random.normal(loc=0, scale=sigma, size=X.shape)
    return X + myNoise


def DA_Scaling(X, sigma=1.1 * 0.5):
    scalingFactor = np.random.normal(loc=1.0, scale=sigma, size=(1, X.shape[1]))  # shape=(1,3)
    myNoise = np.matmul(np.ones((X.shape[0], 1)), scalingFactor)
    return X * myNoise


def GenerateRandomCurves(X, sigma=0.2, knot=4):
    xx = (np.ones((X.shape[1], 1)) * (np.arange(0, X.shape[0], (X.shape[0] - 1) / (knot + 1)))).transpose()
    yy = np.random.normal(loc=1.0, scale=sigma, size=(knot + 2, X.shape[1]))
    x_range = np.arange(X.shape[0])
    cs = []
    for i in range(X.shape[1]):
        cs.append(CubicSpline(xx[:, i], yy[:, i])(x_range))
    ret_lst = np.array(cs).transpose()
    return ret_lst


def DA_MagWarp(X, sigma=0.5):
    return X * GenerateRandomCurves(X, sigma)


def DistortTimesteps(X, sigma=0.2):
    tt = GenerateRandomCurves(X, sigma)  # Regard these samples aroun 1 as time intervals
    tt_cum = np.cumsum(tt, axis=0)  # Add intervals to make a cumulative graph
    # Make the last value to have X.shape[0]
    t_scale = [(X.shape[0] - 1) / tt_cum[-1, i] for i in range(X.shape[1])]
    for i in range(X.shape[1]):
        tt_cum[:, i] = tt_cum[:, i] * t_scale[i]
    return tt_cum


def DA_TimeWarp(X, sigma=0.2):
    tt_new = DistortTimesteps(X, sigma)
    X_new = np.zeros(X.shape)
    x_range = np.arange(X.shape[0])
    for i in range(X.shape[1]):
        X_new[:, i] = np.interp(x_range, tt_new[:, i], X[:, i])

    return X_new


def DA_Rotation(X, sigma=0):
    axis = np.random.uniform(low=-1, high=1, size=X.shape[1])
    angle = np.random.uniform(low=-np.pi, high=np.pi)
    return np.matmul(X, axangle2mat(axis, angle))


def DA_Permutation(X, nPerm=4, minSegLength=10, sigma=0):
    X_new = np.zeros(X.shape)
    idx = np.random.permutation(nPerm)
    bWhile = True
    while bWhile == True:
        segs = np.zeros(nPerm + 1, dtype=int)
        segs[1:-1] = np.sort(np.random.randint(minSegLength, X.shape[0] - minSegLength, nPerm - 1))
        segs[-1] = X.shape[0]
        if np.min(segs[1:] - segs[0:-1]) > minSegLength:
            bWhile = False
    pp = 0
    for ii in range(nPerm):
        x_temp = X[segs[idx[ii]]:segs[idx[ii] + 1], :]
        X_new[pp:pp + len(x_temp), :] = x_temp
        pp += len(x_temp)
    return (X_new)


def RandSampleTimesteps(X, nSample=1000):
    """Generate random sample timesteps for each channel"""
    X_new = np.zeros(X.shape)
    tt = np.zeros((nSample, X.shape[1]), dtype=int)

    for i in range(X.shape[1]):
        # Generate random sample points for this channel
        tt[1:-1, i] = np.sort(np.random.randint(1, X.shape[0] - 1, nSample - 2))
        tt[0, i] = 0  # First point
        tt[-1, i] = X.shape[0] - 1  # Last point

    return tt


def DA_RandSampling(X, nSample=None, sigma=0):
    """
    Random sampling augmentation - samples the signal at random time points
    and interpolates to create a new signal.

    FIX: The original version had nSample default that was too large,
    and also used the fixed value 1000 which often equals X.shape[0],
    resulting in no change. Now defaults to ~70% of signal length.
    """
    if nSample is None:
        # Default to ~70% of the signal length for noticeable effect
        nSample = max(10, int(0.7 * X.shape[0]))

    # Ensure nSample is less than signal length
    nSample = min(nSample, X.shape[0] - 2)

    tt = RandSampleTimesteps(X, nSample)
    X_new = np.zeros(X.shape)

    for i in range(X.shape[1]):
        # Interpolate from randomly sampled points back to original length
        X_new[:, i] = np.interp(np.arange(X.shape[0]), tt[:, i], X[tt[:, i], i])

    return X_new


def DA_Combined(X, nPerm=4, minSegLength=10, sigma=0):
    X_new = DA_Permutation(X)
    return DA_Rotation(X_new)


def DA_Negation(X, sigma=0):
    return -1 * X


def DA_Flip(X, sigma=0):
    return np.flip(X, 0)


def DA_ChannelShuffle(X, sigma=0):
    indx = np.arange(X.shape[1])
    np.random.shuffle(indx)
    return X[:, indx]


def DA_Drop(X, W=7, sigma=0):
    W = sigma
    W = int(W)  # Ensure W is an integer
    if W >= X.shape[0]:
        W = X.shape[0] - 1
    if W <= 0:
        return X

    X_new = X.copy()  # Don't modify in place
    indx = np.arange(X.shape[0] - W)
    np.random.shuffle(indx)
    X_new[indx[0]:indx[0] + W, :] = 0
    return X_new