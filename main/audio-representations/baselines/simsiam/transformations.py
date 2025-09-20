import numpy as np
from scipy.interpolate import CubicSpline      # for warping
from transforms3d.axangles import axangle2mat  # for rotation
import tensorflow as tf

def DA_Jitter(X, sigma=0.8):
    myNoise = np.random.normal(loc=0, scale=sigma, size=X.shape)
    return X+myNoise

def DA_Scaling(X, sigma=1.1*0.5):
    scalingFactor = np.random.normal(loc=1.0, scale=sigma, size=(1,X.shape[1])) # shape=(1,3)
    myNoise = np.matmul(np.ones((X.shape[0],1)), scalingFactor)
    return X*myNoise

def GenerateRandomCurves(X, sigma=0.2, knot=4):
    xx = (np.ones((X.shape[1],1))*(np.arange(0,X.shape[0], (X.shape[0]-1)/(knot+1)))).transpose()
    yy = np.random.normal(loc=1.0, scale=sigma, size=(knot+2, X.shape[1]))
    x_range = np.arange(X.shape[0])
    cs=[]
    for i in range(X.shape[1]):
        cs.append(CubicSpline(xx[:,i], yy[:,i])(x_range))
    #cs_x = CubicSpline(xx[:,0], yy[:,0])
    #cs_y = CubicSpline(xx[:,1], yy[:,1])
    #cs_z = CubicSpline(xx[:,2], yy[:,2])
    ret_lst=np.array(cs).transpose()
    #print("return list size of generate random curves : ",ret_lst.shape)
    return ret_lst
    
def DA_MagWarp(X, sigma = 0.5):
    return X * GenerateRandomCurves(X, sigma)

def DistortTimesteps(X, sigma=0.2):
    tt = GenerateRandomCurves(X, sigma) # Regard these samples aroun 1 as time intervals
    tt_cum = np.cumsum(tt, axis=0)        # Add intervals to make a cumulative graph
    # Make the last value to have X.shape[0]
    t_scale = [(X.shape[0]-1)/tt_cum[-1,0],(X.shape[0]-1)/tt_cum[-1,1],(X.shape[0]-1)/tt_cum[-1,2]]
    for i in range(X.shape[1]):
        tt_cum[:,0] = tt_cum[:,0]*t_scale[0]
    #tt_cum[:,0] = tt_cum[:,0]*t_scale[0]
    #tt_cum[:,1] = tt_cum[:,1]*t_scale[1]
    #tt_cum[:,2] = tt_cum[:,2]*t_scale[2]
    return tt_cum

def DA_TimeWarp(X, sigma=0.2):
    tt_new = DistortTimesteps(X, sigma)
    X_new = np.zeros(X.shape)
    x_range = np.arange(X.shape[0])
    for i in range(X.shape[1]):
        X_new[:,i] = np.interp(x_range, tt_new[:,i], X[:,i])
    
    return X_new

def DA_Rotation(X,sigma=0):
    axis = np.random.uniform(low=-1, high=1, size=X.shape[1])
    angle = np.random.uniform(low=-np.pi, high=np.pi)
    return np.matmul(X , axangle2mat(axis,angle))

def DA_Permutation(X, nPerm=4, minSegLength=10, sigma=0):
    X_new = np.zeros(X.shape)
    idx = np.random.permutation(nPerm)
    bWhile = True
    while bWhile == True:
        segs = np.zeros(nPerm+1, dtype=int)
        segs[1:-1] = np.sort(np.random.randint(minSegLength, X.shape[0]-minSegLength, nPerm-1))
        segs[-1] = X.shape[0]
        if np.min(segs[1:]-segs[0:-1]) > minSegLength:
            bWhile = False
    pp = 0
    for ii in range(nPerm):
        x_temp = X[segs[idx[ii]]:segs[idx[ii]+1],:]
        X_new[pp:pp+len(x_temp),:] = x_temp
        pp += len(x_temp)
    return(X_new)

def RandSampleTimesteps(X, nSample=1000):
    X_new = np.zeros(X.shape)
    tt = np.zeros((nSample,X.shape[1]), dtype=int)
    for i in range(X.shape[1]):
        tt[1:-1,i] = np.sort(np.random.randint(1,X.shape[0]-1,nSample-2))
    #tt[1:-1,0] = np.sort(np.random.randint(1,X.shape[0]-1,nSample-2))
    #tt[1:-1,1] = np.sort(np.random.randint(1,X.shape[0]-1,nSample-2))
    #tt[1:-1,2] = np.sort(np.random.randint(1,X.shape[0]-1,nSample-2))
    tt[-1,:] = X.shape[0]-1
    return tt

def DA_RandSampling(X, nSample=1000, sigma=0):
    tt = RandSampleTimesteps(X, nSample)
    X_new = np.zeros(X.shape)
    for i in range(X.shape[1]):
        X_new[:,i] = np.interp(np.arange(X.shape[0]), tt[:,i], X[tt[:,i],i])
    #X_new[:,0] = np.interp(np.arange(X.shape[0]), tt[:,0], X[tt[:,0],0])
    #X_new[:,1] = np.interp(np.arange(X.shape[0]), tt[:,1], X[tt[:,1],1])
    #X_new[:,2] = np.interp(np.arange(X.shape[0]), tt[:,2], X[tt[:,2],2])
    return X_new

def DA_Combined(X, nPerm=4, minSegLength=10, sigma=0):
    X_new=DA_Permutation(X)
    return DA_Rotation(X_new)
    
def DA_Negation(X,sigma=0):
    return -1*X

def DA_Flip(X,sigma=0):
    return np.flip(X,0)

def DA_ChannelShuffle(X, sigma=0):
    indx=np.arange(X.shape[1])
    np.random.shuffle(indx)
    return X[:,indx]
    
def DA_Drop(X, W=7, sigma=10):
    W=sigma
    indx=np.arange(X.shape[0]-W)
    np.random.shuffle(indx)
    X[indx[0]:indx[0]+W,:]=0
    return X