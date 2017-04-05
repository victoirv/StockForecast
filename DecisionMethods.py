from random import randint
import logging
import numpy
from IR import *

def randombuy(data,**extraArg):
    return randint(-1,1)

def weeklyrise(data,**extraArg):
    difference=extraArg['difference']
    if(data[-1]>data[-6]*(1+difference)):
        return -1
    elif(data[-1]<data[-6]*(1-difference)):
        return 1
    else:
        return 0


def dailyrise(data,**extraArg):
    difference=extraArg['difference']
    if(data[-1]>data[-2]*(1+difference)):
        return -1
    elif(data[-1]<data[-2]*(1-difference)):
        return 1
    else:
        return 0

def LS(data,**extraArg):
    difference=extraArg['difference']
    X=data[1:,0]
    F=data[:-1,1:]
    A=numpy.vstack([F.T,numpy.ones(len(X))]).T
    m=numpy.linalg.lstsq(A,X)[0]
    tomorrow=float(m[0]*F[-1]+m[1])

    logging.debug('X(end): {0:.2f}'.format(X[-1]))
    logging.debug('F(end): {0:.2f}'.format(float(data[-1,1:])))
    logging.debug('Tomorrow value: {0:.2f} from m: {1:.2f} c: {2:.2f}'.format(tomorrow,m[0],m[1]))

    if(tomorrow>X[-1]*(1+difference)):
        return -1
    elif(tomorrow<X[-1]*(1-difference)):
        return 1
    else:
        return 0


def IRpred(data,Nx=1,Nf=5,ratio=0.03,days=5,**extraArg):
    #Args: Nx, Nf, ratio, days
    X=data[1:,0]
    F=data[:-1,1:]

    pred=IR(X,F,Nx,Nf)
    if(pred>(X[-days-1]*(1+ratio))):
        return -1
    elif(pred<(X[-days-1]*(1-ratio))):
        return 1
    else:
        return 0

    
