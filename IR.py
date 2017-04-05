import numpy as np
import sys

def IR(X,F,Nx,Nf):
    #Define what indices prediction should start on
    predstart=max(Nx,Nf)

    xstart=predstart-Nx
    fstart=predstart-Nf

    #print("pred: {} xs {} fs {}".format(predstart,xstart,fstart))
    matlen=len(X)-predstart

    Nimpulses=min(np.shape(F))

    #Initialize matrix to solve.
    Z=np.zeros((matlen,Nx+Nf*Nimpulses+1))
    #Matrix for predicting next value using last of data
    PredZ=np.zeros(Nx+Nf*Nimpulses+1)

    for i in range(0,Nx):
        Z[:,i]=X[xstart+i:xstart+i+matlen]
        PredZ[i]=X[len(X)-Nx+i]
    for i in range(0,Nf):
        for j in range(0,Nimpulses):
            Z[:,(i+Nx+j*Nf)]=F[(fstart+i):(fstart+i+matlen),j]
            PredZ[i+Nx+j*Nf]=F[len(X)-Nf+i,j]

    Z[:,-1]=np.ones(matlen)
    PredZ[-1]=1
    B=X[predstart:(predstart+matlen),None]

    Z=np.hstack([Z,B])

    delmask=np.all(np.isnan(Z), axis=1)
    Z=Z[~delmask]

    B=Z[:,-1]
    A=Z[:,:-1]

    m = np.linalg.lstsq(A,B)[0]
    return np.dot(PredZ,m)


def verifyIR():
    F=np.random.rand(100,2)
    X=np.arange(0.0,100.0,1.0)
    F[:,0]=np.arange(0.0,100.0,1.0)
    for i in range(2,100):
        X[i]=F[i-2,0]*0.4+F[i-1,0]*0.6+0.3

    pred=IR(X,F,2,2)
    print(pred)

    F=np.random.rand(100,2)
    #X=np.arange(0.0,100.0,1.0)
    X=np.squeeze(np.random.rand(100,1))

    for i in range(1,100):
        X[i]=X[i-1]*0.4+X[i-2]*0.2-F[i-1,0]*0.6+F[i-1,1]*0.2+0.5


    pred=IR(X,F,2,3)
    print(pred)



'''
def main(argv):
    F=np.random.rand(100,2)
    X=np.arange(0.0,100.0,1.0)
    F[:,0]=np.arange(0.0,100.0,1.0)
    for i in range(2,100):
        X[i]=F[i-2,0]*0.4+F[i-1,0]*0.6+0.3

    pred=IR(X,F,2,2)
    print(pred)


if __name__ == "__main__":
    main(sys.argv)
'''
