from yahoo_finance import Share
from datetime import date
from math import floor
import numpy as np
import matplotlib.pyplot as plt
import datetime
import logging
import getopt
import sys
from DecisionMethods import *



def getClosing(Tick,start=str(date(date.today().year-1,date.today().month,date.today().day)),end=str(date.today())):
    '''Get closing value of stock
    Default to previous year from current date
    '''
    stock = Share(Tick)
    stockhist=stock.get_historical(start,end)

    vals=list()
    vals=[float(stockhist[i]['Close']) for i in range(len(stockhist))]
    dates=[datetime.datetime.strptime(stockhist[i]['Date'],'%Y-%m-%d') for i in range(len(stockhist))]

    return (dates,vals)



def simulate(simtype, amount, data, returnSeries=0, plot=0, **extraArg):
    '''Simulate buying and selling of stock
    Takes data input and model selection
    Iteratively passes data one more day at a time to model
    Model gives back decision to buy (1), hold (0), or sell(-1)
    Simulation then calculates changes in value and investments.
    Can also be passed a plot=1 flag to plot the process.
    '''

    Holding=0
    Worth=amount
    startpoint=10
    WorthSeries=[Worth for i in range(startpoint)]

    if(type(data[0]) is np.ndarray):
        X=data[:,0]
        logging.info('Data type is 2d numpy array')
    else:
        X=data

    #Once you have an idea of the data length, check for date data
    date=extraArg.get('date',[i for i in range(len(X))])

    #print(X[-1])

    logging.debug('len(X): {0:2d}'.format(len(X)))

    if(plot):
        f, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(12, 12), dpi=100)

    for i in range(startpoint,len(X)-1):
        if(type(data[0]) is np.ndarray):
            buysell=simtype(data[0:i+1,:], **extraArg)
        else:
            buysell=simtype(data[0:i], **extraArg)

        if buysell>0 and Holding==0:
            Holding=floor(Worth/X[i+1])
            logging.info('Buying {0:3d} at {1:.3f} worth {2:.3f}'.format(Holding,X[i+1],Holding*X[i+1]))
            if(plot):
                ax1.plot(date[i],X[i+1],'g+',markersize=10, markeredgewidth=2)
            Worth-=Holding*X[i+1]


        elif buysell<0 and Holding>0:
            Worth+=Holding*X[i+1]
            logging.info('Selling {0:3d}, now worth {1:.3f}'.format(Holding,Worth))
            Holding=0
            if(plot):
                ax1.plot(date[i],X[i+1],'r+',markersize=10, markeredgewidth=2)

        WorthSeries.append(Worth+Holding*X[i+1])


    #Sell off what's left

    Worth+=Holding*X[-1]
    WorthSeries.append(Worth)

    if(plot):
        ax1.plot(date,X,'k')
        ax1.set_ylabel('Stock ($)')

        ax2.plot(date,WorthSeries, 'k')
        ax2.plot(date,[amount for i in range(len(WorthSeries))],'r-.')
        ax2.set_ylabel('Investment ($)')
        ax2.set_xlabel('Day')
        f.subplots_adjust(hspace=0)
        plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
        ax1.grid(True, axis='x')
        ax2.grid(True, axis='x')

        if('ticker' in extraArg):
            ax1.set_title('Simulating use of {} for {}'.format(simtype.__name__,extraArg['ticker']))

        #plt.set_size_inches(5, 10)
        plt.savefig('Plots/{}_{}.png'.format(extraArg['ticker'],simtype.__name__))
        plt.show()

    if(returnSeries):
        return WorthSeries
    else:
        return Worth

    #Just so the doxygen call graph picks up that these are possible calls
    if(0):
        IRpred()
        LS()
        dailyrise()
        randombuy()



def resultPrint(method, ticker, start, end):
    '''Prints results in nice fancy colors
    Only tested with bash, but should(?) work for any terminal that
    supports bash-like syntax and coloring
    '''
    if(start>end):
        change="\x1b[1;31m {:2.3f}% \x1b[0m".format((end-start)/start*100.0)
    else:
        change="\x1b[1;32m {:2.3f}% \x1b[0m".format((end-start)/start*100.0)

    print('{} of {} starts at {:3d}, results in {:.3f}. Change:{}'.format(method,ticker,start,end,change))


def ensembleSim(simtype, startingWorth, data, ticker, ensembleNum=100, **extraArg):
    '''Run a model a number of times, look at median results
    Not sure if it technically counts as an "ensemble" model, but
    good way to observe effects of random variations in models that
    have random variations
    '''

    ensembleRun=np.zeros([ensembleNum,len(simulate(simtype,startingWorth,data,**extraArg,returnSeries=1))])
    for run in range(ensembleNum):
        ensembleRun[run,:] = simulate(simtype,startingWorth,data,**extraArg,returnSeries=1)


    ensMed=np.median(ensembleRun,0)

    date=extraArg.get('date',[i for i in range(len(ensMed))])


    plt.plot(date,ensMed, 'k')
    plt.plot(date,[startingWorth for i in range(len(ensMed))],'r-.')

    plt.fill_between(date,ensMed-2.0*np.std(ensembleRun,0), ensMed+2.0*np.std(ensembleRun,0), color='b', alpha=0.2)

    plt.ylabel('Value ($)')
    plt.xlabel('Date')
    plt.title('Simulating use of {} for {}'.format(simtype.__name__,ticker))
    plt.savefig('Plots/{}_{}_ensemble.png'.format(ticker,simtype.__name__))
    plt.show()


def CommonDate(dateX,X,dateF,F):
    '''Finds intersection of dates between X and F
    Returns a version of F pruned to only have dates that exist
    in X. You can just use dateX for both at that point.
    '''
    Xdic=dict(zip(dateX,X))
    Fdic=dict(zip(dateF,F))
    common=(Xdic.keys() & Fdic.keys())
    F=[Fdic[i] for i in sorted(common)]
    return F

def OptimizeIR(data,startingWorth,**extraArg):
    '''Optimize IR model parameters
    Right now it's just brute force range searching, one variable at a time.
    Nesting would require some pruning of results,
    and should definitely have checks for overfitting.
    More proof of concept than anything useful yet
    '''
    Nx=extraArg.get('Nx',1)
    Nf=extraArg.get('Nf',5)
    ratio=extraArg.get('ratio',0.03)
    days=extraArg.get('days',5)
    ticker=extraArg.get('ticker','ABX')
    #simulate(IRpred,start,data,Nx=Nx,Nf=Nf,ratio=ratio,days=days)
    for Nxtry in range(0,10):
        resultPrint('IR - Nx {}'.format(Nxtry),ticker,startingWorth,simulate(IRpred,startingWorth,data,Nx=Nxtry,Nf=Nf,ratio=ratio,days=days))
    for Nftry in range(0,10):
        resultPrint('IR - Nf {}'.format(Nftry),ticker,startingWorth,simulate(IRpred,startingWorth,data,Nx=Nx,Nf=Nftry,ratio=ratio,days=days))
    for ratiotry in np.arange(0.0,0.1,0.01):
        resultPrint('IR - ratio {}'.format(ratiotry),ticker,startingWorth,simulate(IRpred,startingWorth,data,Nx=Nx,Nf=Nf,ratio=ratiotry,days=days))
    for daystry in [1,5,6,10,12,15]:
        resultPrint('IR - days {}'.format(daystry),ticker,startingWorth,simulate(IRpred,startingWorth,data,Nx=Nx,Nf=Nf,ratio=ratio,days=daystry))



def main(argv):
    opts, args = getopt.getopt(argv[1:],"hlt:",["log=","ticker="])
    for opt, arg in opts:
        if opt=="-h":
            print('python read.py')
        elif opt in ("-l","--log"):
            loglevel=arg
        elif opt in ("-t","--ticker"):
            ticker=arg

    if('ticker' not in vars()):
        ticker = 'ABX'
    if('loglevel' not in vars()):
        loglevel='WARNING'

    logging.basicConfig(format='%(levelname)s:%(message)s',level=loglevel)

    (dateX,X)=getClosing(ticker)
    #Because this somehow gets reversed
    dateX.sort()

    startingWorth=10000
    resultPrint('Randombuy',ticker,startingWorth,simulate(randombuy,startingWorth,X))
    resultPrint('Weekbuy',ticker,startingWorth,simulate(weeklyrise,startingWorth,X,difference=0.1))
    resultPrint('Dailybuy',ticker,startingWorth,simulate(dailyrise,startingWorth,X,difference=0.05))

    #Get extra variables and strip them to times that exist with X
    (dateF,F)=getClosing('^DJI')
    (dateF2, F2)=getClosing('^GSPC')
    (dateF3, F3)=getClosing('^IXIC')
    (dateF4, F4)=getClosing('GE')
    F=CommonDate(dateX,X,dateF,F)
    #F=np.random.randn(1,len(F)).tolist()[0]
    F2=CommonDate(dateX,X,dateF2,F2)
    F3=CommonDate(dateX,X,dateF3,F3)
    F4=CommonDate(dateX,X,dateF4,F4)


    resultPrint('LS',ticker,startingWorth,simulate(LS,startingWorth,np.array([X,F]).T,difference=0.05))
    resultPrint('IR',ticker,startingWorth,simulate(IRpred,startingWorth,np.array([X,F]).T,Nx=1,Nf=7,ratio=0.03,days=5))
    resultPrint('IR2',ticker,startingWorth,simulate(IRpred,startingWorth,np.array([X,F,F2]).T,Nx=1,Nf=7,ratio=0.03,days=5))
    resultPrint('IR3',ticker,startingWorth,simulate(IRpred,startingWorth,np.array([X,F,F2,F3]).T,Nx=1,Nf=7,ratio=0.03,days=5))
    resultPrint('IR4',ticker,startingWorth,simulate(IRpred,startingWorth,np.array([X,F,F2,F3,F4]).T,Nx=1,Nf=7,ratio=0.03,days=5))
    resultPrint('IR5',ticker,startingWorth,simulate(IRpred,startingWorth,np.array([X,F,F4]).T,Nx=1,Nf=7,ratio=0.03,days=5))



    #Simulate and plot ensemble  model. Only really useful for models with random/variable component
    ensembleSim(randombuy,startingWorth,np.array([X,F]).T,ticker)

    simulate(IRpred,startingWorth,np.array([X,F]).T,Nx=1,Nf=7,ratio=0.03,days=5,plot=1,ticker=ticker,date=dateX)
    simulate(IRpred,startingWorth,np.array([X,F,F2]).T,Nx=1,Nf=7,ratio=0.03,days=5,plot=1,ticker=ticker,date=dateX)

    #OptimizeIR(np.array([X,F]).T,startingWorth,Nx=1,Nf=2,ratio=0.03,days=6, ticker=ticker)


if __name__ == "__main__":
    main(sys.argv)
