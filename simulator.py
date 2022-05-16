import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import pandas as pd
import os

def run(N,gamma,B,lmbd,mu,T,T1,T2,serverog,plot=False,documentation=False,plotdensity=100,loadbalance="jsq",threshold=[5 for i in range(1)],d=2,p=1,histogram=False,histbins=30,histmid=1,param=""):
    if type(gamma)==int:
        gamma=[gamma]
    K=len(gamma)
    server=list(np.copy(serverog))
    x=[]
    for i in range(K):
        x.append(np.array([len(server[i][server[i]==j]) for j in range(B+1)]))    #the number of the different kinds of servers by initial queue lenght
    st=[]
    stlen=[]
    histspace = None
    histlist= None
    if histogram:
        histlist=np.zeros(histbins)
        histspace=np.linspace(0,4*histmid,num=histbins)
    for i in range(K):
        st.append([])
        stlen.append([])
        for j in range(gamma[i]):
            st[i].append(list(np.zeros(server[i][j])))    #it keeps track of the members of the queue by time of arrival
            stlen[i].append(list(np.zeros(server[i][j])))
    t=0    #this denominates the time running variable
    stsum=0    #the sum of serving time, later used in calculating mean serving time
    stsqsum=0    #the sum of serving time, later used in calculating mean serving time
    stcount=0    #the number of served demands, later used in calculating mean serving time
    wtsum=0    #the sum of waiting time, later used in calculating mean waiting time
    wtcount=0    #the number of waiting demands, later used in calculating mean waiting time
    stsumlist=[0 for i in range(K)]    #the list of sums of serving time by server types, later used in calculating mean serving time
    stsqsumlist=[0 for i in range(K)]    #the list of squared sums of serving time by server types, later used in calculating mean serving time
    stlensumlist=[[0 for i in range(B+1)] for j in range(K)]
    stcountlist=[0 for i in range(K)]    #the list of numbers of served demands by server types, later used in calculating mean serving time
    stlencountlist=[[0 for i in range(B+1)] for j in range(K)]
    wtsumlist=[0 for i in range(K)]    #the list of sums of waiting time by server types, later used in calculating mean serving time
    wtsqsumlist=[0 for i in range(K)]    #the list of squared sums of waiting time by server types, later used in calculating mean serving time
    wtcountlist=[0 for i in range(K)]    #the list of numbers of waiting demands by server types, later used in calculating mean waiting time
    xlast=np.array(x)
    tlast=0
    if histogram:
        alldem = 0
        lostdem = 0
    xplot=[]
    tplot=[]
    servermu=[]
    for i in range(K):
        servermu.append([])
        for j in range(gamma[i]):
            servermu[i].append(mu[i][server[i][j]])
    while True:
        musum=0
        for i in range(K):    #this segment calculates the total serving rate
            for j in range(gamma[i]):
                musum+=servermu[i][j]
        S=N*lmbd+musum    #the rate of the next state change, by the racing clocks model
        dt=np.random.exponential(1/S)    #the time until the next state change
        lastt=t
        t+=dt
        if np.floor(t)!=np.floor(lastt):
            clear_output(wait=True)
            print(str(int(np.floor(t)))+' / '+str(T+T1+T2))
        change=np.random.random()    #this decide the type of the next state change, based on the racing clocks model
        if change<((N*lmbd)/S):
            cb=np.random.random()     #cb is short for can balance, as the system can balance p of all incoming demands
            if cb<=p:
                if loadbalance=="jsq":
                    [p1,p2]=arrival_jsq(gamma,K,server,x)
                elif loadbalance=="random":
                    [p1,p2]=arrival_random(gamma,N,K,server,x)
                elif loadbalance=="jbt":
                    [p1,p2]=arrival_jbt(gamma,N,K,server,x,threshold)
                elif loadbalance=="jiq":
                    idle=[1 for i in range(K)]
                    [p1,p2]=arrival_jbt(gamma,N,K,server,x,idle)
                elif loadbalance=="jsq-d":
                    [p1,p2]=arrival_jsqd(gamma,N,server,x,d)
                else:
                    raise ValueError("Invalid load balancing method")
            else:
                [p1,p2]=arrival_random(gamma,N,K,server,x)
            if histogram:
                alldem += 1
            if server[p1][p2]<B:    #it makes sure that the queue lenght doesn't pass the limit
                x[p1][server[p1][p2]]-=1    #the number of the servers with a queue lenght of the receiver before the reception decreases by 1
                server[p1][p2]+=1    #the row lenght increases by 1 where the new demand arrived
                st[p1][p2].append(t)    #the arrival time of the new demand is being recorded
                stlen[p1][p2].append((t,server[p1][p2]))
                servermu[p1][p2]=mu[p1][server[p1][p2]]    #the serving rate of the server gets adjusted
                #if len(st[p1][p2])==1:    #if the recepient queue was empty, the serving begins immediately
                    #if t>T1 and t<T1+T:
                        #wtsum+=0
                        #wtsumlist[i]+=0
                        #wtcount+=1
                        #wtcountlist[i]+=1
                x[p1][server[p1][p2]]+=1    #the number of the servers with a queue lenght of the receiver after the reception increases by 1
            elif histogram:
                lostdem += 1
        else:
            divpoint=(N*lmbd)/S
            for i in range(K):
                for j in range(gamma[i]):
                        divpoint+=servermu[i][j]/S    #the division point goes through the servers to decide which server the demand is leaving
                        if change<divpoint:
                        
                            x[i][server[i][j]]-=1    #the number of the servers with a queue lenght of the server from which the demand leaves before the leaving decreases by 1
                            server[i][j]-=1   #the queue lenght of the server from which the demand leaves decreases by 1
                            servermu[i][j]=mu[i][server[i][j]]    #the serving rate of the server gets adjusted
                            tmp=st[i][j].pop(0)
                            tmplen=stlen[i][j].pop(0)
                            if not (tmp<=T1 or tmp>T+T1):
                                if tmp!=0:    #it is to avoid distorsion because of initial demands
                                    acst=t-tmp
                                    if histogram:
                                        h=1
                                        while h<histbins and histspace[h]<acst:
                                            h+=1
                                        if acst <= 4*histmid + histspace[1]:
                                            histlist[h-1]+=1
                                    stsum+=acst    #the serving time of the leaving demand gets added to the sum
                                    stsumlist[i]+=acst    #the serving time of the leaving demand gets added to the sum of the type
                                    stlensumlist[i][tmplen[1]]+=acst
                                    stcount+=1
                                    stcountlist[i]+=1
                                    stlencountlist[i][tmplen[1]]+=1
                                    #stsqsum+=(acst)*(acst)    #service time squared
                                    #stsqsumlist[i]+=(acst)*(acst)
                            #if len(st[i][j])>0 and st[i][j][0]!=0:    #this checks if there is a demand next in line which's serving can begin now, and also makes sure to avoid distorsion by the initial demands
                                #if t>T1:
                                    #wtsum+=t-st[i][j][0]    #the waiting time of the demand which's serving begins now gets added to the sum
                                    #wtsumlist[i]+=t-st[i][j][0]    #the waiting time of the demand which's serving begins now gets added to the sum of the type
                                    #wtcount+=1
                                    #wtcountlist[i]+=1
                            x[i][server[i][j]]+=1    #the number of the servers with a queue lenght of the server from which the demand leaves after the leaving increases by 1
                            break
                else:    #this section along with the break a bit lower just serves to break out both loops once we find the appropriate server
                    continue
                break
        if plot or documentation:
            for i in np.arange(tlast,t+plotdensity/100,plotdensity):   #it is for the plotting
                tplot.append(i)
                xplot=list(xplot)
                xplot.append(xlast)
                xplot=np.array(xplot)
            tlast=tplot[-1]+plotdensity
            xlast=np.array(x)
        if t>=T+T1+T2:
            if plot:
                plotter_sim(K,B,tplot,xplot,loadbalance)
            if documentation:
                if loadbalance == 'jsq-d':
                    filename="sim"+loadbalance+'('+str(d)+')'+"N"+str(N)+"lmbd"+str(lmbd)+'p'+str(p)+'v2'
                else:
                    filename="sim"+loadbalance+"N"+str(N)+"lmbd"+str(lmbd)+'p'+str(p)
                documenter_sim(K,tplot,xplot,filename,param)
            if histogram:
                histlist=histlist/(stcount*(histspace[1]))
                histlist = (1-(lostdem/alldem)) * histlist
                histspace = histspace + histspace[1]/2
                plt.figure()
                plt.bar(histspace,histlist,width=histspace[1])
                plt.show()
                cwd = os.getcwd()
                data = pd.DataFrame(histlist,histspace)
                if loadbalance == 'jsq-d':
                    loadbalance += '(' + str(d) + ')'
                if documentation: 
                    data.to_csv(cwd+'/'+loadbalance+str(N)+'_histcsv.csv', header = False,sep = '\t')
            stsumlist=np.array(stsumlist)
            stsqsumlist=np.array(stsqsumlist)
            stcountlist=np.array(stcountlist)
            wtsumlist=np.array(wtsumlist)
            wtcountlist=np.array(wtcountlist)
            return [x,stsum,stcount,stsqsum,wtsum,wtcount,stsumlist,stsqsumlist,stcountlist,wtsumlist,wtcountlist,stlensumlist,stlencountlist,histspace,histlist] # x is the final state

def arrival_random(gamma,N,K,server,x):  #arriving demand random server
    pos=pos=np.random.randint(0,N)    #it randomly determines the server, to which the new demand arrives
    mnrangeend=0
    mnrangestart=0
    for i in range(K):
        mnrangeend+=gamma[i]
        if pos<mnrangeend:    #it calculates which type is the receiving server in
            pos-=mnrangestart
            return [i,pos]
        mnrangestart=mnrangeend        
        
def arrival_jsq(gamma,K,server,x):  #arriving demand, join shortest queue
    mnlist=[]
    for i in range(K):
        mnlist.append(min(server[i]))
    mnlen=min(mnlist)    #it searches for the minimal queue lenght
    mn=[]
    for i in range(K):
        mn.append([])
        for j in range(len(server[i])):
            if server[i][j]==mnlen:
                mn[i].append(j)     #it collects the servers with the shortest queues
    mnno=0
    for i in range(K):
        mnno+=len(mn[i])    #it counts the servers with the shortest queues
    pos=np.random.randint(0,mnno)    #it randomly determines the server, to which the new demand arrives
    mnrangeend=0
    mnrangestart=0
    for i in range(K):
        mnrangeend+=len(mn[i])
        if pos<mnrangeend:    #it calculates which type is the receiving server in
            pos-=mnrangestart
            return [i,mn[i][pos]]
        mnrangestart=mnrangeend
def arrival_jsqd(gamma,N,server,x,d):  #arriving demand, join shortest queue from the selected d
    mnlist=[]
    dposraw=list(np.random.choice(range(N),d,replace=False))
    dpos=[]
    for i in dposraw:
        j=0
        k=i
        while k>=gamma[j]:
            k-=gamma[j]
            j+=1
        dpos.append((j,k))
    dchosen=[]
    for i in dpos:
        dchosen.append(server[i[0]][i[1]])
    mnlen=min(dchosen)    #it searches for the minimal queue lenght
    mn=[]
    for (i,j) in dpos:
        if (server[i][j]==mnlen):
            mn.append((i,j))     #it collects the appropriate servers with the shortest queues
    mnno=len(mn)
    pos=mn[np.random.randint(len(mn))]    #it randomly determines the server, to which the new demand arrives
    return [pos[0],pos[1]]

def arrival_jbt(gamma,N,K,server,x,threshold):
    tr=[]
    for i in range(K):
        tr.append([])
        for j in range(len(server[i])):
            if server[i][j]<threshold[i]:
                tr[i].append(j)     #it collects the servers with queue lenghts within the treshold
    trno=0
    for i in range(K):
        trno+=len(tr[i])    #it counts the servers with queue lenghts within the treshold
    if trno==0:
        return arrival_random(gamma,N,K,server,x)
    else:
        pos=np.random.randint(0,trno)    #it randomly determines the server, to which the new demand arrives
        mnrangeend=0
        mnrangestart=0
        for i in range(K):
            mnrangeend+=len(tr[i])
            if pos<mnrangeend:    #it calculates which type is the receiving server in
                pos-=mnrangestart
                return [i,tr[i][pos]]
            mnrangestart=mnrangeend
            
def plotter_sim(K,B,tplot,xplot,loadbalance):
    for i in range(K):
        plt.figure()
        plt.plot(tplot,xplot[:,i,0],label='0')
        plt.plot(tplot,xplot[:,i,1],label='1')
        plt.plot(tplot,xplot[:,i,2],label='2')
        for j in range(3,B+1):
            plt.plot(tplot,xplot[:,i,j],label=str(j))
        plt.legend(loc='upper right')
        plt.show()
        
def documenter_sim(K,tplot,xplot,filename,param):
    filename=filename+param
    with open(filename+'.csv','w') as f:
        for i in range(len(xplot)):
            f.write(str(tplot[i]))
            for j in range(K):
                f.write(','+str(xplot[i][j]))
            f.write('\n')