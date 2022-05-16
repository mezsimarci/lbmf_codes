import numpy as np
import matplotlib.pyplot as plt
def differential_equation(loadbalance,gamma,lmbd,mu,B,dt,Tdiff,d=2,threshold=[5],prec=3,plot=False,documentation=False,p=1):
    step=int(Tdiff/dt)
    N=sum(gamma)
    gammarat=[gamma[i]/N for i in range(len(gamma))]
    if loadbalance=='random':
        [v,t]=random(gammarat,lmbd,mu,B,dt,step)
    elif loadbalance=='jiq':
        [v,t]=jiq(gammarat,lmbd,mu,B,dt,step,p)
    elif loadbalance=='jsq':
        [v,t]=jsq(gammarat,lmbd,mu,B,dt,step,p)
    elif loadbalance=='jsq-d':
        [v,t]=jsqd(gammarat,lmbd,mu,d,B,dt,step,p)
    elif loadbalance=='jbt':
        if type(threshold)==int:
            threshold=[threshold]
        [v,t]=jbt(gammarat,lmbd,mu,threshold,B,dt,step,p)
    else:
         raise ValueError('Incorrect load balancing method')
    if plot:
        plotter_diff(v,t)
    if documentation:
        if loadbalance == 'jsq-d':
            loadbalance = loadbalance + '('+str(d)+')'
        filename = 'diff'+loadbalance+str(lmbd)+'p'+str(p)
        documenter_diff(v,t,prec=6,filename=filename)
    nu=[]
    for i in range(len(v)):
        nu.append(v[i][:,len(v[0][0])-1])
    return v,nu

def random(gammarat,lmbd,mu,B,dt,step):
    K=len(gammarat)
    v=[np.zeros([B+1,step]) for i in range(K)]
    for i in range(K):
        v[i][0,0]=gammarat[i]
        for j in range(1,B+1):
            v[i][j,0]=0
    t=np.zeros(step)
    for n in range(1,step):
        for i in range(K):
            v[i][0,n]=v[i][0,n-1]+dt*(-lmbd*v[i][0,n-1]+mu[i][1]*v[i][1,n-1])
            for j in range(1,B):
                v[i][j,n]=v[i][j,n-1]+dt*(lmbd*v[i][j-1,n-1]-lmbd*v[i][j,n-1]+mu[i][j+1]*v[i][j+1,n-1]-mu[i][j]*v[i][j,n-1])
            v[i][B,n]=v[i][B,n-1]+dt*(lmbd*v[i][B-1,n-1]-mu[i][B]*v[i][B,n-1])
        t[n]=t[n-1]+dt
    return v,t

def jiq(gammarat,lmbd,mu,B,dt,step,p=1):
    L=len(gammarat)
    v=[np.zeros([B+1,step]) for i in range(L)]
    for i in range(L):
        v[i][0,0]=gammarat[i]
        for j in range(1,B+1):
            v[i][j,0]=0
    t=np.zeros(step)
    free = [0 for i in range(L)]
    for n in range(1,step):
        nullsum=0
        upkeep=0
        for i in range(L):
            nullsum+=v[i][0,n-1]
            if free[i]==1:
                upkeep+=mu[i][1]*v[i][1,n-1]
        t[n]=t[n-1]+dt
        for i in range(L):
            if free[i]==0:
                v[i][0,n]=v[i][0,n-1] + dt*((-(p*lmbd-upkeep)*(v[i][0,n-1]/nullsum)-lmbd*((1-p)*v[i][0,n-1])) + mu[i][1]*v[i][1,n-1])
                v[i][1,n]=v[i][1,n-1] + dt*(((p*lmbd-upkeep)*(v[i][0,n-1]/nullsum)+lmbd*((1-p)*v[i][0,n-1]))- (lmbd)*(1-p)*v[i][1,n-1]+mu[i][2]*v[i][2,n-1]  - mu[i][1]*v[i][1,n-1])
                for j in range(2,B):
                    v[i][j,n]=v[i][j,n-1]+dt*((lmbd)*(1-p)*v[i][j-1,n-1]-(lmbd)*(1-p)*v[i][j,n-1]+mu[i][j+1]*v[i][j+1,n-1]-mu[i][j]*v[i][j,n-1])
                v[i][B,n]=v[i][B,n-1]+dt*((lmbd)*(1-p)*v[i][B-1,n-1]-mu[i][B]*v[i][B,n-1])
                if v[i][0,n]<0:
                    dt0=dt*v[i][0,n-1]/(v[i][0,n-1]-v[i][0,n])
                    v[i][0,n]=0
                    v[i][1,n]=v[i][1,n-1]+dt0*(((p*lmbd-upkeep)*(v[i][0,n-1]/nullsum)+lmbd*((1-p)*v[i][0,n-1])) - mu[i][1]*v[i][1,n-1])
                    for j in range(2,B):
                        v[i][j,n]=v[i][j,n-1]+dt0*((lmbd)*(1-p)*v[i][j-1,n-1]-(lmbd)*(1-p)*v[i][j,n-1]+mu[i][j+1]*v[i][j+1,n-1]-mu[i][j]*v[i][j,n-1])
                    v[i][B,n]=v[i][B,n-1]+dt0*((lmbd)*(1-p)*v[i][B-1,n-1]-mu[i][B]*v[i][B,n-1])
                    free[i]=1
                    t[n]=t[n-1]+dt0
                        
            elif free[i]==1:
                v[i][0,n]=v[i][0,n-1]+0
                v[i][1,n]=v[i][1,n-1]+dt*(-((lmbd-upkeep))*v[i][1,n-1]+mu[i][2]*v[i][2,n-1])
                for j in range(2,B):
                    v[i][j,n]=v[i][j,n-1]+dt*(((lmbd-upkeep))*v[i][j-1,n-1] - (((lmbd-upkeep))*v[i][j,n-1]) + mu[i][j+1]*v[i][j+1,n-1] - mu[i][j]*v[i][j,n-1])
                v[i][B,n]=v[i][B,n-1]+dt*(((lmbd-upkeep))*v[i][B-1,n-1] - mu[i][B]*v[i][B,n-1])
            else:
                for j in range(B+1):
                    v[i][j,n]=v[i][j,n-1]+0
    return v,t

def jsq(gammarat,lmbd,mu,B,dt,step,p=1):
    L=len(gammarat)
    v=[np.zeros([B+1,step]) for i in range(L)]
    for i in range(L):
        v[i][0,0]=gammarat[i]
        for j in range(1,B+1):
            v[i][j,0]=0
    t=np.zeros(step)
    free = [0 for i in range(L)]
    for n in range(1,step):
        shortsum=0
        upkeep=0
        mn=max(free)
        t[n]=t[n-1]+dt
        for i in range(L):
            shortsum+=v[i][mn,n-1]
            upkeep+=mu[i][free[i]]*v[i][free[i],n-1]
        for i in range(L):
            for j in range(mn):
                v[i][j,n]=0
            v[i][mn,n]=v[i][mn,n-1] + dt*((-(p*lmbd-upkeep)*(v[i][mn,n-1]/shortsum)-lmbd*((1-p)*v[i][mn,n-1])) + mu[i][mn+1]*v[i][mn+1,n-1])
            v[i][mn+1,n]=v[i][mn+1,n-1] + dt*(((p*lmbd-upkeep)*(v[i][mn,n-1]/shortsum)+lmbd*((1-p)*v[i][mn,n-1]))- (lmbd)*(1-p)*v[i][mn+1,n-1]+mu[i][mn+2]*v[i][mn+2,n-1]  - mu[i][mn+1]*v[i][mn+1,n-1])
            for j in range(mn+2,B):
                v[i][j,n]=v[i][j,n-1]+dt*((lmbd)*(1-p)*v[i][j-1,n-1]-(lmbd)*(1-p)*v[i][j,n-1]+mu[i][j+1]*v[i][j+1,n-1]-mu[i][j]*v[i][j,n-1])
            v[i][B,n]=v[i][B,n-1]+dt*((lmbd)*(1-p)*v[i][B-1,n-1]-mu[i][B]*v[i][B,n-1])
        for ix in range(L):
            if v[ix][mn,n]<0:
                for i in range(L):
                    dt0=dt*v[i][mn,n-1]/(v[i][mn,n-1]-v[i][mn,n])
                    for j in range(mn):
                        v[i][j,n]=0
                    v[i][mn,n]=v[i][mn,n-1]+dt0*((-(p*lmbd-upkeep)*(v[i][mn,n-1]/shortsum)-lmbd*((1-p)*v[i][mn,n-1])) + mu[i][mn+1]*v[i][mn+1,n-1])
                    v[i][mn+1,n]=v[i][mn+1,n-1]+dt0*(((p*lmbd-upkeep)*(v[i][mn,n-1]/shortsum)+lmbd*((1-p)*v[i][mn,n-1]))- (lmbd)*(1-p)*v[i][mn+1,n-1]+mu[i][mn+2]*v[i][mn+2,n-1]  - mu[i][mn+1]*v[i][mn+1,n-1])
                    for j in range(mn+2,B-1):
                        v[i][j,n]=v[i][j,n-1]+dt0*((lmbd)*(1-p)*v[i][j-1,n-1]-(lmbd)*(1-p)*v[i][j,n-1]+mu[i][j+1]*v[i][j+1,n-1]-mu[i][j]*v[i][j,n-1])
                    v[i][B,n]=v[i][B,n-1]+dt0*((lmbd)*(1-p)*v[i][B-1,n-1]-mu[i][B]*v[i][B,n-1])
                    t[n]=t[n-1]+dt0
                for i in range(L):
                    free[i]=free[i]+1
    return v,t

def jsqd(gammarat,lmbd,mu,d,B,dt,step,p=1):
    L=len(gammarat)
    v=[np.zeros([B+1,step]) for i in range(L)]
    for i in range(L):
        v[i][0,0]=gammarat[i]
        for j in range(1,B+1):
            v[i][j,0]=0
    t=np.zeros(step)
    for n in range(1,step):
        y=np.zeros([L,B+1])
        z=np.zeros(B+1)
        for j in range(B+1):
            for i in range(L):
                y[i,j]=sum([v[i][k,n-1] for k in range(j,B+1)])
            z[j]=sum(y[l,j] for l in range(L))
        for i in range(L):
            if sum([v[i][0,n-1] for i in range(L)])!=0:
                v[i][0,n]=v[i][0,n-1] + dt*((-lmbd*((p*(v[i][0,n-1]/sum([v[i][0,n-1] for i in range(L)]))*((z[0]**d)-(z[1]**d)))+(1-p)*v[i][0,n-1]))+mu[i][1]*v[i][1,n-1])
            else:
                v[i][0,n]=v[i][0,n-1] + dt*(mu[i][1]*v[i][1,n-1])
            for j in range(1,B):
                if sum([v[i][j-1,n-1] for i in range(L)])!=0 and sum([v[i][j,n-1] for i in range(L)])!=0:
                    v[i][j,n]=v[i][j,n-1] + dt*(((-lmbd*((p*(v[i][j,n-1]/sum([v[i][j,n-1] for i in range(L)]))*((z[j]**d)-(z[j+1]**d)))+(1-p)*v[i][j,n-1])))+mu[i][j+1]*v[i][j+1,n-1]+(lmbd*((p*(v[i][j-1,n-1]/sum([v[i][j-1,n-1] for i in range(L)]))*((z[j-1]**d)-(z[j]**d)))+(1-p)*v[i][j-1,n-1]))-mu[i][j]*v[i][j,n-1])
                elif sum([v[i][j,n-1] for i in range(L)])==0 and sum([v[i][j-1,n-1] for i in range(L)])!=0:
                    v[i][j,n]=v[i][j,n-1] + dt*(mu[i][j+1]*v[i][j+1,n-1]+(lmbd*((p*(v[i][j-1,n-1]/sum([v[i][j-1,n-1] for i in range(L)]))*((z[j-1]**d)-(z[j]**d)))+(1-p)*v[i][j-1,n-1]))-mu[i][j]*v[i][j,n-1])
                elif sum([v[i][j-1,n-1] for i in range(L)])==0 and sum([v[i][j,n-1] for i in range(L)])!=0:
                    v[i][j,n]=v[i][j,n-1] + dt*((-lmbd*((p*(v[i][j,n-1]/sum([v[i][j,n-1] for i in range(L)]))*((z[j]**d)-(z[j+1]**d)))+(1-p)*v[i][j,n-1]))+mu[i][j+1]*v[i][j+1,n-1]-mu[i][j]*v[i][j,n-1])
                else:
                    v[i][j,n]=v[i][j,n-1] + dt*(mu[i][j+1]*v[i][j+1,n-1]-mu[i][j]*v[i][j,n-1])
            if sum([v[i][B-1,n-1] for i in range(L)])!=0:
                v[i][B,n]=v[i][B,n-1] + dt*((lmbd*((p*(v[i][B-1,n-1]/sum([v[i][B-1,n-1] for i in range(L)]))*((z[B-1]**d)-(z[B]**d)))+(1-p)*v[i][B-1,n-1]))-mu[i][B]*v[i][B,n-1])
            else:
                v[i][B,n]=v[i][B,n-1] + dt*(-mu[i][B]*v[i][B,n-1])
        t[n]=t[n-1]+dt
    return v,t

def jbt(gammarat,lmbd,mu,threshold,B,dt,step,p): #different thresholds for the server types
    L=len(gammarat)
    v=[np.zeros([B+1,step]) for i in range(L)]
    for i in range(L):
        v[i][0,0]=gammarat[i]
        for j in range(1,B+1):
            v[i][j,0]=0
    t=np.zeros(step)
    free=0
    for n in range(1,step):
        if free==0:
            x=sum(v[i][j,n-1] for i in range(L) for j in range(threshold[i]))
            for i in range(L):
                v[i][0,n]=v[i][0,n-1] + dt*( - lmbd*((p*v[i][0,n-1]/x)+((1-p)*v[i][0,n-1])) + mu[i][1]*v[i][1,n-1])
                for m in range(1,threshold[i]):
                    v[i][m,n]=v[i][m,n-1] + dt*( - lmbd*((p*v[i][m,n-1]/x)+((1-p)*v[i][m,n-1])) + lmbd*((p*v[i][m-1,n-1]/x)+((1-p)*v[i][m-1,n-1])) - mu[i][m]*v[i][m,n-1] + mu[i][m+1]*v[i][m+1,n-1])
                v[i][threshold[i],n]=v[i][threshold[i],n-1] + dt*(-lmbd*(1-p)*v[i][threshold[i],n-1] + lmbd*((p*v[i][threshold[i]-1,n-1]/x)+((1-p)*v[i][threshold[i]-1,n-1])) - mu[i][threshold[i]]*v[i][threshold[i],n-1] + mu[i][threshold[i]+1]*v[i][threshold[i]+1,n-1])
                for m in range(threshold[i]+1,B):
                    v[i][m,n]=v[i][m,n-1]+dt*(-lmbd*(1-p)*v[i][m,n-1] + lmbd*(1-p)*v[i][m-1,n-1] - mu[i][m]*v[i][m,n-1] + mu[i][m+1]*v[i][m+1,n-1])
                v[i][B,n]=v[i][B,n-1]+dt*(lmbd*(1-p)*v[i][B-1,n-1] - mu[i][B]*v[i][B,n-1])
            t[n]=t[n-1]+dt
        
        if min([v[i][j,n] for i in range(L) for j in range(threshold[i])]) < 0:
            raise Exception('Threshold needs to be raised!')
    return v,t

def plotter_diff(v,t):
    for i in range(len(v)):
        plt.figure()
        for j in range(len(v[i])):
            plt.plot(t,v[i][j],label=str(j))
        plt.legend()
        plt.show()
        
def documenter_diff(v,t,prec,filename):
    #filename=input("What should be the name of the file?")    #this is only placeholder
    with open(filename+'.csv','w') as f:
        for i in range(len(t)):
            f.write(str(round(t[i],prec)))
            for j in range(len(v)):
                f.write(','+str(list(np.round(v[j][:,i],prec))))
                f.write('\n')