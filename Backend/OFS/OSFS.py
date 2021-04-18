import numpy as np
from scipy import special
import scipy.io
from dtit import dtit
import pandas as pd
from pycit import citest
import data

def run_SOSFS(X,Y,w=0,param=0):
    Y = np.transpose(np.array([Y]))
    data = np.append(X,Y,axis=1)
    return np.array(run_OSFS(data=data, class_index=data.shape[1] -1, alpha=param['alpha'])),param

def run_OSFS(data,class_index,alpha):
    n,p=np.shape(data)
    ns=np.max(data)
    # selected_features = set()
    # selected_features1 = set()
    # b= set()
    selected_features = []
    selected_features1 = []
    b= []

    for i in range(p-1):
        n1=np.sum(data[:i])
        if n1==0:
            continue

        stop=0
        CI=1

        # CI=ind(data,i,class_index,[],test,alpha,ns)
        CI,dep,r=cond_indep_fisher_z(data, i,class_index,[],n,alpha)

        if CI==1 or dep is None:
            continue

        if CI==0:
            stop=1
            selected_features.append(i)

        if stop:
            print("Stoppp")
            p2=len(selected_features)
            selected_features1=selected_features

            for j in range(0,p2):
                # b=[]##fix
                b=list(set(selected_features1).difference(data[selected_features[j]]))

                if not len(b)==0:
                    CI,dep=compter_dep_2(b,selected_features[j],class_index,3,0,alpha,'z',data)
                    if CI==1 or dep is None:
                        selected_features1=b

        selected_features=selected_features1
        print(i)
        print(selected_features)
        print("------------------------------")

    return selected_features





def compter_dep_2(bcf, var, target, max_k, discrete, alpha, test, data):
    dep1=0
    x=0
    n_bcf=len(bcf)
    code=bcf
    N=data.shape[0]
    max_cond_size=max_k
    CI=0
    p_value=1
    if max_cond_size>n_bcf:
        max_cond_size=n_bcf

    cond=[]
    cond_size=1

    while cond_size<=max_cond_size:
        cond_index=np.zeros(cond_size,dtype=np.int)
        for i in range(cond_size):
            cond_index[i]=i
        stop=0

        while stop==0:
            cond=[]
            for i in range(cond_size):
                cond=[*cond, code[cond_index[i]]]

            if discrete!=1:
                # ns=max(data)
                # CI=cond_indep_fisher_z(data, var, target, cond)
                CI,r,p_value=cond_indep_fisher_z(data,var,target,cond,N,alpha)
                x=r

            if CI==1 or x is None:
                stop=1
                cond_size=max_cond_size+1

            if stop==0:
                cond_index,stop= next_cond_index(n_bcf,cond_size,cond_index)

        cond_size=cond_size+1
    dep1=x

    return CI,dep1

# def ind(data,alpha,x,y,s=None):
#     # print(data[:,x])
#
#     if not s:
#         p=dtit.test(data[:,[x]],data[:,[y]])
#     else:
#         p=dtit.test(data[:,[x]],data[:,[y]],data[:,[s]])
#     # p1=dtit.test(data[:,[x]],data[:,[y]])
#     # p2=dtit.test(data[:,[x]],data[:,[y]],data[:,[s]])
#     return p<=alpha

def cond_indep_fisher_z(data, X, Y, S, N, alpha=0.05):
    C=np.cov(data[:,[X,Y,*S]])
    size_C=C.shape[1]
    X1=1
    Y1=2
    S1=list(range(3, size_C))

    r,c=partial_corr_coef(C,X1,Y1,S1)
    z=0.5*np.log((1+r)/(1-r))
    z0=0
    W=np.sqrt(N-len(S1)-3)*(z-z0)
    cutoff=norminv(1-0.5*alpha)

    if np.abs(W)<cutoff:
        CI=1
    else:
        CI=0

    p=normcdf(W)
    r=np.abs(r)

    return CI,r,p



# def cond_indep_fisher_z(data, X, Y, S, N, alpha=0.05, ns=None):
#     if ns is None:
#         ns=np.amax(data,0)
#     N=data.shape[0]
#     qi=ns[[s]]
#     tmp=np.cumprod(qi[0:(len(qi)-1)])
#     tmp=np.insert(tmp,0,1)
#
#     dep=-1.0
#     alpha2=1
#
#     try:
#         qs = 1 + (qi - 1).dot(tmp)
#         nijk=np.zeros((ns[[x]],ns[[y]]))
#         tijk=np.zeros((ns[[x]],ns[[y]]))
#     except Exception:
#         print("fsf")


def next_cond_index(n_bcf,cond_size,cond_index1):
    stop=1
    i=cond_size-1
    while i>=1:
        if cond_index1[i]<n_bcf+i-cond_size:
            cond_index1[i]=cond_index1[i]+1
            if i<cond_size:
                for j in range(i+1,cond_size):
                    cond_index1[j]=cond_index1[j-1]+1

            stop=0
            i=-1
        i=i-1
    cond_index=cond_index1
    return cond_index,stop

def partial_corr_coef(S, i, j, Y):
    X=list(range(i,j+1))
    i2=0
    j2=1

    # S[X,X]-S[X,Y]*np.linalg.inv(S[Y,Y])*S[Y,X]

    # S2=S[np.ix_(X,X)]-S[np.ix_(X,Y)]@np.linalg.inv(S[np.ix_(Y,Y)])@S[np.ix_(Y,X)]
    S2 = S[np.ix_(X, X)] - S[np.ix_(X, Y)] @ np.linalg.pinv(S[np.ix_(Y, Y)]) @ S[np.ix_(Y, X)]
    # S2=S[np.ix_(X,X)]-S[np.ix_(X,Y)]@S[np.ix_(Y,X)]
    c=S2[i2,j2]
    r=c/np.sqrt(S2[i2,i2]*S2[j2,j2]) ##fix /2

    return r,c

def norminv(p,mu=0,sigma=1):
    x=np.zeros((1,1))

    k=np.nonzero(sigma <=0 or p<0 or p>1)
    if len(k)>0:
        x[k]=np.full(len(k),np.nan)
    k = np.nonzero(p==0)
    if len(k)>0:
        x[k]=np.full(len(k),-np.inf)
    k = np.nonzero(p==1)
    if len(k)>0:
        x[k]=np.full(len(k),np.inf)
    k = np.nonzero(p > 0  and  p < 1 and sigma > 0)
    if len(k)>0:
        x[k]=np.sqrt(2)*sigma*special.erfinv(2*p-1)+mu
    return x


def normcdf(x,mu=0,sigma=1):
    p = np.zeros((1, 1))

    k1 = np.nonzero(sigma<=0)
    if len(k1) > 0:
        p[k1] = np.full(len(k1), np.nan)

    k = np.nonzero(sigma>0)
    if len(k) > 0:
        p[k] = 0.5*special.erfc(-(x-mu)/(sigma*np.sqrt(2)))

    k2 = np.nonzero(p > 1)
    if len(k2) > 0:
        p[k2] = 1

    return p




if __name__ == '__main__':
    data=np.genfromtxt('C:/Users/Lotan/Desktop/projects/FinalProject/data/spambase.csv',delimiter=',',skip_header=1)
    #print(data)

    # mat = scipy.io.loadmat('C:/Users/Lotan/Desktop/projects/FinalProject/data/wdbc.mat')

    x=run_OSFS(data,57,0.05)
    print(len(x))
    # print(x)
    # cond_indep_fisher_z(data, 2, 4, [1],5)
    # cond_indep_fisher_z(data, None, None, [])


    # x = np.zeros((1,1))
    # sigma=0
    # p=0.6
    # k=np.nonzero(sigma <=0 or p<0 or p>1)
    #
    # x=c



    # x=0
    # print(x==False)


