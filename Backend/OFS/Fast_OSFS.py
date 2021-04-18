import numpy as np
from OSFS import cond_indep_fisher_z,compter_dep_2

def run_S_Fast_OSFS(X,Y,w=0,param=0):
    Y = np.transpose(np.array([Y]))
    data = np.append(X,Y,axis=1)
    return np.array(run_Fast_OSFS(data=data, class_index=data.shape[1] -1, alpha=param['alpha'])),param

def run_Fast_OSFS(data,class_index,alpha):
    n,p=np.shape(data)
    # ns=np.max(data)
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

        if stop:
            if not len(selected_features)==0:
                CI,dep=compter_dep_2(selected_features,i,class_index,3,0,alpha,'z',data)

            if CI==0 and dep is not None:
                selected_features.append(i)
                p2=len(selected_features)
                selected_features1=selected_features

                for j in range(0,p2):
                    b = list(set(selected_features1).difference(data[selected_features[j]]))
                    if not len(b) == 0:
                        # CI,dep=optimal_compter_dep_2(b,selected_features[j],class_index,3,0,alpha,'z',data)
                        CI,dep=compter_dep_2(b,selected_features[j],class_index,3,0,alpha,'z',data)


                        if CI==1 or dep is None:
                            selected_features1=b

        selected_features = selected_features1
        print(i)
        print(selected_features)
        print("------------------------------")

    return selected_features

def optimal_compter_dep_2(bcf, var, target, max_k, discrete, alpha, test, data):
    dep1=0
    x=0
    n_pc=len(bcf)
    code=bcf
    N=data.shape[0]
    max_cond_size=max_k
    CI=0
    p=1
    if max_cond_size>n_pc:
        max_cond_size=n_pc

    cond=[]
    cond_size=1

    while cond_size<max_cond_size:
        cond_index=np.zeros(cond_size,dtype=np.int)
        for i in range(cond_size):
            cond_index[i]=i
        stop=0

        while stop==0:
            cond=[]
            for i in range(cond_size):
                if i==cond_size-1:
                    cond_index[i]=n_pc
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
                cond_index,stop= next_cond_index(n_pc,cond_size,cond_index)

        cond_size=cond_size+1
    dep1=x

    return CI,dep1

def next_cond_index(n_pc, cond_size, cond_index1):
    stop=1
    i=cond_size-1
    while i>=1:
        if cond_index1[i]<n_pc+i-cond_size:
            if i==cond_size:
                cond_index1[i]=n_pc+i-cond_size-1
            else:
                cond_index1[i]=cond_index1[i]+1

            if i<cond_size:
                for j in range(i+1,cond_size):
                    if(j==cond_size-1):
                        cond_index1[j]=n_pc
                    else:
                        cond_index1[j]=cond_index1[j-1]+1
            stop=0
            i = -1
        i=i-1
    cond_index=cond_index1
    return cond_index,stop


if __name__ == '__main__':
    data=np.genfromtxt('C:/Users/Lotan/Desktop/projects/FinalProject/data/spambaseC.csv',delimiter=',',skip_header=1)
    #print(data)
    # data=np.genfromtxt('C:/Users/Lotan/Desktop/projects/FinalProject/data/spambase.csv',delimiter=',',skip_header=1)


    # mat = scipy.io.loadmat('C:/Users/Lotan/Desktop/projects/FinalProject/data/wdbc.mat')

    x=run_Fast_OSFS(data,57,0.05)
    print(len(x))