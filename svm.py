#define a cost function, check that we're minimizing it.
#define the alternative cost function to be sure we'e also minimize that original choice.
#check constraints are obeyed?
def svm_cost(alpha,Kmat,cat,l):
    m = Kmat.shape[0]
    Ka = np.dot(Kmat,alpha);            
    cost=0.5*l*np.dot( alpha, Ka)
    yvec=(1-cat*Ka)/m
    ymsk=yvec>0
    cost+=np.sum(yvec*ymsk)
    return cost                    

#Compute Kernel Matrix, an m x m matrix
#that effectively measures similarity between inputs.
#each K_{ij} is the "distance" between the weighted inputs,
# $x^{(i)}_k, $x^{(j)}_k$.
# def kernel_matrix(x,tau):
#     m=x.shape[0]
#     x=(x>0).astype(int)
#     K=np.zeros([m,m])
#     for i in range(0,m):
#         K[:,i]=np.exp(-np.sum((x-x[i])**2,1)/(2*tau**2));
#     return K

#Compute a column from the Kernel matrix.
#Matrix is assumed to be [m,m], with vec
#of length m.  Returns vector of length m.
# def Kvec(mat,vec,tau):
#     xarg=np.sum((mat-vec)**2,1)
#     Kv=np.exp(-xarg/(2*tau**2));
#     return Kv

def Kbatch(mat,ind,norm2,tau):
    """Kbatch(mat,cvec,ind,norm2,tau)
    Compute a batch of kernel matrix elements. 
    Input: mat  - sparse matrix (nobs x nfeature)
           ind - indices for that subset of rows (nbatch)
           norm2 - column matrix with squared norm for each (nobs,1)
    Return: Kvecs - nbatch x nobs subset of the full kernel matrix.
    """
    nbatch=len(ind)
    #extract chosen rows
    cvec = mat[ind,:].T
    #relying on numpy broadcasting to join (nobs,nbatch) + (nobs,1)
    xarg=-2.0*mat.dot(cvec)+norm2
    #further broadcasting: use a row-vector ind to make a row-vector
    #of relevant norms.
    #then broadcast again from (nobs,nbatch)+ (1,nbatch)
    xarg+=norm2[ind].T
    Kv=np.exp(-xarg/(2*tau**2));
    return Kv

#carry out update on parameters for given loss function for SVM,
#given parameters, a row-vector of inputs K_i
def svm_batchsgd_update(alpha,Kbatch,y,ind,rate,l):
    """svm_batchsgd_update
    alpha  - nobs x 1 vector
    Kbatch - nobs x Nbatch subset of Kernel matrix
    y      - (1xNbatch) labels for inputs
    ind    - (1xNbatch) indices for batch
    """
    nobs = Kbatch.shape[0]
    yK = np.multiply(Kbatch,y.T)   #nobs x Nbatch 
    yKa = np.dot(alpha.T,yK);
    Kalpha = np.multiply(Kbatch,alpha[ind].T)
    #da= (-y_i*K_i)*((1-y_i*Ka) >0)+m*l*K_i*alpha[ind];
    da= -np.multiply(yK,yKa<-1)+nobs*l*Kalpha;
    #sum all changes over columns
    alpha=alpha-rate*np.sum(da,axis=1);
    return alpha
    
#Fit SVM coefficients for spam with stochastic gradient descent.
#use known categories in cat_vec, and word_matrix with nobs x nwords
def svm_fit(word_mat,cat_vec,tau=8,Nbatch=100):
    #just count whether word occurs.
    new_mat=(word_mat>0).astype(int)
    nobs,nword=new_mat.shape;
    alpha=0.1*np.random.randn(nobs,1)    #initialize parameters
    alpha0=alpha
    alpha_tot=np.zeros((nobs,1))
    niter=int(40*nobs/Nbatch);
    l=1/(tau**2*nobs)
    norm2=new_mat.multiply(new_mat).sum(axis=1)
    #multiple iterations of stochastic gradient descent.
    for t in range(0,niter):
        indx=np.random.randint(low=0,high=nobs,size=Nbatch)        
        Kv = Kbatch(new_mat,indx,norm2,tau)
        yt=cat_vec[indx]
        rate=np.sqrt(np.sqrt(1.0/(t+1)))
        alpha=svm_batchsgd_update(alpha,Kv,yt,indx,rate,l)
        alpha_tot=alpha_tot+alpha
        if (10*t % niter ==0):
            print("Iter {} of {}".format(t,niter))
    alpha_tot=alpha_tot/niter
    return alpha0,alpha_tot

#given parameters, predict the output
def svm_predict(train_mat,test_mat,alpha,tau):
    ntrain=train_mat.shape[0]
    ntest=test_mat.shape[0]
    pred_cat=np.zeros(ntest)
    train_new=(train_mat>0).astype(int)
    test_new=(test_mat>0).astype(int)

    train_norm=train_new.multiply(train_new).sum(axis=1)
    test_norm=test_new.multiply(test_new).sum(axis=1)
    train_test_dot = np.dot(train_new,test_new.T)
    for i in range(0,ntest):
        #compute dot-product of param-vector and column of kernel matrix
        dist2 = train_norm-2*train_test_dot[:,i]+test_norm[i]
        Kvec=np.exp(-dist2/(2*tau**2))
        #Kvec=np.exp(-np.sum((train_new-test_new[i])**2,1)/(2*tau**2))
        pred_size= np.dot(alpha.T,Kvec)
        pred_cat[i] = np.sign(pred_size)
    return pred_cat
