import numpy as np

def cond_prob(X_counts,toxic,csmooth=1):
    """cond_prob
    Compute conditional probabilities of toxic/non-toxic words after tokenization, and 
    count vectorization. 

    Intput: X_counts - sparse matrix of counts of each word in a given message
    toxic - whether word was toxic or not, with 0,1
    csmooth - parameter for Laplace/Lidstone smoothing to account for unseen words

    Return:
    ptoxic      - total probability for toxic message
    pword_toxic - conditional probability for word being toxic
    pword_clean - conditional probability for word being clean
    """
    nrows,nwords=X_counts.shape
    ptoxic = np.sum(toxic)/nrows
    
    toxic_mat=X_counts[toxic==1,:]
    clean_mat=X_counts[toxic==0,:]
    #sum across messages
    nword_toxic=np.sum(toxic_mat,axis=0)
    nword_clean=np.sum(clean_mat,axis=0)    

    #estimate probability of word given toxicity by number of times
    #that word occurs in toxic documents, divided by the total number of words
    #in toxic documents
    #Laplace/Lidstone smooth version
    pword_toxic= (nword_toxic+csmooth) \
                / (np.sum(toxic_mat)+nwords*csmooth)

    pword_clean= (nword_clean+csmooth) \
                /(np.sum(clean_mat)+nwords*csmooth)
    x1=np.sum(toxic_mat,0)
    x2=nword_toxic
    return ptoxic,pword_toxic,pword_clean    

def naive_bayes(mat,pword_tox,pword_cln,ptox):
    """Compute probability that a message 
    is toxic via naive_bayes estimate.
    """
    #I screwed up using prod_i[p(w_i|T)p(T)]
    #instead of P(T)prod_i[p(w_i|T)].  Ugh.
    #log probability for toxic/clean comments
    log_Tword = np.log(pword_tox)
    log_Cword = np.log(pword_cln)
    ## now accumulate probabilities by multiplying number of counts
    #per comment, with the weights per word
    #also add on log-normalization.
    msk=mat>0
    log_Tscore = mat.dot(log_Tword.T)+np.log(ptox)
    log_Cscore = mat.dot(log_Cword.T)+np.log(1-ptox)
    #predict based on which has larger probability (or log-likelihood)
    pred=log_Tscore>log_Cscore
    #also output probabilities
    prob=1/(1+np.exp(log_Cscore-log_Tscore))

    return pred,prob,log_Tscore,log_Cscore,log_Tword,log_Cword
