# Contains a bunch of utilities written to help with initial data analysis.
import numpy as np
from sklearn.metrics import log_loss, roc_auc_score
import string
def clean_up(comments):
    """clean_up(comments)
    Removes numbers, some symbols and html trash from a Pandas series of messages.
    """
    com_clean=comments.str.replace('(NEWLINE_TOKEN|\\n|\\t|TAB_TOKEN|TAB)',' ')
    com_clean=comments.str.lower()
    #Remove HTML trash, via non-greedy replacing anything between backticks.
    #Should probably combine into a single regex.
    re_str="(style|class|width|align|cellpadding|cellspacing|rowspan|colspan)=(\`\`|\"\").*?(\`\`|\"\")"
    com_clean=com_clean.str.replace(re_str,' ')
    #remove numbers
    com_clean=com_clean.str.replace("[0-9]+",' ')
    #remove apostrophes without space
    com_clean=com_clean.str.replace("\'",'')
    #remove symbols.    There must be a more comprehensive way of doing this?
    #re_str="[\+\-\=~\?\[\[\{\}=_\;\:\|\(\)\\\/\`\.\,_\"#!$%\^&\*@]+"
    re_str="["+escape_punctuation()+"]+"
    com_clean=com_clean.str.replace(re_str,' ')
    #remove multiple spaces, replace with a single space
    com_clean=com_clean.str.replace('\\s+',' ')
    return com_clean

def escape_punctuation():
    """Puts backslashes on all common punctuation from string module"""
    str_esc=""
    for symbol in string.punctuation:
        str_esc=str_esc+'\{}'.format(symbol)
    return str_esc

def get_subset(frac_perc,dat_mat,labels):
    """get_subset(frac_perc,dat_mat,labels
    Returns random subset of the data and labels.
    Maintains same fraction of toxic/non-toxic data as the full dataset.
    Input: 
    frac_perc: fraction of data to extract
    dat_mat: input data 
    labels:  corresponding labels.
    Returns:
    ind_sub: indices used for extraction
    Xsub : random subarray
    label_sub: corresponding labels for Xsub
    """ 
    #make vector and sample indices for true/false.
    nvec=np.arange(len(labels))
    #get the indices for true/false
    Tvec=nvec[labels]
    Cvec=nvec[~labels]
    #grab a random shuffling of those indices.
    np.random.shuffle(Tvec)
    np.random.shuffle(Cvec)
    #grab some fraction of them.
    it = int(len(Tvec)*frac_perc)
    ic = int(len(Cvec)*frac_perc)
    ind_sub=np.append(Tvec[:it],Cvec[:ic])
    Xsub = dat_mat[ind_sub]
    label_sub = labels[ind_sub].reshape((len(ind_sub),1))
    return ind_sub,Xsub,label_sub

# def check_predictions(pred,actual,epsilon=1E-15):
#     """check_predictions
#     Applies to only a (single class!)
#     Compares predicted class (y_i) against actual class (z_i).
#     Returns the confusion matrix and mean log-loss.
    
#     Log-loss = sum_i{ z_i log[ y_i] }/M

#     Input: pred - predicted values (0,1)
#     actual - true labels 
#     eps    - shift to avoid log(0)
#     Returns: Confusion matrix with [[true positive, false positive],[false negative, true negative]]
#     log-loss - average log-loss
#     """
#     actual=np.reshape(actual,(len(actual),1))
#     pred=np.reshape(pred,(len(actual),1))    
#     print(pred.shape,actual.shape)
#     tp = np.mean((pred==True)&(actual==True))
#     tn = np.mean((pred==False)&(actual==False))
#     fp = np.mean((pred==True)&(actual==False))    
#     fn = np.mean((pred==False)&(actual==True))            
#     scores=np.matrix([[tp,fp],[fn,tn]])
#     print("True Positive {}. False Positive {}".format(tp,fp))
#     print("False Negative {}. True Negative {}".format(fn,tn))
#     pred_num=pred.astype(float)
#     logloss=log_loss(actual,pred_num,eps=epsilon,normalize=True)    
#     #give zero a small correction.
#     #pred_num[pred==False]=epsilon
#     #pred_num[pred==True]=1-epsilon
#     #my (initial) wrong attempt
#     #logloss2=-np.mean(np.multiply(actual,np.log(pred_num)))
#     # logloss2=-np.mean(np.multiply(actual,np.log(pred_num))\
#     #     +np.multiply(1-actual,np.log(1-pred_num)))
#     # print(logloss2)
#     auroc = roc_auc_score(actual,pred)
#     #logloss=0
#     print("Log-loss is {}".format(logloss))
#     print("AUROC is {}".format(auroc))    
#     return scores,logloss



def check_predictions(pred,actual,epsilon=1E-15):
    """check_predictions
    Applies to only a (single class!)
    Compares predicted class (y_i) against actual class (z_i).
    Returns the confusion matrix and mean log-loss.
    
    Log-loss = sum_i{ z_i log[ y_i] }/M

    Input: pred - predicted values (0,1)
    actual - true labels 
    eps    - shift to avoid log(0)
    Returns: Confusion matrix with [[true positive, false positive],[false negative, true negative]]
    log-loss - average log-loss
    """
    # actual=np.reshape(actual,(len(actual),1))
    # pred=np.reshape(pred,(len(actual),1))    
    #print(pred.shape,actual.shape)
    tp = np.mean((pred==True)&(actual==True))
    tn = np.mean((pred==False)&(actual==False))
    fp = np.mean((pred==True)&(actual==False))    
    fn = np.mean((pred==False)&(actual==True))            
    scores=np.matrix([[tp,fp],[fn,tn]])
    print("True Positive {}. False Positive {}".format(tp,fp))
    print("False Negative {}. True Negative {}".format(fn,tn))
    pred_num=pred.astype(float)
    logloss=log_loss(actual,pred_num,eps=epsilon,normalize=True)    
    #give zero a small correction.
    #pred_num[pred==False]=epsilon
    #pred_num[pred==True]=1-epsilon
    #my (initial) wrong attempt
    #logloss2=-np.mean(np.multiply(actual,np.log(pred_num)))
    # logloss2=-np.mean(np.multiply(actual,np.log(pred_num))\
    #     +np.multiply(1-actual,np.log(1-pred_num)))
    # print(logloss2)
    auroc = roc_auc_score(actual,pred)
    #logloss=0
    print("Log-loss is {}".format(logloss))
    print("AUROC is {}".format(auroc))    
    return scores,logloss
