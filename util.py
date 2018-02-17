
def clean_up(comments):
    """clean_up(comments)
    Removes numbers, some symbols and html trash from a Pandas series of messages.
    """
    com_clean=comments.str.replace('NEWLINE_TOKEN',' ')
    com_clean=com_clean.str.replace('TAB_TOKEN',' ')    
    #Remove HTML trash, via non-greedy replacing anything between backticks.
    #Should probably combine into a single regex.
    #re_str="(style|class|width|align|cellpadding|cellspacing|rowspan|colspan)=\`\`.*?\`\`"
    #com_clean=com_clean.str.replace(re_str,' ')
    com_clean=com_clean.str.replace("style=\`\`.*?\`\`",' ')
    com_clean=com_clean.str.replace("class=\`\`.*?\`\`",' ')
    com_clean=com_clean.str.replace("width=\`\`.*?\`\`",' ')
    com_clean=com_clean.str.replace("align=\`\`.*?\`\`",' ')
    com_clean=com_clean.str.replace("cellpadding=\`\`.*?\`\`",' ')
    com_clean=com_clean.str.replace("cellspacing=\`\`.*?\`\`",' ')
    com_clean=com_clean.str.replace("rowspan=\`\`.*?\`\`",' ')
    com_clean=com_clean.str.replace("colspan=\`\`.*?\`\`",' ')
    #remove numbers
    com_clean=com_clean.str.replace("[0-9]+",' ')
    #remove numbers
    com_clean=com_clean.str.replace("_",' ')
    #remove symbols.    There must be a more comprehensive way of doing this?
    com_clean=com_clean.str.replace("[\[\[\{\}=_:\|\(\)\\\/\`]+",' ')
    #remove multiple spaces, replace with a single space
    com_clean=com_clean.str.replace('\\s+',' ')
    return com_clean
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
