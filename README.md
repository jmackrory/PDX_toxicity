# Toxic Comment Classification

This project is based on the Toxic Comment Classification competition
for Kaggle.  Most of the current analysis was done as part of the 
Portland Data Science Group's January 2018 session, which used the original
data, which has been further processed for the Kaggle competition.

Note: Due to the nature of the comments, there is a lot of awful language being used, and tested for
including racism, misogyny, homophobia, etc.  

The key notebooks are:
- init_explore.ipynb - this has my initial thoughts, outline of my approach.
                    This tokenizes the messages, implements Naive Bayes, explores the failures.
                    It also has some work using a neural network.
- deep_network.py - implements a class to train a Tensorflow model, with a deep network trained 
 on the term-frequency matrix.  4 layers with ReLU activation, and a final sigmoidal activation.
- init_svm.py - Support Vector Machines were competitive with neural networks for a long time, but
 they are less popular these days.  This is an attempt at building a SVM that can handle a large number
 of records - in particular a bagged ensemble of small kernel-SVMs, and eventually a random Fourier model.
 
Current work:
The project is being expanded to include a Recurrent Neural Network, and word2vec or fasttext embeddings.

Another analysis that remains to be done is to look at the ratings of toxicity and the worker demographics.
There is a notable variation in how racism is scored.
