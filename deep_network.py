import tensorflow as tf
from tensorflow.contrib.layers import fully_connected, l2_regularizer

import numpy as np
from model import Model

class deep_dropout_NN(Model):

    def __init__(self,X,y):  

        self.Nlayers=4
        self.Nhidden=100
        self.Nout=1
        self.lr = 0.01
        self.keep_prob=0.9
        self.frac_perc=0.01
        self.n_iter=1000
        self.Nobs=X.shape[0]
        self.Nfeature=X.shape[1]
        #only grabbing a fraction of the data
        self.Nsub=np.int(self.Nobs*self.frac_perc)
        self.create_train_graph()


    def add_placeholders(self):
        """Adds placeholders to graph
        """
        
        
    def create_train_graph(self):
        """
        create_train_graph

        Creates the graph for a deep NN using 
        4 layers with ReLU activation and dropout.
        """

        tf.reset_default_graph()

        #load in the training examples, and their labels
        X = tf.placeholder(tf.float32, [self.Nsub,self.Nfeature],name='X')
        y = tf.placeholder(tf.int32,[self.Nsub,self.Nout],name='y')

        X2 = tf.nn.l2_normalize(X,dim=1)

        # #make a hidden layer.  Must be smarter way to scale up. Make a list?
        #Theres a way to do it for RNN/connected layers in MultiCell?
        H1 = fully_connected(inputs=X2,num_outputs=self.Nhidden,
               activation_fn=tf.nn.relu,
               biases_initializer=tf.zeros_initializer,  
               weights_regularizer=l2_regularizer,
               biases_regularizer=l2_regularizer)
        H1_d=tf.nn.dropout(H1,self.keep_prob)

        H2 = fully_connected(inputs=H1_d,num_outputs=self.Nhidden,
            activation_fn=tf.nn.relu,
            biases_initializer=tf.zeros_initializer ,
            weights_regularizer=l2_regularizer,
            biases_regularizer=l2_regularizer)
        H2_d=tf.nn.dropout(H2,self.keep_prob)

        H3 = fully_connected(inputs=H2_d,num_outputs=self.Nhidden,
           activation_fn=tf.nn.relu,
           biases_initializer=tf.zeros_initializer ,
           weights_regularizer=l2_regularizer,
           biases_regularizer=l2_regularizer)
        H3_d=tf.nn.dropout(H3,self.keep_prob)

        H4 = fully_connected(inputs=H3_d,num_outputs=self.Nhidden,
           activation_fn=tf.nn.relu,
           biases_initializer=tf.zeros_initializer ,
           weights_regularizer=l2_regularizer,
           biases_regularizer=l2_regularizer)
        H4_d =tf.nn.dropout(H4,self.keep_prob)

        #Need to add dropout layers too.

        # #just condense the number of inputs down, acting as a linear matrix combining results
        print('NB: Only using 2 layers!')
        outputs=fully_connected(inputs=H2,num_outputs=self.Nout,
            activation_fn=tf.sigmoid)

        #should compute mean log-loss
        eps=1E-15
        logloss = tf.losses.log_loss(y,outputs,epsilon=eps)
        rocloss=tf.metrics.auc(y,outputs)
        #loss = tf.reduce_mean(tf.square(y-outputs2))
        #define optimization function.
        optimizer=tf.train.AdamOptimizer(learning_rate=self.lr)
        training_op=optimizer.minimize(logloss)

        #basically add ops to model for cross-routine access.
        self.training_op=training_op
        self.outputs=outputs

    def get_batch(self,Xi,yi):
        """get_subset
        Returns random subset of the data and labels.
        Maintains same fraction of toxic/non-toxic data as the full dataset.
        """ 
        #make vector and sample indices for true/false.
        nvec=np.arange(len(yi))
        #get the indices for true/false
        Tvec=nvec[yi.astype(bool)]
        Cvec=nvec[~yi.astype(bool)]
        #grab a random shuffling of those indices.
        np.random.shuffle(Tvec)
        np.random.shuffle(Cvec)
        #grab some fraction of them.
        it = int(len(Tvec)*self.frac_perc)
        ic = int(len(Cvec)*self.frac_perc)
        ind_sub=np.append(Tvec[:it],Cvec[:ic])
        Xsub = Xi[ind_sub]
        y_sub = yi[ind_sub].reshape((len(ind_sub),1))
        return ind_sub,Xsub,y_sub

    def run_graph(self,Xi,yi):
        """run_graph

        Runs the deep NN on the reduced term-frequency matrix.

        """
        init=tf.global_variables_initializer()

        #save model and graph
        saver=tf.train.Saver()

        with tf.Session() as sess:
             init.run()
             for iteration in range(self.n_iter):
                 #select random starting point.
                 ind_batch,X_batch,y_batch=self.get_batch(
                 Xi,yi)
                 sess.run(self.training_op, feed_dict={X: X_batch, y:y_batch})             
                 if (iteration+1)%100 ==0:
                    clear_output(wait=True)
                    current_logloss=logloss.eval( feed_dict={X:X_batch,y:y_batch})
                    current_aucloss=aucloss.eval( feed_dict={X:X_batch,y:y_batch})
                    print('iter #{}. Current log-loss:{}'.format( iteration,mse))
                    nn_pred=sess.run(self.outputs,feed_dict={X:X_batch})
                    nn_pred_reduced=np.round(nn_pred).astype(bool)
                    #check_predictions(nn_pred_reduced,y_batch)
                    print('\n')
                    #save the weights
                    saver.save(sess,'tf_models/deep_relu_drop',global_step=iteration)

