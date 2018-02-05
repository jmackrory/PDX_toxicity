import tensorflow as tf
from tensorflow.contrib.layers import fully_connected, l2_regularizer

import numpy as np
from model import Model
import matplotlib.pyplot as plt

#to prevent creating huge logs.
from IPython.display import clear_output

class deep_dropout_NN(Model):

    def __init__(self,Xshape):
        self.Nlayers=4
        self.Nhidden=200
        self.Nout=1
        self.lr = 0.01
        self.keep_prob=0.9
        self.frac_perc=0.01
        self.n_iter=5000
        self.nout=200
        self.Nobs=Xshape[0]
        self.Nfeatures=Xshape[1]
        #only grabbing a fraction of the data
        self.Nbatch=np.int(self.Nobs*self.frac_perc)
        self.build()

    def build(self):
        """Creates essential components for graph, and 
        adds variables to instance. 
        """
        tf.reset_default_graph()
        self.add_placeholders()
        self.pred = self.add_prediction_op()
        self.loss = self.add_loss_op(self.pred)
        self.train_op = self.add_training_op(self.loss)

    def add_placeholders(self):
        """Adds placeholders to graph, by adding
        as instance variables for the model.
        """
        #load in the training examples, and their labels
        self.X = tf.placeholder(tf.float32, [self.Nbatch,self.Nfeatures],name='X')
        self.y = tf.placeholder(tf.int32,[self.Nbatch,self.Nout],name='y')

    def create_feed_dict(self,inputs_batch, labels_batch=None):
        """Make a feed_dict from inputs, labels as inputs for 
        graph.
        Args:
        inputs_batch - batch of input data
        label_batch  - batch of output labels. (Can be none for prediction)
        Return:
        Feed_dict - the mapping from data to placeholders.
        """
        feed_dict={self.X:inputs_batch}
        if labels_batch is not None:
            feed_dict[self.y]=labels_batch
        return feed_dict

    def add_prediction_op(self):
        """The core model to the graph, that
        transforms the inputs into outputs.
        Implements deep neural network with relu activation.
        """
        X2 = tf.nn.l2_normalize(self.X,dim=1)
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
        outputs=fully_connected(inputs=H4,num_outputs=self.Nout,
            activation_fn=tf.sigmoid)
        
        return outputs

    def add_loss_op(self,outputs):
        """Add ops for loss to graph.
        Average loss for a given set of outputs.
        Computes log-loss.  Should upgrade to column-wise.
        """
        eps=1E-15
        logloss = tf.losses.log_loss(self.y,outputs,epsilon=eps)
        #rocloss,roc_op=tf.metrics.auc(self.y,outputs)

        return logloss

    def add_training_op(self,loss):
        """Create op for optimizing loss function.
        Can be passed to sess.run() to train the model.
        Return 
        """
        optimizer=tf.train.AdamOptimizer(learning_rate=self.lr)
        training_op=optimizer.minimize(loss)
        return training_op

    def train_on_batch(self, sess, inputs_batch, labels_batch):
        """Perform one step of gradient descent on the provided batch of data.

        Args:
            sess: tf.Session()
            input_batch:  np.ndarray of shape (Nbatch, Nfeatures)
            labels_batch: np.ndarray of shape (Nbatch, 1)
        Returns:
            loss: loss over the batch (a scalar)
        """
        feed = self.create_feed_dict(inputs_batch, labels_batch=labels_batch)
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        return loss

    def predict_on_batch(self, sess, inputs_batch):
        """Make predictions for the provided batch of data

        Args:
            sess: tf.Session()
            input_batch: np.ndarray of shape (Nbatch, Nfeatures)
        Returns:
            predictions: np.ndarray of shape (Nbatch, 1)
        """
        feed = self.create_feed_dict(inputs_batch)
        predictions = sess.run(self.pred, feed_dict=feed)
        return predictions

        
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

    def run_graph(self,Xi,yi,save_name):
        """run_graph

        Runs the deep NN on the reduced term-frequency matrix.

        """
        init=tf.global_variables_initializer()

        #save model and graph
        saver=tf.train.Saver()

        loss_tot=np.zeros(int(self.n_iter/self.nout+1))
        plt.figure()
        with tf.Session() as sess:
             init.run()
             for iteration in range(self.n_iter+1):
                 #select random starting point.
                 ind_batch,X_batch,y_batch=self.get_batch(
                 Xi,yi)
                 current_loss=self.train_on_batch(sess, X_batch, y_batch)
                 if (iteration)%self.nout ==0:
                     clear_output(wait=True)
                     #current_pred=self.predict_on_batch(sess,X_batch)
                     print('iter #{}. Current log-loss:{}'.format(iteration,current_loss))
                     #nn_pred=sess.run(self.outputs,feed_dict={X:X_batch})
                     #nn_pred_reduced=np.round(nn_pred).astype(bool)
                     print('\n')
                     #save the weights
                     saver.save(sess,save_name,global_step=iteration)
                     loss_tot[int(iteration/self.nout)]=current_loss
             plt.plot(loss_tot)
             plt.ylabel('Log-loss')
             plt.xlabel('Iterations x100')
             plt.show()

            
    def predict_all(self,model_name,input_data):
        """network_predict
        Load a saved Neural network, and predict the output labels
        based on input_data
    
        Input: model_name - string name to where model/variables are saved.
        input_data - transformed data of shape (Nobs,Nfeature).

        Output nn_pred_reduced - vector of predicted labels.
        """
        #tf.reset_default_graph()
        with tf.Session() as sess:
            loader=tf.train.import_meta_graph(model_name+'.meta')
            loader.restore(sess,model_name)
            Nin,Nfeat=input_data.shape
            if (Nin < self.Nbatch):
                print('Number of inputs < Number of batch expected')
                print('Padding with zeros')
                input_dat=np.append(input_dat,
                                    np.zeros((self.Nbatch-Nin,Nfeat)))
            i0=0
            i1=self.Nbatch

            nn_pred_total=np.zeros((Nin,1))
            while (i1 < Nin):
                X_batch=input_data[i0:i1]
                nn_pred=self.predict_on_batch(sess,X_batch)
                nn_pred_total[i0:i1]=nn_pred
                i0=i1
                i1+=self.Nbatch
            #last iter: do remaining operations.  
            X_batch=input_data[-self.Nbatch:]
            nn_pred=self.predict_on_batch(sess,X_batch)
            nn_pred_total[-self.Nbatch:]=nn_pred
            nn_pred_reduced=np.round(nn_pred_total).astype(bool)
        return nn_pred_reduced
