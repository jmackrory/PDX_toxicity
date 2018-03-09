import tensorflow as tf
from tensorflow.contrib.layers import fully_connected, l2_regularizer
from tensorflow.contrib.rnn import MultiRNNCell, BasicRNNCell, GRUCell, LSTMCell,\
    DropoutWrapper

from tensorflow.python.tools import inspect_checkpoint as chkp

import numpy as np
import matplotlib.pyplot as plt

from util import sent_to_matrix, load_glove

#to prevent creating huge logs.
from IPython.display import clear_output
import time

class recurrent_NN(object):
    """
    Make a multi-layer recurrent neural network for predicting toxicity.
    Train via minibatch (with balanced choices).
    
    Need to update for multi-class output.
    """
    def __init__(self,maxlen,Ndim,Noutputs,cell,wordvec):
        #number of outputs per input
        self.Noutputs=Noutputs
        #number of steps
        self.maxlen=maxlen
        #number of dim on input
        self.Nfeatures=Ndim
        self.cell_type=cell
        self.Nlayers=2
        self.Nhidden=Ndim
        self.lr = 0.001
        self.keep_prob=0.5
        self.n_iter=20
        self.nprint=10
        self.is_training=True
        #only grabbing a fraction of the data
        self.Nbatch=100
        self.wordvec=wordvec
        tf.reset_default_graph()        
        self.build()

    def build(self):
        """Creates essential components for graph, and 
        adds variables to instance. 
        """
        self.add_placeholders()
        self.pred = self.add_prediction_op()
        self.loss = self.add_loss_op(self.pred)
        self.train_op = self.add_training_op(self.loss)

    def add_placeholders(self):
        """Adds placeholders to graph, by adding
        as instance variables for the model.
        """
        #load in the training examples, and their labels
        self.X = tf.placeholder(tf.float32, [self.Nbatch,self.maxlen,self.Nfeatures],name='X')
        self.y = tf.placeholder(tf.int32,[self.Nbatch,self.Noutputs],name='y')

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

    def make_RNN_cell(self,Nneurons,fn=tf.nn.relu):
        """
        Returns a new cell (for deep recurrent networks), with Nneurons,
        and activation function fn.
        """
        #Make cell type
        if self.cell_type=='basic':
            cell=BasicRNNCell(num_units=Nneurons,activation=fn)
        elif self.cell_type=='LSTM':
            cell=LSTMCell(num_units=Nneurons,activation=fn)
        elif self.cell_type=='GRU':
            cell=GRUCell(num_units=Nneurons,activation=fn)
        #include dropout when training
        if self.is_training:
            cell=DropoutWrapper(cell,input_keep_prob=self.keep_prob,
                                variational_recurrent=True,
                                input_size=Nneurons,
                                dtype=tf.float32)
        return cell
    
    def add_prediction_op(self):
        """The core model to the graph, that
        transforms the inputs into outputs.
        Implements deep neural network with relu activation.
        """
        cell_list=[]
        for i in range(self.Nlayers):
            cell_list.append(self.make_RNN_cell(self.Nhidden,tf.nn.leaky_relu))

        multi_cell=tf.contrib.rnn.MultiRNNCell(cell_list,state_is_tuple=True)
        #Note that using [cell]*n_layers did not work.  This just made a copy pointing at the SAME cell in memory.
        rnn_outputs,states=tf.nn.dynamic_rnn(multi_cell,self.X,dtype=tf.float32)
        #use states (like CNN) since 
        #this maps the number of hidden units to fewer outputs.
        outputs = fully_connected(states,self.Noutputs,activation_fn=tf.sigmoid)
        outputs=outputs[0]
       
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
        
    def get_random_batch(self,Xi,yi):
        """get_random_batch
        Returns random subset of the data and labels.
        Note: not necessarily balanced.
        """ 
        #make vector and sample indices for true/false.
        nobs=Xi.shape[0]
        ind_sub=np.random.choice(nobs,self.Nbatch,replace=False)
        Xsub = self.get_data(ind_sub,Xi)
        y_sub = yi[ind_sub]#.reshape((len(ind_sub),1))
        return Xsub,y_sub

    def get_pred_batch(self,i0,i1,Xi):
        """get_pred_batch
        Returns subset of the data for indicies between [i0,i1)
        """ 
        #make vector and sample indices for true/false.
        ind_sub=np.arange(i0,i1)
        Xsub = self.get_data(ind_sub,Xi)
        return Xsub
    
    def get_data(self,ind,df_vec_ind):
        """get_data
        Takes indices and finds desired comment.
        Then finds wordvec embedding for word_vectors in that
        comment.
        Will pad with zeros up to Nmax.
        """
        Xi=np.zeros((self.Nbatch,self.maxlen,self.Nfeatures))
        for i in range(self.Nbatch):
            iloc=ind[i]
            vec_indices=df_vec_ind[iloc]
            Xi[i]=sent_to_matrix(vec_indices,self.wordvec,cutoff=self.maxlen)
        return Xi
    
    def train_graph(self,Xi,yi,save_name=None):
        """train_graph
        Runs the deep NN on the reduced term-frequency matrix.
        """
        self.is_training=True
        #save model and graph
        saver=tf.train.Saver()
        init=tf.global_variables_initializer()
        loss_tot=np.zeros(int(self.n_iter/self.nprint+1))
        #Try adding everything by name to a collection
        tf.add_to_collection('X',self.X)
        tf.add_to_collection('y',self.y)
        tf.add_to_collection('loss',self.loss)
        tf.add_to_collection('pred',self.pred)
        tf.add_to_collection('train',self.train_op)
        with tf.Session() as sess:
            init.run()
            saver.save(sess,save_name)
            t0=time.time()
            #Use Writer for tensorboard.
            writer=tf.summary.FileWriter("logdir-train",sess.graph)            
            for iteration in range(self.n_iter+1):
                #select random starting point.
                X_batch,y_batch=self.get_random_batch(Xi,yi)
                current_loss=self.train_on_batch(sess, X_batch, y_batch)
                t2_b=time.time()
                if (iteration)%self.nprint ==0:
                    clear_output(wait=True)
                    #current_pred=self.predict_on_batch(sess,X_batch)
                    print('iter #{}. Current log-loss:{}'.format(iteration,current_loss))
                    print('Total Time taken:{}'.format(t2_b-t0))
                    print('\n')
                    #save the weights
                    if (save_name != None):
                        saver.save(sess,save_name,global_step=iteration,
                                   write_meta_graph=False)
                    #manual logging of loss    
                    loss_tot[int(iteration/self.nprint)]=current_loss
            writer.close()
            #Manual plotting of loss.  Writer/Tensorboard supercedes this .
            plt.figure()                            
            plt.plot(loss_tot)
            plt.ylabel('Log-loss')
            plt.xlabel('Iterations x100')
            plt.show()
            
    def predict_all(self,model_name,num,input_data,reset=False):
        """network_predict
        Load a saved Neural network, and predict the output labels
        based on input_data
    
        Input: model_name - string name to where model/variables are saved.
        input_data - transformed data of shape (Nobs,Nfeature).

        Output nn_pred_reduced - vector of predicted labels.
        """
        if (reset):
            tf.reset_default_graph()        
        self.is_training=False
        with tf.Session() as sess:

            saver=tf.train.import_meta_graph(model_name+'.meta')
            #restore graph structure
            self.X=tf.get_collection('X')[0]
            self.y=tf.get_collection('y')[0]
            self.pred=tf.get_collection('pred')[0]                        
            #restores weights etc.
            saver.restore(sess,model_name+'-'+str(num))

            
            writer=tf.summary.FileWriter("logdir-pred",sess.graph)            
            writer.close()
            Nin=input_data.shape[0]
            if (Nin < self.Nbatch):
                print('Number of inputs < Number of batch expected')
                print('Padding with zeros')
                input_dat=np.append(input_dat,
                                    np.zeros((self.Nbatch-Nin,self.Noutputs)))
            i0=0
            i1=self.Nbatch

            nn_pred_total=np.zeros((Nin,self.Noutputs))
            while (i1 < Nin):
                X_batch=self.get_pred_batch(i0,i1,input_data)
                nn_pred=self.predict_on_batch(sess,X_batch)
                nn_pred_total[i0:i1]=nn_pred
                i0=i1
                i1+=self.Nbatch
            #last iter: do remaining operations.  (some redundancy here)
            X_batch=self.get_pred_batch(Nin-self.Nbatch,Nin,input_data)
            nn_pred=self.predict_on_batch(sess,X_batch)
            nn_pred_total[-self.Nbatch:]=nn_pred
            #nn_pred_reduced=np.round(nn_pred_total).astype(bool)
        return nn_pred_total

    def restore_model(self,sess,model_name):
        """Attempts to reset both TF graph, and 
        RNN stored variables/structure.
        """
        saver=tf.train.import_meta_graph(model_name+'.meta')
        #restore graph structure
        self.X=tf.get_collection('X')[0]
        self.y=tf.get_collection('y')[0]
        self.pred=tf.get_collection('pred')[0]
        self.train=tf.get_collection('train')[0]
        self.loss=tf.get_collection('loss')[0]
        #restores weights etc.
        saver.restore(sess,model_name+'-'+str(num))
        
