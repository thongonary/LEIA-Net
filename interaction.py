"""
Tensorflow implementation of the Interaction networks for the identification of boosted Higgs to bb decays https://arxiv.org/abs/1909.12285 
"""

import os
import itertools
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from lbn import LBNLayer

class LEIA(models.Model):
    def __init__(self, n_constituents, n_targets, params, hidden, fr_activation=0, fo_activation=0, fc_activation=0, sum_O=True):
        super(LEIA, self).__init__()

        # initialize the LBN layer for preprocessing
        self.lbn = LBNLayer(n_particles=n_constituents, n_restframes=n_constituents, boost_mode='pairs')

        self.hidden = int(hidden)
        self.P = params
        self.N = self.lbn.lbn.n_out
        self.Nr = self.N * (self.N - 1)
        self.Dr = 0
        self.De = 8
        self.Dx = 0
        self.Do = 8
        self.n_targets = n_targets
        self.fr_activation = fr_activation
        self.fo_activation = fo_activation
        self.fc_activation = fc_activation 
        self.assign_matrices()
        self.Ra = tf.ones([self.Dr, self.Nr])
        self.fr1 = layers.Dense(self.hidden, input_shape=(2 * self.P + self.Dr,))
        self.fr2 = layers.Dense(int(self.hidden/2), input_shape=(self.hidden,))
        self.fr3 = layers.Dense(self.De, input_shape=(int(self.hidden/2),))
        
        self.fo1 = layers.Dense(self.hidden, input_shape=(self.P + self.Dx + (2 * self.De),))
        self.fo2 = layers.Dense(int(self.hidden/2), input_shape=(self.hidden,))
        self.fo3 = layers.Dense(self.Do, input_shape=(int(self.hidden/2),))
        
        self.fc1 = layers.Dense(hidden)
        self.fc2 = layers.Dense(int(hidden/2))
        self.fc3 = layers.Dense(self.n_targets)
        self.sum_O = sum_O 

    def build(self, input_shape):
        assert len(input_shape) >= 2
        self.built = True

    def assign_matrices(self):
        Rr = np.zeros([self.N, self.Nr], dtype=np.float32)
        Rs = np.zeros([self.N, self.Nr], dtype=np.float32)
        receiver_sender_list = [i for i in itertools.product(range(self.N), range(self.N)) if i[0]!=i[1]]
        for i, (r, s) in enumerate(receiver_sender_list):
            Rr[r, i]  = 1
            Rs[s, i] = 1
        self.Rr = tf.convert_to_tensor(Rr)
        self.Rs = tf.convert_to_tensor(Rs)
        del Rs, Rr

    def call(self, x):
        '''
        Expect input to have shape of (batches, N_particles, N_features)
        '''
        ###PF Candidate - PF Candidate###
        print("input_shape = {}".format(x.shape))
        x = self.lbn(x) # Already in E, px, py, pz
        print("input_shape after lbn = {}".format(x.shape))
        print("n_outs after lbn = {}".format(self.lbn.lbn.n_out))
        x = tf.transpose(x, perm=[0, 2, 1]) # to fit in the IN
        print(f"x after transpose = {x.shape}")
        Orr = self.tmul(x, self.Rr)
        print(f"Orr = {Orr.shape}")
        Ors = self.tmul(x, self.Rs)
        print(f"Ors = {Ors.shape}")
        B = tf.concat([Orr, Ors], 1)
        print(f"B = {B.shape}")
        ### First MLP ###
        B = tf.transpose(B, perm=[0, 2, 1])
        if self.fr_activation == 2:
            B = tf.nn.selu(self.fr1(tf.reshape(B, [-1, 2 * self.P + self.Dr])))
            B = tf.nn.selu(self.fr2(B))
            E = tf.nn.selu(tf.reshape(self.fr3(B), [-1, self.Nr, self.De]))
        elif self.fr_activation == 1:
            B = tf.nn.elu(self.fr1(tf.reshape(B, [-1, 2 * self.P + self.Dr])))
            B = tf.nn.elu(self.fr2(B))
            E = tf.nn.elu(tf.reshape(self.fr3(B), [-1, self.Nr, self.De]))
        else:
            B = tf.nn.relu(self.fr1(tf.reshape(B, [-1, 2 * self.P + self.Dr])))
            print(f"B after fr1 = {B.shape}")
            B = tf.nn.relu(self.fr2(B))
            print(f"B after fr2 = {B.shape}")
            E = tf.nn.relu(tf.reshape(self.fr3(B), [-1, self.Nr, self.De]))
        del B
        print("E after 1st MLP = {}".format(E.shape))
        E = tf.transpose(E, perm=[0, 2, 1])
        print("E after transpose = {}".format(E.shape))
        print("Rr after transpose = {}".format(self.Rr.shape))
        Ebar = self.tmul(E, tf.transpose(self.Rr, perm=[1, 0]))
        print("Ebar after tmul = {}".format(Ebar.shape))
        del E
       
        ####Final output matrix for particles###
        C = tf.concat([x, Ebar], 1)
        del Ebar
        C = tf.transpose(C, perm=[0, 2, 1])
        
        ### Second MLP ###
        if self.fo_activation == 2:
            C = tf.nn.selu(self.fo1(tf.reshape(C, [-1, self.P + self.Dx + self.De])))
            C = tf.nn.selu(self.fo2(C))
            O = tf.nn.selu(tf.reshape(self.fo3(C), [-1, self.N, self.Do]))
        elif self.fo_activation == 1:
            C = tf.nn.elu(self.fo1(tf.reshape(C, [-1, self.P + self.Dx + self.De])))
            C = tf.nn.elu(self.fo2(C))
            O = tf.nn.elu(tf.reshape(self.fo3(C), [-1, self.N, self.Do]))
        else:
            C = tf.nn.relu(self.fo1(tf.reshape(C, [-1, self.P + self.Dx + self.De])))
            C = tf.nn.relu(self.fo2(C))
            O = tf.nn.relu(tf.reshape(self.fo3(C), [-1, self.N, self.Do]))
        del C
       
        if self.sum_O:
            O = tf.reduce_sum(O, 1)

        ### Classification MLP ###
        if self.fc_activation == 2:
            if self.sum_O:
                N = tf.nn.selu(self.fc1(tf.reshape(O, [-1, self.Do * 1])))
            else:
                N = tf.nn.selu(self.fc1(tf.reshape(O, [-1, self.Do * N])))
            N = tf.nn.selu(self.fc2(N))   
        if self.fc_activation == 1:
            if self.sum_O:
                N = tf.nn.elu(self.fc1(tf.reshape(O, [-1, self.Do * 1])))
            else:
                N = tf.nn.elu(self.fc1(tf.reshape(O, [-1, self.Do * N])))
            N = tf.nn.elu(self.fc2(N)) 
        else:
            if self.sum_O:
                N = tf.nn.relu(self.fc1(tf.reshape(O, [-1, self.Do * 1])))
            else:
                N = tf.nn.relu(self.fc1(tf.reshape(O, [-1, self.Do * N])))
            N = tf.nn.relu(self.fc2(N))
        N = self.fc3(N)
        return N

    def tmul(self, x, y):  #Takes (I * J * K)(K * L) -> I * J * L 
        x_shape = tf.shape(x)
        y_shape = tf.shape(y)
        return tf.reshape(tf.matmul(tf.reshape(x, [-1, x_shape[2]]), y), [-1, x_shape[1], y_shape[1]]) 
