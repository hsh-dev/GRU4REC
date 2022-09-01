from re import S
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Embedding, Softmax, Dropout, GRU
from Models.GRU_Layer import GRU_Layer
import tensorflow as tf


class RNN(Model):
    def __init__(self, h_dim, seq_dim, out_dim, item_dim):
        super().__init__()
        '''
        H : h_dim : hidden layer dimension = embedding dimension
        I : item_dim : total number of items
        S : seq_dim : sequence dimension of input
        O : out_dim : output dimension of feedforward layer
        '''

        self.h_dim = h_dim
        self.seq_dim = seq_dim
        self.out_dim = out_dim
        self.item_dim = item_dim

        initializer = tf.keras.initializers.GlorotNormal()
        self.embedding_matrix = tf.Variable(
            initializer(shape=[self.item_dim, self.h_dim], dtype=tf.float32), 
            trainable = True)

        self.gru_layer_1 = GRU(h_dim, return_sequences = True, return_state=True)
        self.gru_layer_2 = GRU(h_dim, return_sequences = True, return_state=True)
        self.gru_layer_3 = GRU(h_dim, return_sequences = True, return_state=True)
        self.gru_layer_4 = GRU(h_dim, return_sequences = False, return_state=False)

        self.dropout = Dropout(0.3)
        self.fc_layer = Dense(self.out_dim, activation="relu")

    def call(self, x):
        ## input x : B x S

        ## after look up : B x S x M
        x = tf.nn.embedding_lookup(self.embedding_matrix, x)

        h_seq, h_out = self.gru_layer_1(x)
        h_seq, h_out = self.gru_layer_2(h_seq)
        h_seq, h_out = self.gru_layer_3(h_seq)
        h_out = self.gru_layer_4(h_seq)    

        ## after GRU Layers : B x H
        x = self.dropout(h_out)
        
        x = self.fc_layer(x)

        return x
