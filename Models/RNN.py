from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Embedding, Softmax, Dropout
from Models.GRU_Layer import GRU_Layer

class RNN(Model):
    def __init__(self, n_dim, m_seq, out_dim, embedding = True):
        super().__init__()
        '''
        n_dim : hidden layer dimension = embedding dimension
        seq_dim : sequence count of GRU block
        out_dim : output dimension of feedforward layer
        '''
        
        self.n_dim = n_dim
        self.seq_dim = m_seq
        self.out_dim = out_dim
        
        self.embedding = embedding
        
        self.embedding_layer = Embedding(out_dim, n_dim)
        self.fc_input_layer = Dense(n_dim)
        
        self.gru_layer_1 = GRU_Layer(n_dim, m_seq)
        self.gru_layer_2 = GRU_Layer(n_dim, m_seq)
        self.gru_layer_3 = GRU_Layer(n_dim, m_seq)
        self.gru_layer_4 = GRU_Layer(n_dim, m_seq)
        
        self.dropout = Dropout(0.3)
        self.fc_layer = Dense(self.out_dim)
        
        self.softmax = Softmax()
        
    def call(self, x):
        if self.embedding:
            x = self.embedding_layer(x)
        else:
            x = self.fc_input_layer(x)
        shortcut = x
        
        x = self.gru_layer_1(x)
        x = self.gru_layer_2(x + shortcut)
        x = self.gru_layer_3(x)
        x = self.gru_layer_4(x + shortcut)
        
        x = self.dropout(x)
        x = self.fc_layer(x)
                
        x = self.softmax(x)
        
        return x