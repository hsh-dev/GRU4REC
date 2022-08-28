from DataModule.DataLoader import DataLoader
from TrainModule.TrainManager import TrainManager

from Models.RNN import RNN

from DataModule.DataLoader import DataLoader

HIDDEN_DIM = 1000     ## hidden layer dimension of embedding layer
SEQ_COUNT = 5       ## sequence block count in gru layer

config = {
    "batch_size" : 128,
    "learning_rate" : 1e-4,
    "optimizer" : "ADAM",
    
    "movies_path" : "ml-1m/movies.dat",
    "ratings_path" : "ml-1m/ratings.dat",
    "users_path": "ml-1m/users.dat",
    
    
    "numpy_seed" : 10,
    "split_ratio" : 0.8,
    "session_length" : 5
    
    
}

if __name__ == "__main__":
    
    # gru = GRU_Layer(8, 3)
    # input = tf.random.normal([10, 8], 0, 10, tf.int32)
    # print("model test")
    # output = rnn(input)
    
    dataloader = DataLoader(config)

    out_dim = dataloader.get_movie_length()    
    model = RNN(n_dim = HIDDEN_DIM, m_seq = SEQ_COUNT, out_dim = out_dim)
    
    trainmanger = TrainManager(model, dataloader, config)

    trainmanger.start()
    

    