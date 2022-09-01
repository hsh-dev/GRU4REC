from DataModule.DataLoader import DataLoader
from TrainModule.TrainManager import TrainManager

from Models.RNN import RNN
from Models.RNN_SEQ import RNN

from DataModule.DataLoader import DataLoader

HIDDEN_DIM = 100     ## hidden layer dimension of embedding layer
SEQ_COUNT = 5       ## sequence block count in gru layer

config = {
    "batch_size" : 32,
    "learning_rate" : 1e-3,
    "optimizer" : "ADAM",
    
    "movies_path" : "ml-1m/movies.dat",
    "ratings_path" : "ml-1m/ratings.dat",
    "users_path": "ml-1m/users.dat",
    
    "loss" : "top_1",           ## top_1, cross_entropy
    "embedding" : True,        ## True when using embedding layer
    
    "numpy_seed" : 10,
    "split_ratio" : 0.8,
    "hidden_dim": 100,          # hidden layer dimension of embedding layer
    "sequence_length": 20        # sequence count of input 
}

if __name__ == "__main__":
    
    # gru = GRU_Layer(8, 3)
    # input = tf.random.normal([10, 8], 0, 10, tf.int32)
    # print("model test")
    # output = rnn(input)
    
    dataloader = DataLoader(config)

    item_count = dataloader.get_movie_length()    
    # model = RNN(n_dim = HIDDEN_DIM, m_seq = SEQ_COUNT, out_dim = out_dim, embedding=False)
    model = RNN(h_dim = config["hidden_dim"], 
                seq_dim = config["sequence_length"], 
                out_dim = item_count, 
                item_dim = item_count)
    
    trainmanger = TrainManager(model, dataloader, config)

    trainmanger.start()
    

    