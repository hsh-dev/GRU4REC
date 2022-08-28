import pandas as pd
import numpy as np
import tensorflow as tf

class DataLoader():
    def __init__(self, config) -> None:
        self.config = config
        
        self.batch_size = config["batch_size"]
        self.split_ratio = config["split_ratio"]
        
        self.movies_path = config["movies_path"]
        self.ratings_path = config["ratings_path"]
        self.users_path = config["users_path"]
        
        self.movie_length = 0
        self.user_length = 0
        self._load_()
        self._init_length_()
        self._make_session_()        
    
    def _load_(self):
        self.movies_data = pd.read_csv(self.movies_path, delimiter = "::", header = None, engine = "python", encoding = "ISO-8859-1")
        self.ratings_data = pd.read_csv(self.ratings_path, delimiter = "::", header = None, engine = "python", encoding = "ISO-8859-1")
        self.users_data = pd.read_csv(self.users_path, delimiter = "::", header = None, engine = "python", encoding = "ISO-8859-1")

    def _init_length_(self):
        movie = np.array(self.movies_data[0].unique())
        max_id = np.max(movie)
        self.movie_length = max_id
        
        user = np.array(self.users_data[0].unique())
        max_user_id = np.max(user)
        self.user_length = max_user_id
    
    def _make_session_(self):
        train_user, valid_user = self.split_user()
        
        train_set = self.collect_movie_list(train_user)
        valid_set = self.collect_movie_list(valid_user)   
        
        self.train_x, self.train_y = self.session_parallel(train_set)   ## make session parallel minibatch
        

    def split_user(self):
        user = np.array(self.users_data[0].unique())
        
        np.random.seed(self.config["numpy_seed"])
        np.random.shuffle(user)
        
        total_len = len(user)
        
        train_user = user[:int(total_len * self.split_ratio)]
        valid_user = user[int(total_len * self.split_ratio):]
        
        return train_user, valid_user
        
    def collect_movie_list(self, user_list):
        ratings = self.ratings_data.iloc[:, 0:3]
        
        data_set = {}
        
        # min_movie_length = 99999
        # min_movie_user = 0
        for user_id in user_list:
            user_ratings = ratings[ratings[0] == user_id]   ## collect movie list and ratings for the user id
            
            user_positive_movies = np.array(user_ratings[1])
            
            # if min_movie_length > len(user_positive_movies):
            #     min_movie_length = len(user_positive_movies)
                # min_movie_user = user_id
                
            data_set[user_id] = user_positive_movies
        
        return data_set

    def make_labels(self, data_set):
        session_length = self.config["session_length"]
        labelled_dataset = {}
        
        keys = data_set.keys()
        
        for key in keys:
            movie_list = data_set[key]
            movie_length = len(movie_list)
            session_count = (movie_length - 1) // session_length
        
            labelled_dataset[key] = {}
            
            x_array = np.empty((0, session_length), dtype=int)
            y_array  = np.empty((0, session_length), dtype=int)
            
            for i in range(0, session_count):
                x = np.array(movie_list[i*session_length : (i+1)*session_length])
                y = np.array(movie_list[i*session_length+1 : (i+1)*session_length+1])
                
                x = np.reshape(x, (1, session_length))
                y = np.reshape(y, (1, session_length))                
                
                x_array = np.append(x_array, x, axis = 0)
                y_array = np.append(y_array, y, axis = 0)
                
            labelled_dataset[key]['x'] = x_array
            labelled_dataset[key]['y'] = y_array
    
        return labelled_dataset
    
    def session_parallel(self, data_set):
        batch_x = []
        batch_y = []
        
        current_user = []
        current_idx = [0] * self.batch_size
                
        keys = list(data_set.keys())
        no_key = False
        
        for i in range(0, self.batch_size):
            current_user.append(keys[i])
        next_idx = self.batch_size
        
        while not no_key:
            for i in range(0, self.batch_size):
                target_user = current_user[i]
                
                x = data_set[target_user][current_idx[i]]    
                y = data_set[target_user][current_idx[i]+1]
                
                batch_x.append(x)
                batch_y.append(y)
                
                current_idx[i] = current_idx[i] + 1
                if (len(data_set[target_user]) - 1) <= current_idx[i]:
                    current_user[i] = keys[next_idx]
                    current_idx[i] = 0
                    
                    if not no_key:
                        next_idx += 1
                    if next_idx == (len(keys)):
                        no_key = True
        
        while True:
            enable = True
            mini_batch_x = []
            mini_batch_y = []
            for i in range(0, self.batch_size):
                target_user = current_user[i]
                
                x = data_set[target_user][current_idx[i]]
                y = data_set[target_user][current_idx[i]+1]
                
                mini_batch_x.append(x)
                mini_batch_y.append(y)
                
                current_idx[i] = current_idx[i] + 1
                if (len(data_set[target_user]) - 1) <= current_idx[i]:
                    enable = False
                    break
            
            if enable:
                batch_x.extend(mini_batch_x)
                batch_y.extend(mini_batch_y)
            else:
                break
                
        return batch_x, batch_y
        
        
    def one_hot_encoding(self, id_list):
        ## 1 in ground truth index, 0 in others
        one_hot_matrix = np.empty((0, self.movie_length+1), dtype = np.int32)
        
        for id in id_list:
            one_hot_vector = np.zeros((1, self.movie_length+1), dtype=np.int32)
            one_hot_vector[0, id] = 1
            
            one_hot_matrix = np.append(one_hot_matrix, one_hot_vector, axis=0)
        
        return one_hot_matrix

    
    def get_train_set(self):
        ## Too much memory
        # one_hot_train_x = self.one_hot_encoding(self.train_x)
        # one_hot_train_y = self.one_hot_encoding(self.train_y)
        
        batch_size = self.batch_size
        
        dataset = tf.data.Dataset.from_tensor_slices((self.train_x, self.train_y))
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(1)
        
        return dataset



    '''
    APIs
    '''
    def get_movie(self, id):
        idx = self.movies_data.index[self.movies_data[0] == id]
        idx = idx.tolist()[0]
        movie = self.movies_data.iat[idx, 1]

        return movie

    def get_movie_length(self):
        return self.movie_length
    
    def get_user_length(self):
        return self.user_length