from re import S
import tensorflow as tf
import numpy as np
import time
import tensorflow.keras.backend as K
from TrainModule.Scheduler import CosineDecayWrapper

class TrainManager():
    def __init__(self, model, dataloader, config) -> None:
        self.config = config    
        self.model = model
        self.dataloader = dataloader

        self.batch_size = config["batch_size"]
        self.loss = config["loss"]
        self.embedding = config["embedding"]
        
        
        self.optimizer_wrap = CosineDecayWrapper(
                optimizer = tf.keras.optimizers.Adam(
                    learning_rate=self.config["learning_rate"], beta_1=0.9, beta_2=0.999),
                max_lr = self.config["learning_rate"],
                min_lr=self.config["learning_rate"] * 0.01,
                max_epochs = 200,
                decay_cycles = 4,
                decay_epochs = 100
            )
    
    
    def start(self):
        total_epoch = 100
        
        for epoch in range(total_epoch):
            print("# Epoch {} #".format(epoch+1))
            self.train_loop()
            
            self.optimizer_wrap.update_lr(epoch)
                
                
    def train_loop(self):
        dataset = self.dataloader.get_train_set()
        self.movie_dim = self.dataloader.get_movie_length()  
        
        self.model.trainable = True

        total_step = len(dataset)
        
        all_loss_list = []
        loss_list = []
        all_hr_list = []
        hr_list = []
        
        start_time = time.time()
        
        for idx, sample in enumerate(dataset):
            x, y = sample
            x = x - 1   ## make IDs start from 0
            y = y - 1   ## make IDs start from 0
            
            loss, y_pred = self.propagation(x, y, self.loss, self.embedding)
            hr = self.hit_rate(y, y_pred)
            
            all_loss_list.append(loss)
            loss_list.append(loss)
            all_hr_list.append(hr)
            hr_list.append(hr)
            
            if (idx+1) % 1000 == 0:
                end_time = time.time()
                
                losses = np.average(np.array(loss_list))
                hr_avg = np.average(np.array(hr_list))
                print("STEP: {}/{} | Loss: {} | Time: {}s".format(
                                                                idx+1, 
                                                                total_step, 
                                                                losses, 
                                                                round(end_time - start_time, 5)
                                                            ))
                print("HR: {} ".format(hr_avg))
                
                loss_list.clear()
                hr_list.clear()
                start_time = time.time()
                
        total_loss = np.average(np.array(all_loss_list))
        total_hr = np.average(np.array(all_hr_list))
        print("Total Loss : {} | HR : {}".format(total_loss, total_hr))
    
    
    @tf.function
    def make_one_hot_vector(self, y, dim):
        dim = tf.cast(dim, dtype = tf.int32)        
        one_hot = tf.one_hot(y, dim)

        return one_hot


    @tf.function
    def propagation(self, x, y, loss = "top_1", embedding = True):
        # if not embedding:
        #     x = self.make_one_hot_vector(x, self.movie_dim)
            
        with tf.GradientTape() as tape:
            output = self.model(x)
            
            if loss == "top_1":
                loss = self.top_1_ranking_loss(y, output)
            else:
                y_one_hot = self.make_one_hot_vector(y, self.movie_dim)
                loss = self.cross_entropy(y_one_hot, output)
                
        gradients = tape.gradient(loss, self.model.trainable_variables)
        
        self.optimizer_wrap.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables))
        
        return loss, output
    
    '''
    Loss Functions
    '''
    @tf.function
    def cross_entropy(self, y_true, y_pred):
        cce = tf.keras.losses.CategoricalCrossentropy()
        err = cce(y_true, y_pred)
        
        return err        
    
    @tf.function
    def top_1_ranking_loss(self, y_true_idx, y_pred):        
        negative_list = tf.gather(y_pred, indices = y_true_idx, axis = 1)

        y_true_idx = tf.expand_dims(y_true_idx, axis = 1)
        positive_list = tf.gather(y_pred, indices = y_true_idx, axis = 1, batch_dims=1)
        
        cal = K.sigmoid(negative_list - positive_list)
        # reg = K.square(negative_list)
        # reg_pos = K.square(positive_list)    
    
        loss = K.mean(cal)
        
        return loss
    
    
    def hit_rate(self, y_true_idx, y_pred):
        y_pred = y_pred.numpy()
        length = len(y_pred)
        k = 5
        
        hit = 0
        for i in range(length):
            indices = (-y_pred[i]).argsort()[:k]
            if y_true_idx[i] in indices:
                hit += 1
                
        hit_rate = hit / length
        
        return hit_rate
    
    