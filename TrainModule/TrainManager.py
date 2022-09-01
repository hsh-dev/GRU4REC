import tensorflow as tf
import numpy as np
import time

from TrainModule.Scheduler import CosineDecayWrapper
from TrainModule.LossManager import LossManager

class TrainManager():
    def __init__(self, model, dataloader, config) -> None:
        self.config = config    
        self.model = model
        self.dataloader = dataloader

        self.batch_size = config["batch_size"]
        self.loss = config["loss"]
        self.embedding = config["embedding"]
        
        self.loss_manager = LossManager()
        
        self.optimizer_wrap = CosineDecayWrapper(
                optimizer = tf.keras.optimizers.Adam(
                    learning_rate=self.config["learning_rate"], beta_1=0.9, beta_2=0.999),
                max_lr = self.config["learning_rate"],
                min_lr = self.config["learning_rate"] * 0.01,
                max_epochs = self.config["max_epoch"],
                decay_cycles = 4,
                decay_epochs = 100
            )

        self.log = {}
    
    
    def start(self):
        total_epoch = self.config["max_epoch"]

        min_valid_loss = 9999
        save_valid_hr = 0
        
        not_update_count = 0
        
        for epoch in range(total_epoch):
            print("\n# Epoch {} #".format(epoch+1))
            print("## Train Start ##")
            self.train_loop("train")
            print("Train Loss : {} | HR@5 : {} \n".format(self.log["train_loss"], self.log["train_hr"]))

            print("## Validation Start ##")
            self.train_loop("valid")
            print("Valid Loss : {} | HR@5 : {}".format(self.log["valid_loss"], self.log["valid_hr"]))
            
            self.optimizer_wrap.update_lr(epoch)
            
            if self.log["valid_loss"] < min_valid_loss:
                not_update_count = 0
                save_valid_hr = self.log["valid_hr"]
            else:
                not_update_count += 1
            
            print("Best Validation Hit Rate : {}".format(save_valid_hr))
             
            if not_update_count >= 20:
                print("No update on valid loss. Early stop...")
                break
        
                
                
    def train_loop(self, phase):
        if phase == "train":
            dataset = self.dataloader.get_dataset("train")
            self.model.trainable = True
        else:
            dataset = self.dataloader.get_dataset("valid")
            
        self.movie_dim = self.dataloader.get_movie_length()  

        total_step = len(dataset)
        print_step = total_step // 4
        
        all_loss_list = []
        loss_list = []
        all_hr_list = []
        hr_list = []
        
        start_time = time.time()
        
        for idx, sample in enumerate(dataset):
            x, y = sample
            x = x - 1   ## make IDs start from 0
            y = y - 1   ## make IDs start from 0
            
            loss, y_pred = self.propagation(x, y, self.loss, self.embedding, phase)
            hr = self.hit_rate(y, y_pred, 5)
            
            all_loss_list.append(loss)
            loss_list.append(loss)
            all_hr_list.append(hr)
            hr_list.append(hr)
            
            if (idx+1) % print_step == 0:
                end_time = time.time()
                
                losses = np.average(np.array(loss_list))
                hr_avg = np.average(np.array(hr_list))
                print("STEP: {}/{} | Loss: {} | Time: {}s".format(
                                                                idx+1, 
                                                                total_step, 
                                                                round(losses, 7), 
                                                                round(end_time - start_time, 5)
                                                                ))
                print("HR: {} ".format(round(hr_avg, 7)))
                
                loss_list.clear()
                hr_list.clear()
                start_time = time.time()
                
        total_loss = np.average(np.array(all_loss_list))
        total_hr = np.average(np.array(all_hr_list))
        
        self.save_logs(total_loss, total_hr, phase)
        
    
    @tf.function
    def make_one_hot_vector(self, y, dim):
        dim = tf.cast(dim, dtype = tf.int32)        
        one_hot = tf.one_hot(y, dim)
        return one_hot


    @tf.function
    def propagation(self, x, y, loss, embedding, phase):
        if not embedding:
            x = self.make_one_hot_vector(x, self.movie_dim)

        with tf.GradientTape() as tape:
            output = self.model(x)
            
            if loss == "top_1":
                loss = self.loss_manager.top_1_ranking_loss(y, output)
            else:
                y = tf.reshape(y, [-1])
                y_one_hot = self.make_one_hot_vector(y, self.movie_dim)
                loss = self.loss_manager.cross_entropy_loss(y_one_hot, output)
                
        gradients = tape.gradient(loss, self.model.trainable_variables)
        
        if phase == "train":
            self.optimizer_wrap.optimizer.apply_gradients(
                zip(gradients, self.model.trainable_variables))
        
        return loss, output

    
    '''
    Caculating Accuracy
    '''
    def hit_rate(self, y_true_idx, y_pred, k):
        '''
        Recording hit if target is in top-k items
        '''
        y_pred = y_pred.numpy()
        length = len(y_pred)
        
        hit = 0
        for i in range(length):
            indices = (-y_pred[i]).argsort()[:k]
            if y_true_idx[i] in indices:
                hit += 1
                
        hit_rate = hit / length
        
        return hit_rate
    
    '''
    Save Functions
    '''
    def save_logs(self, loss, hr, phase):
        loss_key = phase + "_loss"
        hr_key = phase + "_hr"
        
        self.log[loss_key] = loss
        self.log[hr_key] = hr
    