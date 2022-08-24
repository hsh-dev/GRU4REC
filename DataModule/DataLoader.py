import pandas as pd
import numpy as np

class DataLoader():
    def __init__(self) -> None:
        self.movies_path = "ml-1m/movies.dat"
        self.ratings_path = "ml-1m/ratings.dat"
        self.users_path = "ml-1m/users.dat"
        
        self.load()
            
    def load(self):
        self.movies_data = pd.read_csv(self.movies_path, delimiter = "::", header = None, engine = "python", encoding = "ISO-8859-1")
        self.ratings_data = pd.read_csv(self.ratings_path, delimiter = "::", header = None, engine = "python", encoding = "ISO-8859-1")
        self.users_data = pd.read_csv(self.users_path, delimiter = "::", header = None, engine = "python", encoding = "ISO-8859-1")
    
    def get_session(self):
        session = self.ratings_data
        print(session)
        
        ## positive sampling
        
        ## negative sampling
        
        
        

