import numpy as np


class Data_Loader:
    
    def __init__(self, fpath, ratio):
        # loading whole data
        data = np.load(fpath)
        # seperate training and testing
        np.random.shuffle(data)
        count,_,_ = data.shape
        self.train_data = data[:int(count*ratio), 0, :]
        self.train_label = data[:int(count*ratio), 1, :]
        self.test_data = data[int(count*ratio):, 0, :]
        self.test_label = data[int(count*ratio):, 1, :]
        
    def get_data(self):
        return self.train_data, self.train_label, self.test_data, self.test_label


