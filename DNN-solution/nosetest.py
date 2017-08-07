# data loader
from Data_Loader import Data_Loader

def test_loading():
    loader = Data_Loader('./transfer_data.npy', 0.7)
    train_X, train_Y, test_X, test_Y = loader.get_data()
    assert train_X.shape == (int(1000000*0.7), 81)
    assert train_Y.shape == (int(1000000*0.7), 81)
    assert test_X.shape == (1000000 - int(1000000*0.7), 81)
    assert test_Y.shape == (1000000 - int(1000000*0.7), 81)






