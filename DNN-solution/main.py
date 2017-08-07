from Data_Loader import Data_Loader
from Model import Neural_Model



if __name__ == '__main__':
    # loading data
    loader = Data_Loader('./transfer_data.npy', 0.7)
    train_X, train_Y, test_X, test_Y = loader.get_data()
    # build model
    config = {
        'input dim': train_X.shape[1],
        'dense layer 1 dim': 128,
        'dense layer 2 dim': 128,
        'output dim': train_Y.shape[1],
        'epochs': 5,
        'loss': 'mean_absolute_error'
    }
    model = Neural_Model(config)
    model.build_model()
    # training
    model.train_model(train_X, train_Y)
    # evaluate
    model.test_model(test_X, test_Y)