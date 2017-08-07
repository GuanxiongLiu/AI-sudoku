from keras.models import Model
from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers.convolutional import Conv2D



class Neural_Model:
    
    def __init__(self, config):
        self.input_dim = config['input dim']
        self.dense_dim_1 = config['dense layer 1 dim']
        self.dense_dim_2 = config['dense layer 2 dim']
        self.output_dim = config['output dim']
        self.epochs = config['epochs']
        self.loss = config['loss']
        
    def build_model(self):
        # input
        inputs = Input(shape=(self.input_dim,))
        # convolusional
        reshape = Reshape(target_shape=(9,9,1))(inputs)
        conv = Conv2D(filters=128, kernel_size=(3,3))(reshape)
        flat = Flatten()(conv)
        # hidden layers
        hidden1 = Dense(self.dense_dim_1, activation='relu')(flat)
        hidden2 = Dense(self.dense_dim_1, activation='relu')(hidden1)
        # output
        pred = Dense(self.output_dim, activation='relu')(hidden2)
        # build model
        self.model = Model(inputs=inputs, outputs=pred)
        
    def train_model(self, data, labels):
        self.model.compile(optimizer='rmsprop', 
                           loss=self.loss, 
                           metrics=['accuracy'])
        self.model.fit(data, labels, epochs=self.epochs, batch_size=100)  # starts training
        
    def test_model(self, test_X, test_Y):
        result = self.model.evaluate(test_X, test_Y, batch_size=100, verbose=1, sample_weight=None)
        print(result)


