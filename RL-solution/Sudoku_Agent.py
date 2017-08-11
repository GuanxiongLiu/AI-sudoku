# general packages
import random
import numpy as np
from collections import deque
import json

# keras packages
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD , Adam










class RL_sudoku:
    def __init__(self):
        # pre-defined parameters
        self.DATA_PATH = '../transfer_data.npy' # the file path to local data
        self.SPLIT_RATIO = 0.7
        self.LEARNING_RATE = 1e-4 # optimizer learning rate
        self.OBSERVATION = 3200. # steps to observe before training
        self.INITIAL_EPSILON = 0.5 # starting value of epsilon
        self.FINAL_EPSILON = 0.0001 # final value of epsilon
        self.MAX_STEP = 1e5 # maximum steps
        self.ACTIONS = 9**3 # dimension of action
        self.MEMORY_LIMIT = 50000 # maximum number of memory
        self.BATCH = 50 # training batch size
        self.TEST_BATCH = 100 # test size
        self.GAMMA = 0.9 # reward decay rate



    def loadData(self):
        data = np.load(self.DATA_PATH)
        np.random.shuffle(data)
        train_num = int(len(data)*self.SPLIT_RATIO)
        self.train_X = data[:train_num, 0, :]
        self.train_Y = data[:train_num, 1, :]
        self.test_X = data[train_num:, 0, :]
        self.test_Y = data[train_num:, 1, :]
        self.train_index = 0
        self.test_index = 0
        self.INPUT_SIZE = self.train_X.shape[1]

    def genMove(self, mode, action=None):
        if (action is None): # initial
            if (mode == 'Train'):
                self.current_X = np.copy(self.train_X[self.train_index, :])
                self.Question = np.copy(self.train_X[self.train_index, :])
                self.Answer = np.copy(self.train_Y[self.train_index, :])
                self.train_index += 1
                if self.train_index >= len(self.train_X):
                    self.train_index = 0
                return (self.current_X, 0, 0) # (state, reward, terminate)
            elif (mode == 'Test'):
                self.test_base = np.random.randint(low = 0, high = self.test_X.shape[0]-self.TEST_BATCH)
                self.current_X = np.copy(self.test_X[self.test_base + self.test_index, :])
                self.Question = np.copy(self.test_X[self.test_base + self.test_index, :])
                self.Answer = np.copy(self.test_Y[self.test_base + self.test_index, :])
                self.test_index += 1
                return (self.current_X, 0, 0)
        else: # decode action
            loc = np.argwhere(action == 1)
            value = loc / 81 + 1
            index = loc % 81
            return self.evaluateMove(index, value, mode)

    def evaluateMove(self, index, value, mode):
        # self.current_train_X[index] = value
        if (self.Question[index] != 0): # change question basic
            terminate = 1
            reward = -10
            return self.genMove(mode)
        elif (self.current_X[index] == self.Answer[index]): # modify right answer
            terminate = 1
            reward = -10
            return self.genMove(mode)
        elif (self.current_X[index] == value): # stop in filling one specific wrong answer
            terminate = 1
            reward = -10
            return self.genMove(mode)
        else:
            self.current_X[index] = value # really fill in changes
            if (np.array_equal(self.current_X, self.Answer)): # finish one problem
                terminate = 1
                reward = 10
                if (mode == 'Train'):
                    self.current_X = np.copy(self.train_X[self.train_index, :])
                    self.Question = np.copy(self.train_X[self.train_index, :])
                    self.Answer = np.copy(self.train_Y[self.train_index, :])
                    self.train_index += 1
                    if self.train_index >= len(self.train_X):
                        self.train_index = 0
                elif (mode == 'Test'):
                    self.current_X = np.copy(self.test_X[self.test_base + self.test_index, :])
                    self.Question = np.copy(self.test_X[self.test_base + self.test_index, :])
                    self.Answer = np.copy(self.test_Y[self.test_base + self.test_index, :])
                    self.test_index += 1
            elif (self.Answer[index] == value): # fill in a right answer
                terminate = 0
                reward = 1
            else: # fill in a wrong answer
                terminate = 0
                reward = -1
            return (self.current_X, reward, terminate)

    def buildModel(self):
        print("Building Model")
        self.model = Sequential()
        # first dense layer
        self.model.add(Dense(128, input_shape=(self.train_X.shape[1],)))
        self.model.add(Activation('relu'))
        # second dense layer
        self.model.add(Dense(128))
        self.model.add(Activation('relu'))
        # third dense layer
        self.model.add(Dense(128))
        self.model.add(Activation('relu'))
        # fourth dense layer
        self.model.add(Dense(128))
        self.model.add(Activation('relu'))
        # fifth dense layer
        self.model.add(Dense(128))
        self.model.add(Activation('relu'))
        # output layer
        self.model.add(Dense(9*9*9))
        adam = Adam(lr=self.LEARNING_RATE)
        self.model.compile(loss='mean_squared_error',optimizer=adam)
        print("Modeling Finished")

    def trainModel(self):
        # initial replay memory
        self.memory = None
        # get initial state
        state, _, _ = self.genMove(mode = 'Train')
        state = state.reshape(1, state.shape[0])
        # training process
        global_t = 0 # global training step
        epsilon = self.INITIAL_EPSILON
        while (global_t < self.MAX_STEP):
            # generate memory
            action_index = 0
            reward = 0
            action = np.zeros([self.ACTIONS])
            # epsilon greedy algorithm
            if (random.random() <= epsilon):
                action_index = random.randrange(self.ACTIONS)
                action[action_index] = 1
            else:
                q = self.model.predict(state) 
                max_Q = np.argmax(q)
                action_index = max_Q
                action[max_Q] = 1
            # execute action
            next_state, reward, terminate = self.genMove(mode = 'Train', action = action)
            next_state = next_state.reshape(1, next_state.shape[0])
            # store the transition in D
            feedback = np.array([action_index, reward, terminate])
            mem_row = np.concatenate((state.reshape(-1), next_state.reshape(-1), feedback), axis=0).reshape(1, -1)
            if (self.memory is None):
                self.memory = mem_row
            else:
                self.memory = np.concatenate((self.memory, mem_row), axis=0)
            if (self.memory.shape[0] > self.MEMORY_LIMIT):
                self.memory = self.memory[1:, :]
            # reduced the epsilon gradually
            if (epsilon > self.FINAL_EPSILON and len(self.memory) > self.OBSERVATION):
                epsilon -= (self.INITIAL_EPSILON - self.FINAL_EPSILON) / (self.MAX_STEP*0.6)
            # update state
            state = next_state
            # memory replay training when there are enough memory
            if (self.memory.shape[0] > self.OBSERVATION):
                # sample a minibatch to train on
                np.random.shuffle(self.memory)
                minibatch = self.memory[:self.BATCH, :]
                # decode
                mem_cur_state = minibatch[:, :self.INPUT_SIZE].reshape(self.BATCH, -1)
                mem_nex_state = minibatch[:, self.INPUT_SIZE:self.INPUT_SIZE*2].reshape(self.BATCH, -1)
                mem_action = minibatch[:, self.INPUT_SIZE*2:self.INPUT_SIZE*2+1]
                mem_reward = minibatch[:, self.INPUT_SIZE*2+1:self.INPUT_SIZE*2+2]
                mem_terminate = minibatch[:, self.INPUT_SIZE*2+2:self.INPUT_SIZE*2+3]
                # calculate Q for current and next state
                Q_current = self.model.predict(mem_cur_state)
                Q_next = self.model.predict(mem_nex_state)
                # update target Q based on current Q and reward
                Q_current[:, mem_action] = \
                    mem_reward + (1 - mem_terminate) * self.GAMMA * np.max(Q_next)
                # calculate loss
                loss = self.model.train_on_batch(mem_cur_state, Q_current)
                # update training steps
                global_t += 1
            # monitering the progress every 1000 steps
            if (global_t > 0 and global_t % 100 == 0):
                # print out info
                print("TIMESTEP", global_t, "/ EPSILON", epsilon, "/ Loss ", loss)
                if (global_t % 1000 == 0):
                    # save progress
                    print("Model Saved")
                    self.model.save_weights("model.h5", overwrite=True)
                    with open("model.json", "w") as outfile:
                        json.dump(self.model.to_json(), outfile)
                    # test model
                    self.testModel()
        print("Training Finished")
        
    def testModel(self):
        # load the saved weights
        self.model.load_weights("model.h5")
        print('Model Loaded')
        # initial test
        state, _, _ = self.genMove(mode = 'Test')
        state = state.reshape(1, state.shape[0])
        result = []
        while (self.test_index < self.TEST_BATCH):
            # generate action
            action = np.zeros([self.ACTIONS])
            q = self.model.predict(state)      
            max_Q = np.argmax(q)
            action[max_Q] = 1
            # execute action
            next_state, reward, terminate = self.genMove(mode = 'Test', action = action)
            next_state = next_state.reshape(1, next_state.shape[0])
            # check right or wrong when terminate
            if (terminate == 1):
                if (reward == 10):
                    result.append(1)
                elif (reward == -100):
                    result.append(0)
            # update state
            state = next_state
        print('Current model has ' + str(sum(result)/float(self.TEST_BATCH)) + '% test accuracy')
        
    def manualTest(self):
        # load saved weights
        self.model.load_weights("model.h5")
        print("Model Loaded")
        # initial question
        state, _, terminate = self.genMove(mode = 'Test')
        state = state.reshape(1, state.shape[0])
        # start to solve
        if_continue = input("Input 1 to continue: ")
        while (if_continue == 1):
            # generate action
            action = np.zeros([self.ACTIONS])
            q = self.model.predict(state)      
            max_Q = np.argmax(q)
            action[max_Q] = 1
            # execute action
            next_state, reward, terminate = self.genMove(mode = 'Test', action = action)
            next_state = next_state.reshape(1, next_state.shape[0])
            # visualize question, current action, state and right answer
            print(self.Question.reshape(9,-1))
            print(self.Answer.reshape(9,-1))
            print((max_Q/81+1, (max_Q%81)/9+1, (max_Q%81)%9+1))
            if (terminate != 1):
                print(next_state.reshape(9,-1))
            else:
                print("Solving process is terminated")
            # update state
            state = next_state
            # ask user to continue
            if_continue = input("Input 1 to continue: ")



if __name__ == "__main__":
    # initial sudoku instance
    rl_agent = RL_sudoku()
    # load data into this instance
    rl_agent.loadData()
    # build DQN model
    rl_agent.buildModel()
    # train rl agent
    rl_agent.trainModel()





