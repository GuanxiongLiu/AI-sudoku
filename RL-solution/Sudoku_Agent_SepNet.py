# general packages
import random
import numpy as np
from collections import deque
import json

# keras packages
from keras.models import model_from_json
from keras.models import Model
from keras.layers import Input
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD , Adam










class RL_sudoku:
    def __init__(self):
        # pre-defined parameters
        self.DATA_PATH = '../transfer_data.npy' # the file path to local data
        self.SPLIT_RATIO = 0.7
        self.LEARNING_RATE = 1e-4 # optimizer learning rate
        self.OBSERVATION = 3200. # steps to observe before training
        self.INITIAL_EPSILON = 0.9 # starting value of epsilon
        self.FINAL_EPSILON = 0.0001 # final value of epsilon
        self.MAX_STEP = 1e5 # maximum steps
        self.LOCATIONS = 81 # possible locations
        self.VALUES = 9 # possible values
        self.MEMORY_LIMIT = 50000 # maximum number of memory
        self.BATCH = 50 # training batch size
        self.TEST_BATCH = 100 # test size
        self.GAMMA = 0.9 # reward decay rate
        self.ATTEMPT_LIMIT = 5 # max number of attempts for each location



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

    def genMove(self, mode, location=None, value=None):
        if (location is None and value is None): # initial
            if (mode == 'Train'):
                self.current_X = np.copy(self.train_X[self.train_index, :])
                self.Question = np.copy(self.train_X[self.train_index, :])
                self.Answer = np.copy(self.train_Y[self.train_index, :])
                self.ATTEMPT = np.zeros((self.INPUT_SIZE))
                self.train_index += 1
                if self.train_index >= len(self.train_X):
                    self.train_index = 0
                return (self.current_X, 0, 0, 0) # (state, loc_reward, val_reward, terminate)
            elif (mode == 'Test'):
                self.test_base = np.random.randint(low = 0, high = self.test_X.shape[0]-self.TEST_BATCH)
                self.current_X = np.copy(self.test_X[self.test_base + self.test_index, :])
                self.Question = np.copy(self.test_X[self.test_base + self.test_index, :])
                self.Answer = np.copy(self.test_Y[self.test_base + self.test_index, :])
                self.ATTEMPT = np.zeros((self.INPUT_SIZE))
                self.test_index += 1
                return (self.current_X, 0, 0, 0)
        else:
            return self.evaluateMove(location, value, mode)

    def evaluateMove(self, location, value, mode):
        # initial variables
        location = np.argmax(location)
        value = np.argmax(value) + 1
        loc_reward = 0
        val_reward = 0
        # evaluate location network
        if (self.Question[location] != 0 or \
            self.current_X[location] == self.Answer[location]):
        # 1. change question, 2. modify right answer
            terminate = 1
            loc_reward = -10
            self.genMove(mode) # start new game
        # evaluate value network
        else:
            # positive reward for right location
            loc_reward = 10
            # apply changes
            self.current_X[location] = value
            # wrong guess
            if (self.Answer[location] != value):
                terminate = 1
                val_reward = -10
                self.genMove(mode)
            # finish one problem
            elif (np.array_equal(self.current_X, self.Answer)):
                terminate = 1
                val_reward = 10
                self.genMove(mode)
            else:
                terminate = 0
                val_reward = 1
        return (self.current_X, loc_reward, val_reward, terminate)

    def buildModel(self):
        print("Building Model")
        # main input
        main_input = Input(shape=(81,), name='main_input')
        # We stack a deep densely-connected network on top
        x = Dense(128, activation='relu')(main_input)
        x = Dense(128, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        # we add location network output
        loc_output = Dense(81, name='loc_output')(x)
        # we add value network output
        val_output = Dense(9, name='val_output')(x)
        # finish model building
        self.model = Model(inputs=[main_input], \
                           outputs=[loc_output, val_output])
        adam = Adam(lr=self.LEARNING_RATE)
        self.model.compile(loss='mean_squared_error',optimizer=adam)
        print("Modeling Finished")

    def trainModel(self):
        # initial replay memory
        self.memory = None
        # get initial state
        state, _, _, _ = self.genMove(mode = 'Train')
        state = state.reshape(1, state.shape[0])
        # training process
        global_t = 0 # global training step
        epsilon = self.INITIAL_EPSILON
        while (global_t < self.MAX_STEP):
            # generate memory
            location = np.zeros([self.LOCATIONS])
            value = np.zeros([self.VALUES])
            # epsilon greedy algorithm
            if (random.random() <= epsilon):
                loc_index = random.randrange(self.LOCATIONS)
                location[loc_index] = 1
                val_index = random.randrange(self.VALUES)
                value[val_index] = 1
            else:
                [loc_q, val_q] = self.model.predict(state)
                loc_index = np.argmax(loc_q)
                location[loc_index] = 1
                val_index = np.argmax(val_q)
                value[val_index] = 1
            # execute action
            next_state, loc_reward, val_reward, terminate = self.genMove(mode = 'Train', \
                                                                         location = location, \
                                                                         value = value)
            next_state = next_state.reshape(1, next_state.shape[0])
            # store the transition in D
            feedback = np.array([loc_index, val_index, loc_reward, val_reward, terminate])
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
                status = minibatch[:, self.INPUT_SIZE*2:]
                mem_loc = status[:, 0]
                mem_val = status[:, 1]
                mem_loc_reward = status[:, 2]
                mem_val_reward = status[:, 3]
                mem_terminate = status[:, 4]
                # calculate Q for current and next state
                [loc_q_current, val_q_current] = self.model.predict(mem_cur_state)
                [loc_q_next, val_q_next] = self.model.predict(mem_nex_state)
                # update target Q based on current Q and reward
                loc_q_current[:, mem_loc] = \
                    mem_loc_reward + (1 - mem_terminate) * self.GAMMA * np.max(loc_q_next)
                val_q_current[:, mem_val] = \
                    mem_val_reward + (1 - mem_terminate) * self.GAMMA * np.max(val_q_next)
                # calculate loss
                loss = self.model.train_on_batch(mem_cur_state, [loc_q_current, val_q_current])
                # update training steps
                global_t += 1
            # monitering the progress every 1000 steps
            if (global_t > 0 and global_t % 100 == 0):
                # print out info
                print("TIMESTEP", global_t, "/ EPSILON", epsilon, "/ Loss ", loss)
                if (global_t % 1000 == 0):
                    # save progress
                    print("Model Saved")
                    self.model.save_weights("model_sepnet.h5", overwrite=True)
                    with open("model_sepnet.json", "w") as outfile:
                        json.dump(self.model.to_json(), outfile)
                    # test model
                    self.testModel()
        print("Training Finished")
        
    def testModel(self):
        # load the saved weights
        self.model.load_weights("model_sepnet.h5")
        print('Model Loaded')
        # initial test
        state, _, _, _ = self.genMove(mode = 'Test')
        state = state.reshape(1, state.shape[0])
        result = []
        while (self.test_index < self.TEST_BATCH):
            # generate action
            location = np.zeros([self.LOCATIONS])
            value = np.zeros([self.VALUES])
            [loc_q, val_q] = self.model.predict(state)
            loc_index = np.argmax(loc_q)
            location[loc_index] = 1
            val_index = np.argmax(val_q)
            value[val_index] = 1
            # execute action
            next_state, loc_reward, val_reward, terminate = self.genMove(mode = 'Test', \
                                                                         location = location, \
                                                                         value = value)
            next_state = next_state.reshape(1, next_state.shape[0])
            # check right or wrong when terminate
            if (terminate == 1):
                if (val_reward == 10):
                    result.append(1)
                else:
                    result.append(0)
            # update state
            state = next_state
        print('Current model has ' + str(sum(result)/float(self.TEST_BATCH)) + '% test accuracy')
        
    def manualTest(self):
        # load saved weights
        self.model.load_weights("model_sepnet.h5")
        print("Model Loaded")
        # initial question
        state, _, _, terminate = self.genMove(mode = 'Test')
        state = state.reshape(1, state.shape[0])
        # start to solve
        if_continue = input("Input 1 to continue: ")
        new_question = True
        while (if_continue == 1):
            # generate action
            location = np.zeros([self.LOCATIONS])
            value = np.zeros([self.VALUES])
            [loc_q, val_q] = self.model.predict(state)
            loc_index = np.argmax(loc_q)
            location[loc_index] = 1
            val_index = np.argmax(val_q)
            value[val_index] = 1
            # execute action
            next_state, loc_reward, val_reward, terminate = self.genMove(mode = 'Test', \
                                                                         location = location, \
                                                                         value = value)
            next_state = next_state.reshape(1, next_state.shape[0])
           # visualize question, current action, state and right answer
            if (new_question):
                new_question = False
                print(self.Question.reshape(9,-1))
                print(self.Answer.reshape(9,-1))
            print((loc_index/9 + 1, loc_index%9 + 1, val_index + 1))
            if (terminate != 1):
                print(next_state.reshape(9,-1))
            else:
                new_question = True
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





