# AI-sudoku

## Deep Neural Network Solusion
* Initially trained with a fully connected model which includes two hidden layers with 128 units. The test accuracy is around 10%.
* Currently, it implements a convolusional layer on top of fully connected layers. However, it's nevery trained due to the computing power.
* Based on [other people's test](https://github.com/Kyubyong/sudoku), a 10 layer pure convolusional model could achieve over 80% test accuracy. 

## Deep Q-learning Solusion
* Utilize deep Q network technique to form a reinforcement learning agent in solving sudoku problem.
* The deep Q network is tried to implemented with 2 to 5 fully connected layers which contain 128/256/512 units.
* The general convergence of deep Q-learning solution is not good and the possible issue is due to the high dimensionality of action.
