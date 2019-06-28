# TetrisDQN
Playing Tetris with deep Q-learning. Code from my dissertation.

Agents are trained by running play.py, specifying the number of interactions to perform with the Tetris environment. Code for the 
environment is found in tetris.py and tetriminos.py. 

The main training procedure is contained in DQN.py, which uses one of the neural networks from networks.py. This gives
the option of using full-connected, duelling, or convolutional network architectures. While fully-connected architectures
were able to learn good policied on smaller versions of the game, convolutional layers were essential for efficient learning 
on the full game.

Agents are trained from an offline experience replay buffer, implemented in replay.py. This offers both the regular and
prioritised variants of experience replay. Prioritised replay demonstrated significantly more sample-efficient learning. 
