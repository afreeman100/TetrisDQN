# TetrisDQN
Code from my dissertation: Playing Tetris with Deep Reinforcement Learning

This project applied the DQN family of algorithms to the game of Tetris, learning to play from scratch with no prior domain
knowledge or hand-crafted features. Despite its seemingly simple nature, the combinatorial complexity of Tetris makes this a surprisingly challenging problem that has not yet been solved.


Agents are trained by running play.py, specifying the number of interactions to perform with the Tetris environment. Code for the 
environment is found in tetris.py and tetriminos.py. 

The main training procedure is contained in DQN.py, which uses one of the neural networks from networks.py. This gives
the option of using full-connected, duelling, or convolutional network architectures. While fully-connected architectures
are able to learn good policies on smaller versions of the game, convolutional layers are essential for efficient learning 
on the full game.

Agents are trained from an offline experience replay buffer, implemented in replay.py. This offers both the regular and
prioritised variants of experience replay. Prioritised replay demonstrates significantly more sample-efficient learning. 



The videos below show agents playing on different sized versions of the game. As the game grid gets larger the state space increases exponentially, and the time taken for training becomes the limiting factor in the agent's final performance.

## 5x5
![5x5](img/5x5.gif)


## 7z7
![7x7](img/7x7.gif)


## Full game (20x10)
![20x10](img/20x10.gif)
