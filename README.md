Implementation of a text based, heads up No-Limit Hold'em poker game with Deep Q learning agent training and testing on the environment.

Consists of implementation of core game logic needed to play a game of heads up (2 player) NLHE poker, including things like player options (betting, checking, raising, etc),
street structure (pre-flop, flop, turn, river), and a hand-ranking mechanism by which two sets of 7 card holdings can be compared based on the best 5 card combination.

A text based interface is used to perform various actions like calling, folding, and raising.

Furthermore, a deep Q learning agent was implemented in PyTorch and trained then tested against a randomly playing agent (AI that simply chooses random moves and bet sizes). 
