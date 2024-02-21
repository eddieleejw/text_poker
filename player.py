from card import Card, CardCollection, Deck, Hand, Board
import random
import math
from HandEvaluator import find_best_hand
from termcolor import colored


import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import deepqutil as dqutil


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class Player:

    def __init__(self, playerID = -1, stack = 100, position = "", sweat = True, playerName = "DEFAULT PLAYER", preflopRange = None):
        '''
        Arguments:
            playerID (int): Unique ID to denote the player
            stack (int): the starting stack amount of the player
            position (str): the position of the player ("BB" or "button")
            sweat (bool): whether we want to see this player's hole cards or not
            playername (str): name of the player
            prefloprange (cardRange): the range of possible two card hand combiations that the player can start with
        '''
        self.playerID = playerID
        self.stack = stack
        self.hand = Hand()
        self.position = position
        self.playerName = playerName
        self.preflopRange = preflopRange
        
        # this determine whether we can see their hole cards during the hand
        self.sweat = sweat
    
    def take_action(self, game) -> str:
        '''
        1: call
        2: raise
        3: fold
        '''
        return input(f"1. Call {game.prev_raise} (total so far: {game.total_bet})\n2. Raise\n3. Fold\n")

    def determine_sizing(self, game) -> str:
        return input(f"Raise Size")
    
    def action_to_string(self, action: int) -> str:
        if action == 1:
            return "call"
        elif action == 2:
            return "raise"
        elif action == 3:
            return "fold"
        else:
            return colored("ERROR: INVALID ACTION", "red")
        
    def draw_from_preflop_range(self):
        '''
        This method assume that player has a valid and nonempty preflop range
        '''
        assert(self.preflopRange is not None)
        assert(self.preflopRange.length() > 0)

        return self.preflopRange.seek_random()
    
    def train(self, game, reward = None):
        return






class RandomAgent(Player):
    '''
    This agent makes moves and chooses bet sizes at random
    '''

    def __init__(self, playerID = -1, stack = 100, position = "", sweat = True, playerName = "BOT"):
        super().__init__(playerID, stack, position, sweat, playerName)


    def take_action(self, game) -> str:
        '''
        The agent returns an action to take
        1: call
        2: raise
        3: fold
        '''
        ret = random.randint(1,3)
        action = self.action_to_string(ret)
        if game.print:
            print(f"{self.playerName} {action}s")
        return str(ret)

    def determine_sizing(self, game) -> str:
        '''
        The agent decies how much the sizing of a bet/raise should be
        '''
        ret = random.randint(1, math.ceil(self.stack))
        if game.print:
            print(f"{self.playerName} raises to {ret}")
        return str(ret)


class ValueAgent(Player):
    '''
    This agent bets with made hands, and bets larger with larger hands

    They never fold with made hands, and always fold with HC hands.

    It always calls preflop
    '''

    def __init__(self, playerID = -1, stack = 100, position = "", sweat = True, playerName = "BOT"):
        super().__init__(playerID, stack, position, sweat, playerName)
    
    def take_action(self, game) -> str:
        '''
        if preflop, always calls only

        postflop does the following:
        1) if checked to, value bets with pairs +, checks otherwise
        2) if bet to, raises with trips +, calls with pair +, folds with HC
        3) if all in, then calls with two pair +

        1: call
        2: raise
        3: fold
        '''


        if len(game.board.collection) == 0: # pre flop
            ret = 1
        else: # post flop
            # determine hand strength
            best_hand = find_best_hand(self.hand.collection + game.board.collection)
            hand_rank = best_hand[0]
            if game.state in [1,3]: # we are first to act or checked to
                # bet if we have pair or better
                if hand_rank <= 8:
                    ret = 2
                else:
                    ret = 1
            elif game.state == 2: # we are raised to

                # check for all in
                if game.all_in == True:
                    # call with two pair +
                    if hand_rank <= 7:
                        ret = 1
                    else:
                        ret = 3
                # not all in
                # raise with trips +
                elif hand_rank <= 6:
                    ret = 2
                # call with pair +
                elif hand_rank <= 8:
                    ret = 1
                # fold otherwise
                else:
                    ret = 3
            else:
                print(f"ERROR. AGENT ASKED TO TAKE ACTION IN INVALID STATE ({game.state})")
                exit()
        if game.print:
            print(f"{self.playerName} {self.action_to_string(ret)}s")

        return str(ret)

                


    def determine_sizing(self, game) -> str:
        '''
        Bets with a sizing that scales linearly to the strength of the hand

        pair = 30% pot
        two pair = 40% pot
        ...
        straight flush = 100% pot
        '''
        best_hand = find_best_hand(self.hand.collection + game.board.collection)
        hand_rank = best_hand[0]

        scale = 0.3 + (8 - hand_rank)*0.1
        ret = int(scale * game.pot)
        
        if game.print:
            print(f"{self.playerName} raises to {ret}")

        return str(ret)


class DQNAgentTrainer(Player):
    '''
    This agent uses deep Q-learning to improve its play overtime
    '''

    def __init__(self, playerID = -1, stack = 100, position = "", sweat = True, playerName = "BOT"):
        super().__init__(playerID, stack, position, sweat, playerName)


        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # set policy network hyperparameters
        self.BATCH_SIZE = 128
        self.GAMMA = 0.99
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 1000
        self.TAU = 0.005
        self.LR = 1e-4


        # Get number of actions from gym action space
        self.n_actions = 5
        # Get the number of state observations
        self.n_observations = 113
        self.policy_net = dqutil.DQN(self.n_observations, self.n_actions).to(self.device)
        self.target_net = dqutil.DQN(self.n_observations, self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LR, amsgrad=True)
        self.memory = dqutil.ReplayMemory(10000)

        self.steps_done = 0

        self.raise_size_indicator = -1

        self.state = None
        self.action = None
        self.next_state = None
        self.reward = None

        self.prev_stack = 0




    def determine_state(self, game):
        '''
        given a game, determines the state of the game, and returns it as as a numpy array of dimension 1
        '''

        state = np.zeros(113)

        # 1. check hole cards
        assert(len(self.hand.collection)==2)

        valid_values = ["2","3","4","5","6","7","8","9","T","J","Q","K","A"]
        valid_suits = ["s","c","d","h"]

        for card in self.hand.collection:
            # set node_idx to be some value from 0 to 12
            node_idx = valid_values.index(card.value)

            # this will add either 0, 13, 26, or 39 to the node index depending on the suit
            node_idx += 13 * valid_suits.index(card.suit)

            state[node_idx] = 1
            
        
        # 2. check community cards
            
        assert(len(game.board.collection) > 0)
        
        for card in game.board.collection:
            # set node_idx to be some value from 0 to 12
            node_idx = valid_values.index(card.value)

            # this will add either 0, 13, 26, or 39 to the node index depending on the suit
            node_idx += 13 * valid_suits.index(card.suit)

            state[node_idx + 52] = 1
        

        # 3. find hero stack ratio
        hero_stack = self.stack
        villain_stack = game.get_villain(self).stack
        pot = game.pot

        assert(hero_stack > 0)
        assert(villain_stack > 0)
        assert(pot > 0)

        hero_stack_ratio = round(hero_stack/(hero_stack + villain_stack + pot), 2)

        state[104] = hero_stack_ratio

        # 4. find villain stack ratio

        villain_stack_ratio = round(villain_stack/(hero_stack + villain_stack + pot), 2)
        
        state[105] = villain_stack_ratio

        # 5. previous action
        state[106:112] = game.prev_action_training


        # 6. set in position or not
        
        if game.button is self:
            state[112] = 1
    



        return state

        
        


    
    def select_action(self, state, game) -> int:
        '''
        given a state, uses the policy network to get the next action
        Does so with an epsilon greedy policy

        '''
        sample = random.random()

        # calculate the current epsilon threshold
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1

        if sample > eps_threshold:
            # use the current policy_net to choose the next action
            with torch.no_grad():
                policy_result = self.policy_net(state)
                ret = policy_result.max(1).indices.view(1, 1)

                if game.all_in == True and ret.item() > 1:
                    # game is all in, we are not allowed to raise. we should pick the highest of call or fold
                    if policy_result[0][0].item() > policy_result[0][1].item():
                        ret = torch.tensor(0, device = self.device).view(1,1)
                    else:
                        ret = torch.tensor(1, device = self.device).view(1,1)

        else:
            # choose random action
            ret = torch.tensor([[random.randint(0,4)]], device=self.device, dtype=torch.long)

            while game.all_in == True and ret.item() > 1:
                ret = torch.tensor([[random.randint(0,4)]], device=self.device, dtype=torch.long)

        
        # ret should be of shape [1,1]
        assert(ret.dim() == 2)
        assert(ret.shape[0] == 1)
        assert(ret.shape[1] == 1)

        return ret 


    
    def take_action(self, game) -> str:
        '''
        Chooses the next action using its policy network

        In particular the output of the policy network corresponds to
        0: fold
        1: call
        2: raise 33%
        3: raise 75%
        4: raise 125%

        The outputs correspond to:
        "1": call
        "2": raise
        "3": fold
        '''

        self.action = self.select_action(self.state, game)

        assert(self.raise_size_indicator == -1)

        if self.action == 0:
            ret = 3
        elif self.action == 1:
            ret = 1
        else:
            ret = 2

            if self.action == 2:
                self.raise_size_indicator = 0.33
            elif self.action == 3:
                self.raise_size_indicator = 0.75
            else:
                self.raise_size_indicator = 1.25


        return str(ret)

                


    def determine_sizing(self, game) -> str:
        '''
        Uses the policy network to choose a bet/raise sizing

        The options are:
        2: raise 33%
        3: raise 75%
        4: raise 125%
        '''
        assert(self.raise_size_indicator != -1)

        ret = int(game.pot * self.raise_size_indicator)

        self.raise_size_indicator = -1

        return str(ret)
    

    def train(self, game, reward = None):
        '''
        This tells the Q network to do its training step.

        It is called by the game after the DQN agent's actions have been taken,
        the game has played out, and it is the agent's turn to act again
        (e.g. could be next street, or a new hand entirely)

        The agent should retrieve the state of the game and the reward (unless reward is given by the game, which may be simpler to implement)

        Determine if the game is terminated or not

        push (state, action, next_state, reward) to "memory" (this requires keeping track of the prev state and action, which may be difficult?)

        setting current self.state = next_state

        calling self.optimize_model()

        Updating the target net

        This function should be called RIGHT BEFORE the agent makes a move
        '''

        

        cur_state = self.determine_state(game)
        reward = self.stack - self.prev_stack
        self.prev_stack = self.stack

        if self.state is None:
            self.state = torch.tensor(cur_state, dtype=torch.float32, device=self.device).unsqueeze(0)
            return
        
        
        reward = torch.tensor([reward], device=self.device)

        self.next_state = torch.tensor(cur_state, dtype=torch.float32, device=self.device).unsqueeze(0)



        # Store the transition in memory
        self.memory.push(self.state, self.action, self.next_state, reward)

        self.state = self.next_state

        self.optimize_model()

        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()

        # iterates through each set of parameters (e.g. each layers weights and biases) in the policy network
        # updates the target network with the update rule denoted by θ′ ← τ θ + (1 −τ )θ′
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.TAU + target_net_state_dict[key]*(1-self.TAU)
        
        # load this soft update
        self.target_net.load_state_dict(target_net_state_dict)


    
    def optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            return

        transitions = self.memory.sample(self.BATCH_SIZE)
        assert(type(transitions[0]) is dqutil.Transition)


        batch = Transition(*zip(*transitions))


        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=self.device, dtype=torch.bool)
        assert(non_final_mask.dim() == 1)
        

        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        assert(non_final_next_states.dim() == 2)
        assert(non_final_next_states.shape[1] == self.n_observations)



        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        assert(state_batch.dim() == 2 and state_batch.shape[1] == self.n_observations)
        assert(action_batch.dim() == 2 and action_batch.shape[1] == 1)
        assert(reward_batch.dim() == 1)


        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        assert(state_action_values.dim() == 2)
        assert(state_action_values.shape[1] == 1)


        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values

        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        criterion = nn.SmoothL1Loss()

        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()




class DQNAgentTester(Player):

    def __init__(self, playerID = -1, stack = 100, position = "", sweat = True, playerName = "BOT"):
        super().__init__(playerID, stack, position, sweat, playerName)


        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Get number of actions from gym action space
        self.n_actions = 5
        # Get the number of state observations
        self.n_observations = 113

        self.policy_net = dqutil.DQN(self.n_observations, self.n_actions).to(self.device)


        self.steps_done = 0

        self.raise_size_indicator = -1

        self.state = None
        self.action = None
        self.next_state = None
        self.reward = None

        self.prev_stack = 0

        self.calls_made = 0
        self.raises_made = 0
        self.folds_made = 0




    def determine_state(self, game):
        '''
        given a game, determines the state of the game, and returns it as as a numpy array of dimension 1


        As a reminder, these are the state values we have to get

        1. 52 input nodes, one for each hole card (0 or 1)

        2. 52 input nodes, one for each community card (0 or 1)

        3. hero stack ratio: (hero stack)/(hero stack + villain stack + pot)

        4. villain stack ratio: (villain stack)/(hero stack + villain stack + pot)

        5. 2 nodes for each street (so 6 total)
            1 node belonding to each player on each street ,denoting the number of times they bet/raised

        6. one node denoting if player is in position or not

        The total number of elements in the state is 52 + 52 + 1 + 1 + 6 + 1 = 113
        '''

        state = np.zeros(113)

        # 1. check hole cards
        assert(len(self.hand.collection)==2)

        valid_values = ["2","3","4","5","6","7","8","9","T","J","Q","K","A"]
        valid_suits = ["s","c","d","h"]

        for card in self.hand.collection:
            # set node_idx to be some value from 0 to 12
            node_idx = valid_values.index(card.value)

            # this will add either 0, 13, 26, or 39 to the node index depending on the suit
            node_idx += 13 * valid_suits.index(card.suit)

            state[node_idx] = 1
            
        
        # 2. check community cards
            
        assert(len(game.board.collection) > 0)
        
        for card in game.board.collection:
            # set node_idx to be some value from 0 to 12
            node_idx = valid_values.index(card.value)

            # this will add either 0, 13, 26, or 39 to the node index depending on the suit
            node_idx += 13 * valid_suits.index(card.suit)

            state[node_idx + 52] = 1
        

        # 3. find hero stack ratio
        hero_stack = self.stack
        villain_stack = game.get_villain(self).stack
        pot = game.pot

        assert(hero_stack >= 0)
        assert(villain_stack >= 0)
        assert(pot >= 0)

        hero_stack_ratio = round(hero_stack/(hero_stack + villain_stack + pot), 2)

        state[104] = hero_stack_ratio

        # 4. find villain stack ratio

        villain_stack_ratio = round(villain_stack/(hero_stack + villain_stack + pot), 2)
        
        state[105] = villain_stack_ratio

        # 5. previous action
        state[106:112] = game.prev_action_training


        # 6. set in position or not
        
        if game.button is self:
            state[112] = 1
    



        return state

        
        


    
    def select_action(self, state, game) -> int:
        '''
        given a state, uses the policy network to get the next action
        Does so with an epsilon greedy policy

        '''

        # calculate the current epsilon threshold
        self.steps_done += 1

        # use the current policy_net to choose the next action
        with torch.no_grad():
            policy_result = self.policy_net(state)
            ret = policy_result.max(1).indices.view(1, 1)

            if game.all_in == True and ret.item() > 1:
                # game is all in, we are not allowed to raise. we should pick the highest of call or fold
                if policy_result[0][0].item() > policy_result[0][1].item():
                    ret = torch.tensor(0, device = self.device).view(1,1)
                else:
                    ret = torch.tensor(1, device = self.device).view(1,1)

        if ret == 0:
            pass
        
        # ret should be of shape [1,1]
        assert(ret.dim() == 2)
        assert(ret.shape[0] == 1)
        assert(ret.shape[1] == 1)

        return ret 


    
    def take_action(self, game) -> str:
        '''
        Chooses the next action using its policy network

        In particular the output of the policy network corresponds to
        0: fold
        1: call
        2: raise 33%
        3: raise 75%
        4: raise 125%

        The outputs correspond to:
        "1": call
        "2": raise
        "3": fold
        '''

        self.state = torch.tensor(self.determine_state(game), dtype=torch.float32, device=self.device).unsqueeze(0)

        self.action = self.select_action(self.state, game)

        assert(self.raise_size_indicator == -1)

        if self.action == 0:
            self.folds_made += 1
            ret = 3
        elif self.action == 1:
            self.calls_made += 1
            ret = 1
        else:
            self.raises_made += 1
            ret = 2

            if self.action == 2:
                self.raise_size_indicator = 0.33
            elif self.action == 3:
                self.raise_size_indicator = 0.75
            else:
                self.raise_size_indicator = 1.25


        return str(ret)

                


    def determine_sizing(self, game) -> str:
        '''
        Uses the policy network to choose a bet/raise sizing

        The options are:
        2: raise 33%
        3: raise 75%
        4: raise 125%
        '''
        assert(self.raise_size_indicator != -1)

        ret = int(game.pot * self.raise_size_indicator)

        self.raise_size_indicator = -1

        return str(ret)
    


    def load_state_dict(self, MODEL_SAVE_PATH):
        self.policy_net.load_state_dict(torch.load(f = MODEL_SAVE_PATH, map_location= self.device))