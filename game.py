from pathlib import Path
import torch
import random
from card import Card, CardCollection, Deck, Hand, Board
from player import Player, RandomAgent, ValueAgent, DQNAgentTrainer, DQNAgentTester
from HandEvaluator import find_best_hand
import cardRange

from tqdm import tqdm


class Game:


    def __init__(self, player1 = None, player2 = None):
        '''
        :type player1: Player
        :type player2: Player
        :type deck: Deck
        :type board: Board
        :type pot: int
        '''

        
        self.player1 = Player(1, 100, playerName = "Player 1") if player1 is None else player1
        self.player2 = Player(2, 100, playerName = "Player 2") if player2 is None else player2

        self.deck = Deck()
        self.deck.new_deck()
        self.board = Board()
        self.pot = 0

        self.button = self.player1
        self.player1.position = "Button"
        self.BB = self.player2
        self.player2.position = "Big Blind"

        self.prev_raise = 0 # e.g. if P1 bet 5, and P2 raise to 12, then self.prev_raise = 12-5 = 7
        self.total_bet = 0 # the cumulative bet on the given street

        self.winner = None
        self.all_in = False

        self.game_over = False
        self.hand_number = 0

        self.state = 1

        self.BB_INVESTED = 0

        # this is so that the Deep Q Network can have as input the previous actions of this and previous streets
        self.prev_action_training = [0] * 6
        self.current_street = None

        self.print = False
        
        self.ACCESS = -1

    def print_preflop(self):
        print("Stacks:")
        self.print_stack(self.player1)
        self.print_stack(self.player2)
        print()

        print("Hands:")
        if self.player1.sweat:
            self.print_hand(self.player1)
        if self.player2.sweat:
            self.print_hand(self.player2)
        print()

    def print_state(self):
        '''Prints the current state of the game'''

        print("Stacks:")
        self.print_stack(self.player1)
        self.print_stack(self.player2)
        print()

        print("Hands:")
        self.print_hand(self.player1)
        self.print_hand(self.player2)
        print()

        print("Board:")
        self.print_board()
        print()

        print("Pot size:")
        self.print_pot()
        print()
    
    def print_stack(self, player: Player):
        print(f"{player.playerName} has {player.stack}")
    
    def print_hand(self, player: Player):
        print(f"{player.playerName} ({player.position}) has {player.hand}")

    def print_board(self):
        print(f"The board is {self.board}")
    
    def print_pot(self):
        print(f"The pot size is {self.pot}")

    def new_hand(self):
        '''starts a new hand'''
        self.hand_number += 1
        if self.print:
            print(f"\n---------- HAND NUMBER {self.hand_number}----------\n")

        self.game_over = False

        # switch blinds
        self.button, self.BB = self.BB, self.button
        self.button.position = "Button"
        self.BB.position = "Big Blind"

        # BB and SB
        # self.pot = 1.5
        # self.button.stack -= .5
        # self.BB.stack -= 1
        # self.prev_raise = 0.5

        # top up any player that needs to
        if self.button.stack < 100:
            self.button.stack = 100
        if self.BB.stack < 100:
            self.BB.stack = 100

        self.pot = 0
        self.prev_raise = 0
        self.total_bet = 0

        if self.ACCESS == 1:
            # deal cards
            for _ in range(2):
                self.button.hand.append(self.deck.pop())
                self.BB.hand.append(self.deck.pop())
        else:

            h1 = self.button.draw_from_preflop_range()
            h2 = self.BB.draw_from_preflop_range()


            # ensure no overlapping of cards
            while h1.overlap(h2):
                h1 = self.button.draw_from_preflop_range()
                h2 = self.BB.draw_from_preflop_range()


            # move card from deck to the player's hand
            for card in h1.collection:
                self.button.hand.append(self.deck.collection.pop(self.deck.collection.index(card)))

            assert(self.button.hand.length() == 2)
            assert(self.deck.length() == 50)

            for card in h2.collection:
                self.BB.hand.append(self.deck.collection.pop(self.deck.collection.index(card)))

            assert(self.BB.hand.length() == 2)
            assert(self.deck.length() == 48)
            

        # # preflop
        # print("--- PREFLOP ---\n")
        # self.print_preflop()
        # self.preflop(self.button, self.BB)
        # if self.game_over:
        #     return

        self.button.stack -= self.BB_INVESTED
        self.BB.stack -= self.BB_INVESTED
        self.pot += int(2 * self.BB_INVESTED)
        self.prev_raise = 0
        self.total_bet = 0

        self.prev_action_training = [0] * 6
        self.current_street = 0

        # flop
        
        self.deal_community(3)
        if self.print:
            print("--- FLOP ---\n")
            self.print_board()
            print()
        if not self.all_in:
            self.street(self.BB, self.button)
        if self.game_over:
            return


        self.current_street = 1

        # turn
        self.deal_community(1)
        if self.print:
            print("--- TURN ---\n")
            self.print_board()
            print()
        if not self.all_in:
            self.street(self.BB, self.button)
        if self.game_over:
            return
        
        self.current_street = 2

        # river
        
        self.deal_community(1)
        if self.print:
            print("--- RIVER ---\n")
            self.print_board()
            print()
        if not self.all_in:
            self.street(self.BB, self.button)
        if self.game_over:
            return
        
        # show down
        self.showdown()
    
        self.end_hand()

    def showdown(self):
        if self.print:
            self.print_hand(self.BB)
            self.print_hand(self.button)
            print()
            self.print_board()
            print()
        bb_showdown = find_best_hand(self.BB.hand.collection + self.board.collection)
        button_showdown = find_best_hand(self.button.hand.collection + self.board.collection)

        if self.print:
            print(f"{self.BB.playerName} has " + bb_showdown[2])
            print(f"{self.button.playerName} has " + button_showdown[2])
            print()

        if bb_showdown[0] < button_showdown[0]:
            if self.print:
                print(f"{self.BB.playerName} wins!")
            self.winner = self.BB
        elif bb_showdown[0] > button_showdown[0]:
            if self.print:
                print(f"{self.button.playerName} wins!")
            self.winner = self.button
        else:
            if self.print:
                print("Tie break!")
            if bb_showdown[1] > button_showdown[1]:
                if self.print:
                    print(f"{self.BB.playerName} wins!")
                self.winner = self.BB
            elif bb_showdown[1] < button_showdown[1]:
                if self.print:
                    print(f"{self.button.playerName} wins!")
                self.winner = self.button
            else:
                if self.print:
                    print("Chop!")


    def preflop(self, button: Player, BB: Player):
        '''represents a street of action, when that street is preflop
        
        This street is unique, because both players start with some amount invested in the pot already

        I ignore the possibility that a player is forced all in due to the blinds. This is for several reasons
        1) In real play, with the exception of tournament style play where top-ups are not permitted, a player
        with that short of a stack is likely to top up their stack to some playable amount
        2) It is too complicated to implement and I plan to enforce a minimum stack depth (e.g. 20 BB)
        '''

        # players invest money
        button.stack -= 0.5
        BB.stack -= 1
        self.pot += 1.5
        self.prev_raise = 0.5
        self.total_bet = 1

        self.state = 1

        # button has opton
        action = self.option(button)
        
        first, second = BB, button
        # if button call, then BB can check it back, raise, or fold
        if action == 1:
            # self.prev_raise = 0, but this is already done in action_call
            # then we give option to BB, while in state 3 of the DFA i.e. equivalent as BB being checked to
            self.state = 3
            while self.state not in [4,5]:
                action = self.option(first)
                self.state = self.street_DFA(self.state, action)
                first, second = second, first
        # if button raise, then BB can call, raise or fold. i.e. in DFA state 2
        elif action == 2:
            self.state = 2
            while self.state not in [4,5]:
                action = self.option(first)
                self.state = self.street_DFA(self.state, action)
                first, second = second, first
        # if button fold, then end of hand i.e. DFA state 4
        elif action == 3:
            self.state = 4




        if self.state == 4:
            self.end_hand()
        elif self.state == 5:
            self.prev_raise = 0
            self.total_bet = 0
        else:
            print("Error. Street DFA is invalid")

    def street(self, first: Player, second: Player):
        '''defines a street of action, where "first" goes first and "second" goes second
        
        model a street as a deterministic finite automaton. the states are:
        (1) initial
        (2) raised to
        (3) checked to
        (4) end of hand
        (5) end of street
        '''

        self.state = 1

        while self.state not in [4,5]:
            action = self.option(first)

            if action == 2:
                if first is self.BB: # player is OOP
                    self.prev_action_training[self.current_street*2] += 1
                else: # player is IP
                    self.prev_action_training[self.current_street*2 + 1] += 1

            

            self.state = self.street_DFA(self.state, action)

            # switch player to act
            first, second = second, first

        
        if self.state == 4:
            self.end_hand()
        elif self.state == 5:
            self.prev_raise = 0
            self.total_bet = 0
        else:
            print("Error. Street DFA is invalid")
    
    def street_DFA(self, state, action):
        '''Represents the DFA for a street of action. The states are as follows
        
        1) initial
        2) raised to
        3) checked to
        4) end of hand
        5) end of street
        
        '''

        if state == 1:
            if action == 1: # call
                state = 3
            elif action == 2: # raise
                state = 2
            elif action == 3: # fold
                state = 4
        elif state == 2:
            if action == 1: # call
                state = 5
            elif action == 2: # raise
                state = 2
            elif action == 3: # fold
                state = 4
        elif state == 3:
            if action == 1: # call
                state = 5
            elif action == 2: # raise
                state = 2
            elif action == 3: # fold
                state = 4
        
        return state
        
    def option(self, player):
        '''gives the player the option to act
        
        They can choose to check, bet, call, raise, or fold

        returns the action taken
        '''
        if self.print:
            print(f"--- OPTION to {player.playerName}---\n")

        if self.all_in == True:
            return self.all_in_option(player)

        villain = self.get_villain(player)

        while True:
            if self.print:
                print(f"Pot size: {self.pot}. {player.playerName} stack ({player.position}): {player.stack}. {villain.playerName} stack ({villain.position}): {villain.stack}")
                self.print_board()
                if player.sweat: # only show if we can sweat them
                    if len(self.board.collection) >= 3:
                        print(f"{player.playerName} hand: {player.hand} ({find_best_hand(player.hand.collection + self.board.collection)[2]})")
                    else:
                        print(f"{player.playerName} hand: {player.hand}")
            # action = input(f"1. Call {self.prev_raise} (total so far: {self.total_bet})\n2. Raise\n3. Fold\n")
            player.train(self)
            action = player.take_action(self)
            if self.print:
                print()
            if not action.isnumeric() or int(action) not in [1,2,3]:
                if self.print:
                    print("Invalid action")
            else:
                action = int(action)
                break

        if action == 1: # call
            self.action_call(player)
        elif action == 2: # raise
            self.action_raise(player)
        elif action == 3: # fold
            self.winner = villain
        
        return action

    def all_in_option(self, player):
        villain = self.get_villain(player)

        while True:
            if self.print:
                print(f"Pot size: {self.pot}. {player.playerName} stack: {player.stack}. {villain.playerName} stack: {villain.stack}")
                self.print_board()
                if player.sweat:
                    if len(self.board.collection) >= 3:
                        print(f"{player.playerName} hand: {player.hand} ({find_best_hand(player.hand.collection + self.board.collection)[2]})")
                    else:
                        print(f"{player.playerName} hand: {player.hand}")
            # action = input(f"1. Call {self.prev_raise} (total so far: {self.total_bet})\n3. Fold\n")
            action = player.take_action(self)
            if not action.isnumeric() or int(action) not in [1,3]:
                if self.print:
                    print("Invalid action")
            else:
                action = int(action)
                break

        if action == 1: # call
            self.action_call(player)
        elif action == 3: # fold
            self.winner = villain

        return action
             
    def action_call(self, player: Player):
        '''player makes the call'''
        player.stack -= self.prev_raise
        self.pot += self.prev_raise
        self.prev_raise = 0

    def action_raise(self, player: Player):
        '''player makes a raise'''

        # a player should only have been able to raise, if they have enough to call and then raise
        # so first deduct the call from their stack
        player.stack -= self.prev_raise
        self.pot += self.prev_raise

        while True:
            # size = input(f"Raise Size")
            size = player.determine_sizing(self)
            if size.isnumeric():
                size = int(size)
                break
            else:
                if self.print:
                    print("Invalid raise size")

        villain = self.get_villain(player)



        # check for all in
        if size - self.total_bet >= min(player.stack, villain.stack):
            # player is all in, can ignore the min bet rules
            if self.print:
                print("Player is ALL-IN")
            self.all_in = True
            size = self.total_bet + min(player.stack, villain.stack)
        else:
            if self.prev_raise == 0: #i.e. if first to betting on street
                size = max(1, size) #must bet atleast 1 BB
            else: #i.e. someone has bet previously
                size = max(self.total_bet + self.prev_raise, size)
            
            # check again for all in, since minimum betting rules might have forced player all in
            if size - self.total_bet >= min(player.stack, villain.stack):
                if self.print:
                    print("NOTE: Minimum betting rules has forced player ALL-IN")
                # player is all in, can ignore the min bet rules
                self.all_in = True
                size = self.total_bet + min(player.stack, villain.stack)



        self.prev_raise = size - self.total_bet
        player.stack -= self.prev_raise
        self.pot += self.prev_raise
        self.total_bet = size
    
    def end_hand(self):
        '''hands the hand and does clean up procedure'''

        if self.print:
            print("\n--- END OF HAND ---\n")
            if self.winner:
                print(f"{self.winner.playerName} wins the hand for a pot of {self.pot}")
            else:
                print(f"Pot of {self.pot} is chopped")

        # gives the winner the pot, or chops
        if self.winner:
            self.winner.stack += self.pot
            self.pot = 0
        else: # chop
            self.player1.stack += self.pot/2
            self.player2.stack += self.pot/2

        #recover cards from hands and board
        while self.player1.hand.collection:
            self.deck.append(self.player1.hand.pop())

        while self.player2.hand.collection:
            self.deck.append(self.player2.hand.pop())

        while self.board.collection:
            self.deck.append(self.board.pop())

        #shuffle deck
        self.deck.shuffle()

        self.winner = None
        self.all_in = False
        self.game_over = True

    def deal_community(self, n: int):
        '''deals "n" number of community cards i.e. 3 for flop, 1 for turn, 1 for river'''

        for _ in range(n):
            self.board.append(self.deck.pop())

    def get_villain(self, player: Player):
        '''given player, returns the villain'''
        return self.button if player is self.BB else self.BB

                 


if __name__ == "__main__":

    # different ACCESS states will allow you to access different versions of the application
    # 1. run a game of poker between any two agents (e.g. user vs user, user vs CPU, CPU vs CPU)
    # 2. same as 1 but skip preflop by manually assigning ranges
    # 3. run in training mode to train a DQN agent
    # 4. test agent trained in 3
    ACCESS = 1

    if ACCESS == 1:

        # p1 = RandomAgent(1, 100, playerName = "RandomAgent1", sweat = True)
        p2 = RandomAgent(2, 100, playerName = "RandomAgent2", sweat = True)

        p1 = Player(1, 100, playerName = "User1", sweat = True)
        # p2 = ValueAgent(1, 100, playerName = "ValueAgent1", sweat = True)

        g = Game(player1 = p1, player2 = p2)
        g.ACCESS = ACCESS
        g.print = True

        for _ in range(10):
            g.new_hand()


        print(f"GAME OVER. RESULTS:")
        g.print_stack(g.player1)
        g.print_stack(g.player2)
    elif ACCESS == 2:

        # p1 = RandomAgent(1, 100, playerName = "RandomAgent1", sweat = True)
        p2 = RandomAgent(2, 100, playerName = "RandomAgent2", sweat = True)

        p1 = Player(1, 100, playerName = "User1")
        # p2 = ValueAgent(1, 100, playerName = "ValueAgent1", sweat = True)

        p1.preflopRange = cardRange.CardRange()
        p2.preflopRange = cardRange.CardRange()

        p1.preflopRange.add_linear_range(base = 14, high = 13, low = 10, suited = False)
        p2.preflopRange.add_linear_range(base = 14, high = 13, low = 2, suited = True)

        g = Game(player1 = p1, player2 = p2)
        g.ACCESS = ACCESS
        g.print = True

        # this denotes the number of BB already invested preflop by each player
        g.BB_INVESTED = 20

        for _ in range(10):
            g.new_hand()


        print(f"GAME OVER. RESULTS:")
        g.print_stack(g.player1)
        g.print_stack(g.player2)
    elif ACCESS == 3:
        p1 = DQNAgentTrainer(1, 100, playerName = "DQN", sweat = True)
        p2 = ValueAgent(2, 100, playerName = "ValueAgent2", sweat = False)

        p1.preflopRange = cardRange.CardRange()
        p2.preflopRange = cardRange.CardRange()

        p1.preflopRange.add_linear_range(base = 14, high = 13, low = 10, suited = False)
        p2.preflopRange.add_linear_range(base = 14, high = 13, low = 10, suited = False)


        g = Game(player1 = p1, player2 = p2)
        g.ACCESS = ACCESS
        g.print = False

        # this denotes the number of BB already invested preflop by each player
        g.BB_INVESTED = 20

        for _ in range(128):
            g.new_hand()

        print(f"GAME OVER. RESULTS:")
        g.print_stack(g.player1)
        g.print_stack(g.player2)

    elif ACCESS == 4:
        p1 = DQNAgentTester(1, 100, playerName = "DQN", sweat = True)
        p2 = ValueAgent(2, 100, playerName = "ValueAgent2", sweat = False)

        
        # load in the model
        MODEL_SAVE_PATH = Path("model.pt")
        p1.load_state_dict(MODEL_SAVE_PATH)


        p1.preflopRange = cardRange.CardRange()
        p2.preflopRange = cardRange.CardRange()

        p1.preflopRange.add_linear_range(base = 14, high = 13, low = 10, suited = False)
        p2.preflopRange.add_linear_range(base = 14, high = 13, low = 10, suited = False)


        g = Game(player1 = p1, player2 = p2)
        g.ACCESS = ACCESS

        # this denotes the number of BB already invested preflop by each player
        g.BB_INVESTED = 20

        for _ in tqdm(range(10000), unit = " hands"):
            g.new_hand()


        print(f"GAME OVER. RESULTS:")
        g.print_stack(g.player1)
        g.print_stack(g.player2)