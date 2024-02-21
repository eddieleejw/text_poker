import random
from termcolor import colored, cprint

class Card:
    valid_values = {"2","3","4","5","6","7","8","9","T","J","Q","K","A"}
    valid_suits = {"s","c","d","h"}

    def __init__(self, value, suit):
        assert(value in self.valid_values)
        assert(suit in self.valid_suits)
        self.value = value
        self.suit = suit
    
    def __str__(self):
        color = "black"
        if self.suit == "c":
            color = "green"
        elif self.suit == "h":
            color = "red"
        elif self.suit == "d":
            color = "blue"
        
        return colored(f"{self.value}{self.suit}", color)
    
    def __eq__(self, other):
        if self.value == other.value and self.suit == other.suit:
            return True
        else:
            return False
    
    def __ne__(self, other):
        return not self == other
    
    def __lt__(self, other):
        return (self.value, self.suit) < (other.value, other.suit)

    def __hash__(self):
        return hash((self.value, self.suit))
        
    

class CardCollection:
    '''Defines a general class that is some collection of cards, represented as a list of those cards'''

    def __init__(self, collection = None):
        if collection is None:
            self.collection = []
        else:
            self.collection = collection


    def __str__(self):
        if self.collection:
            ret = ""

            for card in self.collection:
                ret += str(card)
                ret += " "

            return ret
        else:
            return "empty"
    
    def pop(self):
        return self.collection.pop()

    def append(self, x):
        self.collection.append(x)

    def overlap(self, other):
        '''
        Returns True if at least one Card in self is in other (and vice versa)
        '''
        s1 = set(self.collection)
        s2 = set(other.collection)

        if s1.intersection(s2):
            return True
        else:
            return False
        
    def length(self) -> int:
        if self.collection is not None:
            return len(self.collection)
        else:
            return 0
    
    def remove(self, card):
        '''
        Remove a given card from the collection
        '''
        self.collection.remove(card)

    def __hash__(self):
        # self.collectio is list(Card), so sort the list, and then hash the resulting tuple
        return hash(tuple(sorted(self.collection)))
    
    def __eq__(self,other):
        s1 = sorted(self.collection)
        s2 = sorted(other.collection)

        return s1 == s2
    
class Deck(CardCollection):
    def __init__(self, deck = None):
        '''
        :param deck: the current collection of cards
        :type deck: list
        '''
        super().__init__(deck)
    
    def new_deck(self):
        '''Resets the deck to a new shuffled 52 card deck'''
        self.collection = []
        for value in ["2","3","4","5","6","7","8","9","T","J","Q","K","A"]:
            for suit in ["s","c","d","h"]:
                self.collection.append(Card(value, suit))
        
        # shuffle deck
        self.shuffle()
    
    def shuffle(self):
        '''Shuffle the deck'''
        random.shuffle(self.collection)

    




class Hand(CardCollection):
    def __init__(self, hand = None):
        '''
        :param hand: the current two card hand
        :type hand: list
        '''
        super().__init__(hand)
    
    

class Board(CardCollection):
    def __init__(self, board = None):
        super().__init__(board)



if __name__ == "__main__":
    hand1 = Hand()
    hand2 = Hand()
    hand3 = Hand()
    
    hand1.append(Card("6","s"))
    hand1.append(Card("A","c"))


    print(hand1)

    hand1.remove(Card("A", "c"))
    
    print(hand1)