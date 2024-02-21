import random
import card

class CardRange(card.CardCollection):
    '''
    A collection of hands

    Note: we do not have mixed strategies (i.e. the range either contains a hand or doesn't, rather than
    x% chance of containing)
    '''

    NUM_TO_VALUE = ["-1","-1","2","3","4","5","6","7","8","9","T","J","Q","K","A"]
    SUITS = ["s","c","d","h"]

    def __init__(self):
        super().__init__([])
        self.collection_set = set()
    
    def pop_random(self) -> card.Hand:
        if len(self.collection) == 0:
            assert(self.check_valid())
            return None
        else:
            ret = self.collection.pop(random.randint(0, len(self.collection)-1))
            self.collection_set.remove(ret)
            assert(self.check_valid())
            return ret
    
    def seek_random(self) -> card.Hand:
        '''
        Same as pop_random, but doesn't pop it
        '''
        assert(self.check_valid())
        if len(self.collection) == 0:
            return None
        else:
            return self.collection[random.randint(0, len(self.collection)-1)]


        
    def add(self, hand : card.Hand):
        self.collection.append(hand)
        self.collection_set.add(hand)
        assert(self.check_valid())

    def remove(self, hand: card.Hand):
        self.collection.remove(hand)
        self.collection_set.remove(hand)
        assert(self.check_valid())
    
    def check_valid(self):
        return len(self.collection) == len(self.collection_set)


    def add_linear_range(self, base: int, high: int, low: int, suited: bool):
        '''
        Adds to the range a collection of cards of the form (base, high~low, suit/offsuit)

        e.g. if base = 8, high = 6, low = 3, suited = False
        then it will add 86o, 85o, 84o, 83o

        note: 14 >= base > high >= low >= 2
        '''
        assert(base <= 14)
        assert(base > high)
        assert(high >= low)
        assert(low >= 2)

        if suited == False:
            for i in range(low, high+1):
                for first_suit in CardRange.SUITS:
                    for second_suit in CardRange.SUITS:
                        if first_suit == second_suit:
                            # we are only adding offsuit cards
                            continue
                        
                        first_card = card.Card(CardRange.NUM_TO_VALUE[base], first_suit)
                        second_card = card.Card(CardRange.NUM_TO_VALUE[i], second_suit)
                        # print(f"Debug: Hand added: {first_card} {second_card}")
                        self.add(card.Hand([first_card, second_card]))
        else: # add suited hands
            for i in range(low, high+1):
                for suit in CardRange.SUITS:

                    first_card = card.Card(CardRange.NUM_TO_VALUE[base], suit)
                    second_card = card.Card(CardRange.NUM_TO_VALUE[i], suit)
                    # print(f"Debug: Hand added: {first_card} {second_card}")
                    self.add(card.Hand([first_card, second_card]))

    
    def add_pocket_pairs(self, high: int, low: int):
        '''
        Add pocket pairs from [low, high]

        note: 14 >= high >= low >= 2
        '''
        assert(14 >= high)
        assert(high >= low)
        assert(low >= 2)

        for val in range(low, high+1):
            for i in range(4):
                first_suit = CardRange.SUITS[i]
                for j in range(i+1, 4):
                    second_suit = CardRange.SUITS[j]
                    first_card = card.Card(CardRange.NUM_TO_VALUE[val], first_suit)
                    second_card = card.Card(CardRange.NUM_TO_VALUE[val], second_suit)
                    # print(f"Debug: Hand added: {first_card} {second_card}")
                    self.add(card.Hand([first_card, second_card]))







if __name__ == "__main__":
    cardrange1 = CardRange()
    cardrange2 = CardRange()

    cardrange1.add_linear_range(base = 14, high = 13, low = 2, suited = False)
    cardrange2.add_linear_range(base = 13, high = 12, low = 2, suited = False)


    for _ in range(1000):
        c1 = cardrange1.seek_random()
        c2 = cardrange2.seek_random()

        if c1.overlap(c2):
            print(f"{c1} and {c2} overlap")

        # cardrange1.add(c1)
        # cardrange2.add(c2)



