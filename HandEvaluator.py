from card import Card, CardCollection, Deck, Hand, Board

def find_best_hand(all_cards) -> list:
    '''given some collection of cards, determines the best 5 card combination, and outputs the result

    The return value is a list [a, b. c], where a is the "rank" of the combination corresponding to below (e.g.
    2 is a four of a kind). b is used to break ties between combinations of equal rank, and is unique to
    the rank (e.g. for flushes, the highest card is sufficient. For pair, the value of the pair, and the
    next 3 highest card is sufficient). "c" is the string form of the hand made
    
    In order to do that, it checks for the following hands in the order of decreasing strength:

    0) Royal flush 
    1) Straight flush
    2) Four of a kind
    3) Full house
    4) Flush
    5) Straight
    6) Three of a kind
    7) Two pair
    8) Pair
    9) High card

    To do that, first check for flush (5 cards of the same suit)

    If there is a flush:
        Check if there is a straight with the cards of the same suit
            If there is, determine the highest card in that flush (if A high, then royal flush, else straight flush)
            
            If not, then still determine the highest card in that flush
                Note: We don't need to find the remaining cards in the flush, because there can be no ties
                e.g. you and the opponent cannot both have an A-high flush
    
    If there is no flush:
        Count how many instances of 4 of a kind, 3 of a kind, and 2 of a kind there

        If there is are 4 of a kind, find the highest one
            Note: once again, there can be no ties in 4 of a kind
        
        If there are no 4 of a kind, check if there is atleast one 3 of a kind and atleast 2 of a kind
            If there is, make full house by picking the highest value 3 of a kind, and highest value 2 of a kind
        
            
        If no full house, check for straight
            If there is, pick the highest straight
        
        If no straight, check if there is at least one 3 of a kind.
            If there is, pick the highest one.
                Fill in the remining 2 with highest value cards
            
        If no 3 of a kind, check for atleast one 2 of a kind.
            If there are two, pick the two highest one
                Pick the remaining 1 card with highest value
            
            If only one, pick that one
                Pick remaining 3 cards of highest value
        
        If nothing, then pick the 5 highest cards in descending order

        
    In order to do that, we can represent the 7 cards with two lists:
        1) The first is of length 13, where each index represents one of the cards from 2~A, and the entry
            correlates to the number of occurencres of that card

        2) Second is of length 4, where each index represents on of the 4 suits. Each index holds a list of all
            the values of that suit, so that the number of cards of some suit is the length of the list at that
            index
        
    Then, we can find flushes based on checking the length of each entry of list 2
    We can find straights by checking if there are 5 consecutive non-zero entries in list 1
        And handle the case of A~5 straight as a special case by itself.
    '''
    VALUES = ["2","3","4","5","6","7","8","9","T","J","Q","K","A"]
    SUITS = ["s", "c", "d", "h"]

    value_count = [0 for _ in range(13)]
    suit_count = [[] for _ in range(4)]
    all_values = []

    # all_cards = hand.collection + board.collection

    for card in all_cards:
        idx = value_index(card.value)
        suit_idx = suit_index(card.suit)
        if idx == -1:
            print(f"ERROR: card value is invalid: {card.value}")
            assert(False)
        if suit_idx == -1:
            print(f"ERROR: card suit is invalid: {card.suit}")
            assert(False)

        value_count[idx] += 1
        suit_count[suit_idx].append(idx)

        all_values.append(idx)
    
    all_values.sort(reverse=True)

    # check for flush
    flush_suit = -1
    for i in range(4):
        if len(suit_count[i]) >= 5:
            flush_suit = i
    
    # if flush, check for straight flush
    if flush_suit != -1:
        straight_flush_high = find_highest_straight(suit_count[flush_suit])
        if straight_flush_high != -1:
            # there is a straight flush
            return [1, straight_flush_high, f"{VALUES[straight_flush_high]} high straight flush"]


    # no straight flush, check for 4 of a kind

    # to make this efficient so we don't have to recompute later, traverse the entire value count
    # and keep track of how many occurences of 4/3/2 of a kind
    x_of_a_kind_dict = {1:[], 2:[], 3:[], 4:[]}
    for i, count in enumerate(value_count):
        if count > 0:
            x_of_a_kind_dict[count].append(i)

    # sort each in descending order
    for l in x_of_a_kind_dict.values():
        l.sort(reverse=True)
    
    # check for 4 of a kind
    if x_of_a_kind_dict[4]:
        tie_break = x_of_a_kind_dict[4][0]

        # pick the highest vard from the remaining
        for val in all_values:
            if val != tie_break:
                kicker = val
                break

        return [2, (tie_break, kicker), f"Four of a kind ({VALUES[tie_break]}) with {VALUES[kicker]} kicker"]
    

    # check for full house

    '''
    There are a couple of different ways to make full houses.
    They all require:
        1) At least one trip
        2) At least one other pair or trip (we can ignore quad, because that should have been caught above)

    So the only two cases we are concerned with is trip + pairs, or trip + trip

    Note that they are mutually exclusive. That is, if you have a trip and at least one pair, you 
    cannot have another trip. Similarly, if you have two trips, you cannot have another pair

    So we don't need to worry about the tie breaks between the two cases
    '''

    # first possible ful house
    if x_of_a_kind_dict[3] and x_of_a_kind_dict[2]:
        FH_three = max(x_of_a_kind_dict[3]) # there can only be one 
        FH_two = max(x_of_a_kind_dict[2])
        return [3, (FH_three, FH_two) , f"Full house ({VALUES[FH_three]}s full of {VALUES[FH_two]}s)"]

    # second possible full house
    if len(x_of_a_kind_dict[3]) == 2:
        FH_three = max(x_of_a_kind_dict[3])
        FH_two = min(x_of_a_kind_dict[3])
        return [3, (FH_three, FH_two) , f"Full house ({VALUES[FH_three]}s full of {VALUES[FH_two]}s)"]
    

    # check for regular fush
    if flush_suit != -1:

        # for tie break, return the 5 highest cards in the flush in order
        suit_count[flush_suit].sort(reverse = True)

        tie_break = suit_count[flush_suit][:5]
        return [4, tie_break, f"{VALUES[tie_break[0]]} high flush"]

    # check for straight
    highest_straight = find_highest_straight(all_values)
    if highest_straight != -1:
        return [5, highest_straight, f"{VALUES[highest_straight]} high straight"]

    # check for trips
    # for trips, we have to start thinking about the kickers
    # to do this, just take the highest remaining 2 cards that doesn't contribute to the trip
    # further note that we cannot have a remaining pair or better, as this would give a FH
    if x_of_a_kind_dict[3]:
        tie_break = max(x_of_a_kind_dict[3]) # note: there can only be one trip, else we have FH
        # find kickers
        top_kicker = x_of_a_kind_dict[1][0]
        bottom_kicker = x_of_a_kind_dict[1][1]
        return [6, (tie_break, top_kicker, bottom_kicker), f"Three of a kind ({VALUES[tie_break]} with {VALUES[top_kicker]}, {VALUES[bottom_kicker]} kicker)"]

    # check for two pair
    # note that at this point, we cannot have any quads or trips, but we can have 0,1,2,3 pairs made
    # we do have to account for the fact that we can have 3 pairs, in which case we cannot find a kicker
    if len(x_of_a_kind_dict[2]) >= 2:
        top_pair = x_of_a_kind_dict[2][0]
        bottom_pair = x_of_a_kind_dict[2][1]

        if x_of_a_kind_dict[1]:
            kicker = x_of_a_kind_dict[1][0]
        else: # this means that we have no lone card, so we must only have pairs
            if x_of_a_kind_dict[2]:
                kicker = x_of_a_kind_dict[2][2]
            else:
                print("ERROR. TWO PAIR KICKER NOT WORKING")
                print(x_of_a_kind_dict)
                exit()
    
        # kicker = x_of_a_kind_dict[1][0]
        return [7, (top_pair, bottom_pair, kicker), f"Two pair ({VALUES[top_pair]}s and {VALUES[bottom_pair]}s, with {VALUES[kicker]} kicker)"]


    # check for pair
    # again, at this point we cannot have quads or trips, and we can have 0 or 1 pairs made
    if len(x_of_a_kind_dict[2]) == 1:
        pair_value = x_of_a_kind_dict[2][0]

        top_kicker = x_of_a_kind_dict[1][0]
        mid_kicker = x_of_a_kind_dict[1][1]
        bot_kicker = x_of_a_kind_dict[1][2]
        return [8, (pair_value, top_kicker, mid_kicker, bot_kicker), f"Pair of {VALUES[pair_value]}s, with {VALUES[top_kicker]} {VALUES[mid_kicker]} {VALUES[bot_kicker]} kicker"]

    # return HC
    # if we get to this point, all we have is high card
    high_cards = tuple(x_of_a_kind_dict[1][:5])
    return [9,high_cards, f"{VALUES[high_cards[0]]} high, with {VALUES[high_cards[1]]} {VALUES[high_cards[2]]} {VALUES[high_cards[3]]} {VALUES[high_cards[4]]} kicker"]
    
    
def find_highest_straight(l: list):
    '''given a list, find the highest straight, and return the highest value

    handle special case of A~5 straight
    
    if there is no straight, return -1
    '''

    if len(l) == 0:
        return -1

    l.sort(reverse=True)

    consecutive = 1
    ret = l[0]

    for i in range(len(l)-1):
        if l[i] == l[i+1] + 1:
            consecutive += 1
        elif l[i] == l[i+1]:
            continue
        else:
            consecutive = 1
            ret = l[i+1]
        
        if consecutive == 5:
            return ret
        
    if consecutive == 4 and ret == 3: # if "l" contains 5,4,3,2
        if l[0] == 12: # if "l" contains A
            return ret
        else:
            return -1

    return -1


def value_index(s):
    '''
    Given a string like "3" or "J", gives the corresponding index in the value count
    '''
    for i, value in enumerate(["2","3","4","5","6","7","8","9","T","J","Q","K","A"]):
        if s == value:
            return i
    return -1


def suit_index(s):
    for i, suit in enumerate(["s", "c", "d", "h"]):
        if s == suit:
            return i
        
    return -1





if __name__ == "__main__":
    hand = Hand()
    board = Board()

    straight_flush = [Card("A", "h"), Card("K", "h"), Card("Q","h"), Card("J","h"), Card("T","h"), Card("9","h"), Card("8","h")]
    four_of_a_kind = [Card("T", "h"), Card("T", "c"), Card("T","d"), Card("T","s"), Card("J","h"), Card("A","h"), Card("A","s")]
    full_house = [Card("9", "h"), Card("9", "c"), Card("9","d"), Card("A","s"), Card("A","h"), Card("A","c"), Card("3","d")]
    flush = [Card("Q", "h"), Card("2", "h"), Card("3","h"), Card("4","h"), Card("7","h"), Card("5","h"), Card("K","h")]
    straight = [Card("7", "h"), Card("6", "h"), Card("5","s"), Card("4","c"), Card("3","s"), Card("2","c"), Card("A","s")]
    trips = [Card("9", "h"), Card("9", "c"), Card("9","d"), Card("A","s"), Card("J","h"), Card("4","h"), Card("5","s")]
    two_pair = [Card("9", "h"), Card("2", "c"), Card("3","d"), Card("A","s"), Card("3","h"), Card("5","h"), Card("5","s")]
    pair =  [Card("A", "h"), Card("J", "c"), Card("A","d"), Card("3","s"), Card("5","h"), Card("6","h"), Card("Q","s")]
    high_card = [Card("9", "h"), Card("J", "c"), Card("2","d"), Card("3","s"), Card("5","h"), Card("6","h"), Card("Q","s")]


    two_pair_debug = [Card("2","h"), Card("2","s"), Card("3","h"), Card("3","s"), Card("4","h"), Card("4","s")]

    print(find_best_hand(two_pair_debug)[2])


