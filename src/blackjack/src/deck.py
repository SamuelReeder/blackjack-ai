from .card import Card
import random


class Deck:
    def __init__(self) -> None:
        self.reset()
                
    def shuffle(self) -> None:
        random.shuffle(self.cards)
        
    def reset(self) -> None:
        self.cards = [v for v in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10] * 4]
        self.cards = self.cards * 6
        self.number_of_decks = 6
        self.original_length = len(self.cards)
        self.true_count = 0
        self.current_count = 0
        self.face_tally = 0
        self.shuffle()
        self.card_dict = {}
        self.card_arr = [0] * 10
        self.init_card_dict()
        self.cards.insert(random.randrange(len(self.cards) // 2 + 1, len(self.cards)), -1)
        
    def update_count(self, card: int) -> None:
        if card in [2, 3, 4, 5, 6]:
            self.current_count += 1
            cards_left = self.number_of_decks - ((self.original_length - len(self.cards))) 
            if cards_left != 0:
                self.true_count = self.current_count / (cards_left / 52)
        elif card == 10 or card == 1:
            self.current_count -= 1
            self.true_count -= 1

    def deal(self) -> int:
        if len(self.cards) < 1:
            self.reset()
            
        if self.cards[-1] == 10 or self.cards[-1] == 1:
            self.face_tally += 1
    
        card = self.cards.pop()
        
        if card != -1:
            self.card_dict[card] -= 1
            self.card_arr[card - 1] -= 1
            self.update_count(card)
        # else:
        #     self.reset()
        #     return self.deal()
        return card
    
    def init_card_dict(self) -> None:
        for card in self.cards:
            if card in self.card_dict:
                self.card_dict[card] += 1
                self.card_arr[card - 1] += 1
            else:
                self.card_dict[card] = 1
                self.card_arr[card - 1] = 1
                
    def remove_card(self, card: int) -> None:
        self.card_dict[card] -= 1
        self.card_arr[card - 1] -= 1
        self.update_count(card)