from .card import Card

class Hand:
    def __init__(self, dealer=False):
        self.dealer = dealer
        self.bet = 0
        self.cards = []
        self.value = 0
        self.has_ace = False
        self.insurance_possible = False
        self.split_possible = False
        self.complete = False

    def add_card(self, card: int) -> None:
        self.cards.append(card)
   
    def calculate_value(self, hide_dealer: bool = False) -> int:
        self.value = 0
        self.has_ace = False
        for i, card in enumerate(self.cards):
            if self.dealer and i == 0 and hide_dealer:
                continue
            if card == 1:
                self.has_ace = True
                self.value += 11
            else:
                self.value += min(card, 10)
        
        if self.has_ace and self.value > 21:
            self.value -= 10
           
        return self.value
   
    def reset(self) -> None:
        self.cards = []
        self.value = 0
        self.has_ace = False
        self.insurance_possible = False
        self.split_possible = False
        self.complete = False
   
    def split(self) -> 'Hand':
        temp = self.cards.pop()
        new_hand = Hand()
        new_hand.add_card(temp)
        new_hand.bet = self.bet
        return new_hand

    def to_dict(self) -> dict:
        return {
            "cards": self.cards,
            "value": self.calculate_value(),
            "bet": self.bet,
            "complete": self.complete
        }