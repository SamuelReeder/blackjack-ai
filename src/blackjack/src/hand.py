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

    def add_card(self, card: Card) -> None:
        self.cards.append(card)
    
    def calculate_value(self, hide_dealer: bool = False) -> None:
        self.value = 0
        for i, card in enumerate(self.cards):
            if self.dealer and i > 0 and hide_dealer:
                break
            if card > 1:
                self.value += int(card)
            else:
                self.has_ace = True
                self.value += 11

        if self.has_ace and self.value > 21:
            self.value -= 10
            
        return self.value
    
    def reset(self) -> None:
        self.cards = []
        self.value = 0
        
    def display(self, hide: bool = False) -> None:
        if self.dealer and hide:
            print("Dealer's hand:", self.cards[1])
            print("Dealer's hand:", self.cards)
            if self.cards[1] == 1:
                self.insurance_possible = True
            return
        elif self.dealer:
            print("Dealer's hand:", self.cards)
            return
            
        if len(self.cards) == 2:
            if self.cards[0] == self.cards[1]:
                self.split_possible = True
            
        print("Your hand:", self.cards)
    
    def split() -> tuple:
        temp = self.cards.pop()
        new_hand = Hand()
        new_hand.add_card(temp)
        return new_hand