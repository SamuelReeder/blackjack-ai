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
            if card < 10 and card > 1:
                self.value += int(card)
            else:
                if card == 1:
                    self.has_ace = True
                    self.value += 11
                else:
                    self.value += 10

        if self.has_ace and self.value > 21:
            self.value -= 10

    def get_value(self, one_card: bool = False) -> int:
        self.calculate_value(hide_dealer=one_card)
        return self.value
    
    def reset(self) -> None:
        self.cards = []
        self.value = 0
        
    def display(self, hide: bool = True) -> None:
        if self.dealer:
            if hide:
                print("hidden")
                print(self.cards[1].rank)
                if self.cards[1].rank == 1:
                    self.insurance_possible = True
                return 
            if len(self.cards) == 2:
                if self.cards[0].rank == self.cards[1].rank:
                    self.split_possible = True
            for card in self.cards:
                print(card, end=" ")
            # print("Value:", self.get_value())
            
                
        else:
            for card in self.cards:
                print(card, end=" ")
            # print("Value:", self.get_value())
    
    def split() -> tuple:
        temp = self.cards.pop()
        new_hand = Hand()
        new_hand.add_card(temp)
        return new_hand