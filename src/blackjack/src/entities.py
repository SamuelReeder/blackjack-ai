from .hand import Hand
from typing import Optional

class Player:
    def __init__(self, balance: int = 1000000000):
        self.hands = [Hand()]
        self.balance = balance
        self.insurance = False
       
    def change_balance(self, amount: int) -> None:
        self.balance += amount
       
    def reset(self) -> None:
        self.hands = [Hand()]
        self.insurance = False

    def add_hand(self, hand: Hand) -> None:
        self.hands.append(hand)

    def get_active_hand(self) -> Optional[Hand]:
        for hand in self.hands:
            if not hand.complete:
                return hand
        return None

class Dealer(Player):
    def __init__(self):
        super().__init__()
        self.hand = Hand(dealer=True)

    def reset(self) -> None:
        self.hand = Hand(dealer=True)
        self.insurance = False