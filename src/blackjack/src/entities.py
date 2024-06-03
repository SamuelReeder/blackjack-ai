from .hand import Hand

class Player:
    def __init__(self, balance: int = 1000000000):
        self.hand = Hand()
        self.balance = balance
        self.insurance = False
        self.split = False
        
    def change_balance(self, amount: int) -> None:
        print(f"Changing balance by {amount}")
        self.balance += amount
        
    def reset(self) -> None:
        self.hand = Hand()
        self.insurance = False
        self.split = False
        self.hand.bet = 0

class Dealer(Player):
    def __init__(self):
        super().__init__()