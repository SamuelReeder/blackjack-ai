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

class Dealer(Player):
    def __init__(self):
        super().__init__()