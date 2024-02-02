from .hand import Hand

class Player:
    def __init__(self, balance: int = 1000000000):
        self.hand: List[Hand] = []
        self.balance = balance
        self.bet = 0
        self.insurance = False
        self.split = False
        self.new_hand = None
    def change_balance(self, amount: int) -> None:
        self.balance += amount

class Dealer(Player):
    def __init__(self):
        super().__init__()