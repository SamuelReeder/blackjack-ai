from .game import Game 
from .deck import Deck
from .entities import Player, Dealer
from typing import List


class Manager:
    
    def __init__(self) -> None:
        self.deck = Deck()
        self.player: Player = Player()
        self.dealer = Dealer()
        self.balance: List[float] = []
            
    def new_game(self) -> None:
        self.player.reset()
        self.dealer.reset()
        self.current_game = Game(self.player, self.dealer, self.deck, debug=True)
        return self.current_game.init_round()
    
    def play_game(self, action: int) -> None:
                        
        match action:
            case 0:
                self.current_game.stay(self.player.hand)
            case 1:
                self.current_game.hit(self.player.hand)
            case 2:
                self.current_game.split(self.player.hand)
            case 3:
                self.current_game.insure()
        
        if self.player.hand.complete:
            self.current_game.dealer_play()
            return self.current_game.end()
        
        redundant = -10 if (not self.player.hand.split_possible and action == 3) or (not self.player.hand.insurance_possible and action == 2) else 0
        
        return self.current_game.get_state(self.player.hand), redundant, False, False, {}
            
    
    
        
    