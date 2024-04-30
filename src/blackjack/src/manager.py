from .game import Game 
from .deck import Deck
from .entities import Player, Dealer
from typing import List


class Manager:
    
    def __init__(self) -> None:
        self.deck = Deck()
        self.players: List[Player] = [Player()]
        self.dealer = Dealer()
        self.balance: List[float] = []
        self.game_over = False
    
    def new_game(self) -> None:
        self.current_game = Game(self.players, self.dealer, self.deck)
        return self.current_game.init_round()
    
    def play_game(self, action: int) -> None:
        wins, losses = 0, 0
        
        hand_index = 0
        curr_hand = self.players[0].hand[0]
        for i, element in enumerate(self.players[0].hand):
            if not element.complete:
                curr_hand = element
                hand_index = i
                
        match action:
            case 0:
                self.current_game.stay(curr_hand)
            case 1:
                self.current_game.hit(curr_hand)
            case 2:
                self.current_game.split(curr_hand)
            case 3:
                self.current_game.insure()
        
        if curr_hand.complete and hand_index == len(self.players[0].hand) - 1:
            self.current_game.dealer_play()
            return self.current_game.end()
        
        redundant = -10 if (not curr_hand.split_possible and action == 3) or (not curr_hand.insurance_possible and action == 2) else 0
        
        return self.current_game.get_state(curr_hand), redundant, False, False, {}
            
    
    
        
    