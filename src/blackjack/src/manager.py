from .game import Game, GameResult
from .deck import Deck
from .entities import Player, Dealer
from .hand import Hand
from typing import List, Tuple, Dict, Any, Optional

class Manager:
    def __init__(self, balance: int = 1000) -> None:
        self.deck = Deck()
        self.player: Player = Player(balance)
        self.dealer = Dealer()
        self.current_game: Game = None

    def new_game(self, bet: int) -> Tuple[List[Any], bool]:
        self.player.reset()
        self.dealer.reset()
        self.current_game = Game(self.player, self.dealer, self.deck, bet)
        return self.current_game.init_round()

    
    def play_game(self, action: int) -> Tuple[List[Any], int, bool, bool, Dict[str, Any]]:
        if not self.current_game:
            raise ValueError("Game not initialized. Call new_game() first.")

        current_hand = self.player.get_active_hand()
        if not current_hand:
            raise ValueError("No active hand.")

        game_over = False
        result = {}

        match action:
            case 0:  # Stay
                self.current_game.stay(current_hand)
                game_over = True
            case 1:  # Hit
                self.current_game.hit(current_hand)
            case 2:  # Split
                if current_hand.split_possible:
                    new_hand = self.current_game.split(current_hand)
                    if new_hand:
                        self.player.add_hand(new_hand)
                else:
                    game_over = True  # End game if invalid split action
            case 3:  # Insure
                if self.dealer.hand.insurance_possible:
                    self.current_game.insure()
                else:
                    game_over = True  # End game if invalid insurance action
            case 4:  # Double Down
                if len(current_hand.cards) == 2 and self.player.balance >= current_hand.bet:
                    self.current_game.double_down(current_hand)
                    game_over = True
                else:
                    game_over = True  # End game if invalid double down action
            case _:
                raise ValueError(f"Invalid action: {action}")

        if current_hand.complete or game_over:
            return self.current_game.end()

        return self.current_game.get_state(current_hand), 0, False, False, result
    
    def get_valid_actions(self) -> List[int]:
        if not self.current_game or not self.player.hands:
            return []

        current_hand = self.player.get_active_hand()
        valid_actions = [0, 1]  # Stay and Hit are always valid

        if current_hand.split_possible:
            valid_actions.append(2)  # Split
        if self.dealer.hand.insurance_possible:
            valid_actions.append(3)  # Insure
        if len(current_hand.cards) == 2 and self.player.balance >= current_hand.bet:
            valid_actions.append(4)  # Double Down

        return valid_actions

    def get_game_state(self) -> Dict[str, Any]:
        if not self.current_game:
            return {"status": "No active game"}

        return {
            "player_hands": [hand.to_dict() for hand in self.player.hands],
            "dealer_hand": self.dealer.hand.to_dict(),
            "player_balance": self.player.balance,
            "valid_actions": self.get_valid_actions(),
            "deck_cards_left": len(self.deck.cards)
        }
