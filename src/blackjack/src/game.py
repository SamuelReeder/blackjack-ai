from .deck import Deck
from .entities import Player, Dealer
from .hand import Hand
from typing import List, Tuple, Dict, Any, Optional
from enum import Enum

class GameResult(Enum):
    PLAYER_WIN = 1
    DEALER_WIN = 2
    PUSH = 3
    PLAYER_BLACKJACK = 4
    DEALER_BLACKJACK = 5

class Game:

    def __init__(self, player: Player, dealer: Dealer, deck: Deck, bet: int) -> None:
        self.player = player
        self.dealer = dealer
        self.deck = deck
        self.bet = bet
        self.queue_shuffle = False
        
    def init_round(self) -> Tuple[List[Any], bool]:
        self.check_shuffle()

        self.player.reset()
        self.dealer.reset()

        self.player.hands[0].bet = self.bet
        self.player.change_balance(-self.bet)
        
        for _ in range(2):
            player_card = self.deal_card()
            dealer_card = self.deal_card()
            if player_card is None or dealer_card is None:
                raise ValueError("Not enough cards to start the game")
            self.player.hands[0].add_card(player_card)
            self.dealer.hand.add_card(dealer_card)

        player_has_blackjack, dealer_has_blackjack = self.check_for_blackjack()
        if player_has_blackjack:
            reward = self.show_blackjack_results(player_has_blackjack, dealer_has_blackjack, self.player.hands[0])
            self.player.change_balance(reward)
            return self.get_state(self.player.hands[0]), True, {"reward": reward - self.bet}
        
        return self.get_state(self.player.hands[0]), False, {}    
    
    def check_for_blackjack(self) -> Tuple[bool, bool]:
        return self.player.hands[0].calculate_value() == 21, self.dealer.hand.calculate_value() == 21
    
    def show_blackjack_results(self, player_has_blackjack: bool, dealer_has_blackjack: bool, hand: Hand) -> float:
        if player_has_blackjack and dealer_has_blackjack:
            return hand.bet
        return hand.bet * 2.5
    
    def hit(self, hand: Hand) -> None:
        card = self.deal_card()
        if card is None:
            raise ValueError("No more cards in the deck")
        hand.add_card(card)
        if self.is_bust(hand) or hand.calculate_value() == 21:
            hand.complete = True
            
    def stay(self, hand: Hand) -> None:
        hand.complete = True
        
    def insure(self) -> None:
        if self.dealer.hand.insurance_possible:
            insurance_amount = self.player.hands[0].bet / 2
            if self.player.balance >= insurance_amount:
                self.player.change_balance(-insurance_amount)
                self.player.insurance = True
                self.dealer.hand.insurance_possible = False

    def split(self, hand: Hand) -> Optional[Hand]:
        if hand.split_possible and self.player.balance >= hand.bet:
            new_hand = hand.split()
            self.player.change_balance(-hand.bet)  # Deduct the bet for the new hand
            hand.split_possible = False
            return new_hand
        return None

    def double_down(self, hand: Hand) -> None:
        if len(hand.cards) == 2 and self.player.balance >= hand.bet:
            self.player.change_balance(-hand.bet)  # Deduct the doubled amount
            hand.bet *= 2  # Double the bet
            self.hit(hand)  # Deal one final card
            hand.complete = True


    def dealer_play(self) -> None:
        while self.dealer.hand.calculate_value() <= 17:
            card = self.deal_card()
            while card is None:
                self.deck.reset()
                card = self.deal_card()
            self.dealer.hand.add_card(card)

    def end(self) -> Tuple[List[Any], int, bool, bool, Dict[str, Any]]:
        # self.check_insurance()
        # self.dealer_play()
        
        results = {}
        total_win_loss = 0
        self.dealer_play()
        
        for i, hand in enumerate(self.player.hands):
            result = self.determine_result(hand)
            payout = self.calculate_payout(hand, result)
            total_win_loss += payout - hand.bet
            self.player.change_balance(payout)
            results[f"hand_{i+1}"] = {"result": result.name, "payout": payout}
        
        return (self.get_state(self.player.hands[0]), 
                total_win_loss, 
                True,  # Round is over
                False,  # Game is not over
                results)

    def is_bust(self, hand: Hand) -> bool:
        return hand.calculate_value() > 21

    def check_insurance(self) -> None:
        if self.player.insurance:
            if self.dealer.hand.calculate_value() == 21:
                self.player.change_balance(self.player.hands[0].bet)
            self.player.insurance = False

    def check_shuffle(self) -> None:
        if self.queue_shuffle:  
            self.deck.reset()
            self.queue_shuffle = False
    
    def get_state(self, hand: Hand) -> List[Any]:
        self.dealer.hand.calculate_value(),
        return [
            hand.calculate_value(),
            self.dealer.hand.calculate_value(hide_dealer=True),
            self.deck.current_count,
            self.deck.face_tally,
            len(self.deck.cards),
            *self.deck.card_arr,
            # self.dealer.hand.insurance_possible,
            # hand.split_possible
        ]

    def deal_card(self) -> Optional[int]:
        card = self.deck.deal()
        if card == -1:
            self.queue_shuffle = True
            card = self.deck.deal()
        return card if card != -1 else None

    def determine_result(self, hand: Hand) -> GameResult:
        player_value = hand.calculate_value()
        dealer_value = self.dealer.hand.calculate_value()

        if self.is_bust(hand):
            return GameResult.DEALER_WIN
        elif self.is_bust(self.dealer.hand):
            return GameResult.PLAYER_WIN
        elif player_value > dealer_value:
            return GameResult.PLAYER_WIN
        elif dealer_value > player_value:
            return GameResult.DEALER_WIN
        else:
            return GameResult.PUSH

    def calculate_payout(self, hand: Hand, result: GameResult) -> int:
        if result == GameResult.PLAYER_WIN:
            return hand.bet * 2
        elif result == GameResult.PUSH:
            return hand.bet
        else:
            return 0
