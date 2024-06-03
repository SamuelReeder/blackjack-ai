from .deck import Deck
from .entities import Player, Dealer
from .hand import Hand
from typing import List


class Game:
    def __init__(self, player: Player, dealer: Dealer, deck: Deck, debug: bool = False) -> None :
        self.player = player
        self.dealer = dealer
        self.deck = deck
        self.queue_shuffle = False
        self.debug = debug
        
    def hit(self, hand: Hand) -> None:
        hand.add_card(self.deck.deal())
        hand.display()
        if self.player_is_over(hand):
            hand.complete = True
        elif hand.calculate_value() == 21:
            hand.complete = True        
            
    def stay(self, hand: Hand) -> None:
        hand.complete = True
        
    def insure(self) -> None:
         if self.dealer.hand.insurance_possible:
            if self.debug: 
                print("Insuring!")
            self.player.change_balance(-self.player.hand.bet / 2)
            self.player.insurance = True
            self.dealer.hand.insurance_possible = False
            
    def split(self, hand: Hand) -> None:
        if hand.split_possible:
            if self.debug:
                print("Splitting!")
            self.player.hand.append(hand.split())
            hand.display()
            self.player.hand.display()
            self.player.hand.bet = 100
            self.player.change_balance(-self.player.hand.bet)
            hand.split_possible = False    

    def init_round(self) -> tuple:
        self.initial_balance = self.player.balance
        self.check_shuffle()

        self.player.hand = Hand()
        self.dealer.hand = Hand(dealer=True)

        self.player.hand.bet = 100
        self.player.change_balance(-self.player.hand.bet)
        
        for i in range(2):
            card0 = self.deck.deal()
            if card0 == -1:
                self.queue_shuffle = True
                card0 = self.deck.deal()
            card1 = self.deck.deal()

            self.player.hand.add_card(card0)
            self.dealer.hand.add_card(card1)
        self.player.hand.display()
        self.dealer.hand.display(hide=True)
        
        player_has_blackjack, dealer_has_blackjack = self.check_for_blackjack()
        if player_has_blackjack:
            self.show_blackjack_results(player_has_blackjack, dealer_has_blackjack, self.player.hand)
            return (self.get_state(self.player.hand), True)
        
        return (self.get_state(self.player.hand), False)
        
    def dealer_play(self) -> tuple:
        while self.dealer.hand.calculate_value() < 17:
            self.dealer.hand.add_card(self.deck.deal())
            self.dealer.hand.display(hide=True)

    def end(self) -> tuple:
        self.check_insurance()
        
        dealer_hand_value = self.dealer.hand.calculate_value()
        player_hand_value = self.player.hand.calculate_value()
        
        player_blackjack, dealer_blackjack = self.check_for_blackjack()
        if player_blackjack and dealer_blackjack:
            self.show_blackjack_results(
                player_blackjack, dealer_blackjack, self.player.hand
            )
            self.player.change_balance(self.player.hand.bet)
                
        return (self.get_state(self.player.hand), self.player.balance - self.initial_balance , True, False, {})

    def player_is_over(self, hand: Hand) -> bool:
        return hand.calculate_value() > 21

    def check_for_blackjack(self) -> tuple:
        return self.player.hand.calculate_value() == 21, self.dealer.hand.calculate_value() == 21
    
    def check_insurance(self) -> None:
        if self.player.insurance:
            if self.dealer.hand.calculate_value() == 21:
                if self.debug:
                    print("Dealer has blackjack! You get your insurance!")
                self.player.change_balance(self.player.hand.bet)
            else:
                if self.debug:                    
                    print("Dealer does not have blackjack! You lose your insurance!")
                self.player.insurance = False
                    

    def show_blackjack_results(self, player_has_blackjack: bool, dealer_has_blackjack: bool, hand: Hand) -> None:
        if player_has_blackjack and dealer_has_blackjack:
            self.player.change_balance(hand.bet)
            if self.player.insurance:
                self.player.change_balance(hand.bet)
        elif player_has_blackjack:
            self.player.change_balance(hand.bet * 2.5)
        elif dealer_has_blackjack:
            if self.player.insurance:
                self.player.change_balance(hand.bet)
    
    def check_shuffle(self) -> None:
        if self.queue_shuffle:  
            self.deck.reset()
            self.queue_shuffle = False
    
    def get_state(self, hand: Hand, hide_dealer:bool = True) -> tuple:
        print("Hand value:", hand.calculate_value(), "Dealer's hand value:", self.dealer.hand.calculate_value(hide_dealer==hide_dealer), "Current count:", self.deck.current_count, "Face tally:", self.deck.face_tally, "Deck length:", len(self.deck.cards), "Insurance possible:", self.dealer.hand.insurance_possible, "Split possible:", hand.split_possible)
        return [hand.calculate_value(), self.dealer.hand.calculate_value(hide_dealer=hide_dealer), self.deck.current_count, self.deck.face_tally, len(self.deck.cards), self.deck.card_arr] # , self.dealer.hand.insurance_possible, hand.split_possible