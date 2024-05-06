from .deck import Deck
from .entities import Player, Dealer
from .hand import Hand
from typing import List


class Game:
    def __init__(self, players: List[Player], dealer: Dealer, deck: Deck, debug: bool = False) -> None :
        self.players: List[Player] = players
        self.dealer = dealer
        self.deck = deck
        self.queue_shuffle = False
        self.debug = debug
        
    def hit(self, hand: Hand) -> None:
        hand.add_card(self.deck.deal())
        hand.display()
        if self.player_is_over(hand):
            if self.debug:
                print("You have lost!")
            hand.complete = True
            
    def stay(self, hand: Hand) -> None:
        hand.complete = True
        
    def insure(self) -> None:
         if self.dealer.hand.insurance_possible:
            if self.debug: 
                print("Insuring!")
            self.players[0].change_balance(-self.players[0].hand[0].bet / 2)
            self.players[0].insurance = True
            self.dealer.hand.insurance_possible = False
            
    def split(self, hand: Hand) -> None:
        if hand.split_possible:
            if self.debug:
                print("Splitting!")
            self.players[0].hand.append(hand.split())
            hand.display()
            self.players[0].hand[-1].display()
            self.players[0].hand[-1].bet = 100
            self.players[0].change_balance(-self.players[0].hand[-1].bet)
            hand.split_possible = False    

    def init_round(self) -> tuple:
        self.initial_balance = self.players[0].balance
        self.check_shuffle()

        self.players[0].hand[0] = Hand()
        self.dealer.hand = Hand(dealer=True)

        self.players[0].hand[0].bet = 100
        self.players[0].change_balance(-self.players[0].hand[0].bet)
        
        for i in range(2):
            card0 = self.deck.deal()
            if card0.rank == -1:
                self.queue_shuffle = True
                card0 = self.deck.deal()
            card1 = self.deck.deal()

            self.players[0].hand[0].add_card(card0)
            self.dealer.hand.add_card(card1)
        if self.debug:
            print("Your hand[0] is:")
        self.players[0].hand[0].display()
        if self.debug:
            print("Dealer's hand is:")
        self.dealer.hand.display()
        
        player_has_blackjack, dealer_has_blackjack = self.check_for_blackjack(0)
        if player_has_blackjack:
            self.show_blackjack_results(player_has_blackjack, dealer_has_blackjack, self.players[0].hand[0])
            return (self.get_state(self.players[0].hand[0]), True)
        
        return (self.get_state(self.players[0].hand[0]), False)
        
    def dealer_play(self) -> tuple:
        while self.dealer.hand.get_value() < 17:
            self.dealer.hand.add_card(self.deck.deal())
            if self.debug:
                print("Dealer's Hand:")
            self.dealer.hand.display(hide=False)

    def end(self) -> tuple:
        
        self.check_insurance()
        dealer_hand_value = self.dealer.hand.get_value()
        for i, element in enumerate(self.players[0].hand):
            player_hand_value = element.get_value()
           
            player_blackjack, dealer_blackjack = self.check_for_blackjack(i)
            if player_blackjack and dealer_blackjack:
                self.show_blackjack_results(
                    player_blackjack, dealer_blackjack, element
                )
                self.players[0].change_balance(element.bet)
                continue
                
            if self.player_is_over(element):
                if self.debug:
                    print("You have lost!")
                continue
            if dealer_hand_value > 21:
                if self.debug:
                    print("Dealer busts! You win!")
                self.players[0].change_balance(element.bet * 2)
            elif player_hand_value > dealer_hand_value:
                if self.debug:
                    print("You Win!")
                self.players[0].change_balance(element.bet * 2)
            elif player_hand_value == dealer_hand_value:
                if self.debug:
                    print("Tie!")
                self.players[0].change_balance(element.bet)
            else:
                if self.debug:
                    print("Dealer Wins!")
                
        return (self.get_state(self.players[0].hand[0]), self.players[0].balance - self.initial_balance , True, False, {})

    def player_is_over(self, hand: Hand) -> bool:
        return hand.get_value() > 21

    def check_for_blackjack(self, index: int) -> tuple:
        player = False
        dealer = False
        if self.players[0].hand[index].get_value() == 21:
            player = True
        if self.dealer.hand.get_value() == 21:
            dealer = True
        return player, dealer
    
    def check_insurance(self) -> None:
        if self.players[0].insurance:
            if self.dealer.hand.get_value() == 21:
                if self.debug:
                    print("Dealer has blackjack! You get your insurance!")
                self.players[0].change_balance(self.players[0].hand[0].bet)
            else:
                if self.debug:                    
                    print("Dealer does not have blackjack! You lose your insurance!")
                self.players[0].insurance = False
                    

    def show_blackjack_results(self, player_has_blackjack: bool, dealer_has_blackjack: bool, hand: Hand) -> None:
        if player_has_blackjack and dealer_has_blackjack:
            if self.debug:
                print("Draw!")
            self.players[0].change_balance(hand.bet)
            if self.players[0].insurance:
                self.players[0].change_balance(hand.bet)
        elif player_has_blackjack:
            if self.debug:
                print("You win!")
            self.players[0].change_balance(hand.bet * 2.5)
        elif dealer_has_blackjack:
            if self.debug:
                print("Dealer wins!")
            if self.players[0].insurance:
                self.players[0].change_balance(hand.bet)
    
    def check_shuffle(self) -> None:
        if self.queue_shuffle:  
            self.deck.reset()
            self.queue_shuffle = False
    
    def get_state(self, hand: Hand) -> tuple:
        # print("Hand value:", hand.get_value(), "Dealer's hand value:", self.dealer.hand.get_value(), "Current count:", self.deck.current_count, "Face tally:", self.deck.face_tally, "Deck length:", len(self.deck.cards), "Insurance possible:", self.dealer.hand.insurance_possible, "Split possible:", hand.split_possible)
        return [hand.get_value(), self.dealer.hand.get_value(one_card=True), self.deck.current_count, self.deck.face_tally, len(self.deck.cards), self.deck.card_arr] # , self.dealer.hand.insurance_possible, hand.split_possible