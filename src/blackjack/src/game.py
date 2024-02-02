from .deck import Deck
from .entities import Player, Dealer
from .hand import Hand
from typing import List


# TODO: hit for dealer
#
class Game:
    def __init__(self, players: int = 1):
        self.players: List[Player] = []
        self.players.append(Player())
        self.dealer_hand = None
        self.queue_shuffle = True
        self.deck = Deck()

    def init_players(self, num_players) -> None:
        for _ in range(num_players):
            self.players.append(Player(10))

    def play(self) -> None:
        playing = True
        while playing:
            self.play_round(0)

            again = input("Play Again? [Y/N] ")
            while again.lower() not in ["y", "n"]:
                again = input("Please enter Y or N ")
            if again.lower() == "n":
                print("Thanks for playing!")
                playing = False
            else:
                game_over = False
                
    def init_game(self) -> None:
        self.deck = Deck()
        self.deck.shuffle()

    def init_round(self) -> tuple:
        if self.queue_shuffle:  
            self.deck.shuffle()
            self.queue_shuffle = False

        self.players[0].bet = 100
        self.players[0].change_balance(-self.players[0].bet)

        # for i in range(0, len(self.players)):
        self.players[0].hand[0] = Hand()

        self.dealer_hand = Hand(dealer=True)

        # for i in range(len(self.players)):
        #     for j in range(2):
        #         self.players[0].hand.add_card(self.deck.deal())
        # self.dealer_hand.add_card(self.deck.deal())
        for i in range(2):
            card0 = self.deck.deal()
            if card0.rank == "Yellow":
                self.queue_shuffle = True
                card0 = self.deck.deal()
            card1 = self.deck.deal()

            self.players[0].hand[0].add_card(card0)
            self.dealer_hand.add_card(card1)
        

        print("Your hand[0] is:")
        self.players[0].hand[0].display()
        print()
        print("Dealer's hand is:")
        self.dealer_hand.display()
        
        player_has_blackjack, dealer_has_blackjack = self.check_for_blackjack()
        if player_has_blackjack:
            self.show_blackjack_results(player_has_blackjack, dealer_has_blackjack)
            return (self.get_state(), True)
        
        if self.dealer_hand.insurance_possible:
            return (self.get_state(), False)
        
        if self.players[0].hand[0].split_possible:
            return (self.get_state(), False)
        
        return (self.get_state(), False)
        
    # 0: stay 1: hit, 2: split, 3: insure 
    def play_round(self, action: int) -> tuple:
        print(action) 
        
        if self.queue_shuffle:
            self.deck.reset()
            self.queue_shuffle = False
        
        if self.players[0].split:
            play_split(action)
            
        if self.dealer_hand.insurance_possible:
            if action == 3:
                self.players[0].change_balance(-self.players[0].bet / 2)
                self.players[0].insurance = True
                self.dealer_hand.insurance_possible = False
                return (self.get_state(), False)
        
        if self.players[0].hand.split_possible:
            if action == 2:
                self.players[0].new_hand = self.players[0].hand[0].split()
                self.players[0].hand[0].display()
                self.players[0].new_hand.display()
                self.players[0].split = True
                self.players[0].change_balance(-self.players[0].bet)
                return (self.get_state(), False)
        
        player_has_blackjack, dealer_has_blackjack = self.check_for_blackjack()
        if player_has_blackjack:
            self.show_blackjack_results(player_has_blackjack, dealer_has_blackjack)
            return (self.get_state(), 1, True, False, {})

        if action == 1:
            self.players[0].hand.add_card(self.deck.deal())
            self.players[0].hand[0].display()
            if self.player_is_over():
                print("You have lost!")
                return (self.get_state(), -1, True, False, {})
            return (self.get_state(), 0, False, False, {})


        else:
            while self.dealer_hand.get_value() < 17:
                self.dealer_hand.add_card(self.deck.deal())
            print("Dealer's Hand:")
            self.dealer_hand.display(hide=False)

            player_hand_value = self.players[0].hand.get_value()
            dealer_hand_value = self.dealer_hand[0].get_value()
            
            if self.players[0].insurance:
                if dealer_hand_value == 21:
                    print("Dealer has blackjack! You get your insurance!")
                    self.players[0].change_balance(self.players[0].bet)
                else:
                    print("Dealer does not have blackjack! You lose your insurance!")

            player_blackjack, dealer_blackjack = self.check_for_blackjack()
            if player_blackjack and dealer_blackjack:
                game_over = True
                self.show_blackjack_results(
                    player_has_blackjack, dealer_has_blackjack
                )
                return (self.get_state(), 0, True, False, {}) 

            print("Final Results")
            print("Your hand:", player_hand_value)
            print("Dealer's hand:", dealer_hand_value)

            if self.dealer_hand.get_value() > 21:
                print("Dealer busts! You win!")
                self.players[0].change_balance(self.players[0].bet * 2)
                return (self.get_state(), 1, True, False, {}) 
            elif player_hand_value > dealer_hand_value:
                print("You Win!")
                self.players[0].change_balance(self.players[0].bet * 2)
                return (self.get_state(), 1, True, False, {}) 
            elif player_hand_value == dealer_hand_value:
                print("Tie!")
                self.players[0].change_balance(self.players[0].bet)
                return (self.get_state(), 0, True, False, {}) 
            else:
                print("Dealer Wins!")
                return (self.get_state(), -1, True, False, {}) 

    def play_split(self, action: int) -> tuple:
        

    def player_is_over(self):
        return self.players[0].hand[0].get_value() > 21

    def check_for_blackjack(self):
        player = False
        dealer = False
        if self.players[0].hand[0].get_value() == 21:
            player = True
        if self.dealer_hand[0].get_value() == 21:
            dealer = True

        return player, dealer

    def show_blackjack_results(self, player_has_blackjack, dealer_has_blackjack):
        if player_has_blackjack and dealer_has_blackjack:
            print("Both players have blackjack! Draw!")
            self.players[0].change_balance(self.players[0].bet)
            if self.players[0].insurance:
                self.players[0].change_balance(self.players[0].bet)

        elif player_has_blackjack:
            print("You have blackjack! You win!")
            self.players[0].change_balance(self.players[0].bet * 2)

        elif dealer_has_blackjack:
            print("Dealer has blackjack! Dealer wins!")
            if self.players[0].insurance:
                self.players[0].change_balance(self.players[0].bet)
                
    def reset(self):
        pass
    
    
    def get_state(self):
        return (self.players[0].hand[0].get_value(), self.dealer_hand[0].get_value(), self.deck.current_count, self.deck.face_tally, len(self.deck.cards), self.dealer_hand.insurance_possible, self.players[0].hand[0].split_possible)
