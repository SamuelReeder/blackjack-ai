from typing import List
from blackjack import Hand, Deck, Player, Dealer, Card
import action
from time import sleep 


class OnlineManager:
    
    def __init__(self) -> None:
        self.deck = Deck()
        self.player = Player()
        self.dealer = Dealer()
        self.balance: List[float] = []
        self.game_over = False
        self.game_on = False
        self.action_interface = action.ActionInterface("web-metadata\\actions.json")
    
    def new_game(self) -> None:
        self.player.hand.reset()
        self.dealer.hand.reset()
        self.action_interface.execute("start")
        self.action_interface.execute("bet")
    
    def deal(self) -> None:
        self.player.hand.reset()
        self.dealer.hand.reset()
        self.action_interface.execute("next")
        self.action_interface.execute("deal")
        sleep(2)
        self.action_interface.scan_cards()
        self.get_cards()
    
    def get_cards(self):
        player = action.detect_document('cropped_img_one.png')
        self.yield_cards(player)
        dealer = action.detect_document('cropped_img_two.png')
        self.yield_cards(dealer)
        for card in player:
            count = player.count(card) - self.player.hand.cards.count(card)
            if count > 0:
                self.player.hand.add_card(card)
                self.deck.remove_card(card)
        if self.player.hand.get_value() == 21:
            sleep(1)
            return self.get_state(), True, True, False, {}
        for card in dealer:
            count = dealer.count(card) - self.dealer.hand.cards.count(card)
            if count > 0:
                self.dealer.hand.add_card(card)
                self.deck.remove_card(card)
                
        print("Player hand:", [i for i in self.player.hand.cards], "Dealer's hand:", [i for i in self.dealer.hand.cards])

    
    def play_game(self, action: int) -> None:
                
        match action:
            case 0:
                self.stay()
                sleep(3)
                return self.get_state(), False, True, False, {}
            case 1:
                self.hit()
                value = self.player.hand.get_value()
                dealer_value = self.dealer.hand.get_value()
                print("Player hand value:", value, "Dealer's hand value:", dealer_value)
                if value > 21:
                    sleep(1)
                    return self.get_state(), False, True, False, {}
                elif value == 21:
                    sleep(1)
                    return self.get_state(), True, True, False, {}
                elif dealer_value == 21:
                    return self.get_state(), False, True, False, {}
                
        return self.get_state(), False, False, False, {}
    
    def hit(self) -> None:
        self.action_interface.execute("hit")
        self.action_interface.scan_cards()
        self.get_cards()
            
    def yield_cards(self, cards):
        for i in range(len(cards)):
            try :
                cards[i] = int(cards[i])
                if cards[i] == 0:
                    cards[i] = 10
            except:
                if cards[i] == 'A':
                    cards[i] = 1
                else:
                    cards[i] = 10
        
    
    def stay(self) -> None:
        self.action_interface.execute("stand")
        # then do next game
        
    def get_state(self) -> tuple:
        # print("Hand value:", hand.get_value(), "Dealer's hand value:", self.dealer.hand.get_value(), "Current count:", self.deck.current_count, "Face tally:", self.deck.face_tally, "Deck length:", len(self.deck.cards), "Insurance possible:", self.dealer.hand.insurance_possible, "Split possible:", hand.split_possible)
        return [self.player.hand.get_value(), self.dealer.hand.get_value(one_card=True), self.deck.current_count, self.deck.face_tally, len(self.deck.cards), self.deck.card_arr] # , self.dealer.hand.insurance_possible, hand.split_possible
        
            
    
    
        
    