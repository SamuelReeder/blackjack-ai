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
        # self.current_game = Game(self.players, self.dealer, self.deck)
        # return self.current_game.init_round()
        self.action_interface.execute("start")
        print("Game started")
        self.action_interface.execute("bet")
        print("Bet placed")
        self.deal()
    
    def deal(self) -> None:
        print("Dealing")
        self.action_interface.execute("next")
        self.action_interface.execute("deal")
        print("Dealt")
        sleep(2)
        self.action_interface.scan_cards()
        self.get_cards()
    
    def get_cards(self):
        player = action.detect_document('cropped_img_one.png')
        self.yield_cards(player)
        print(player)
        dealer = action.detect_document('cropped_img_two.png')
        self.yield_cards(dealer)
        print(dealer)
        for card in player:
            print(card)
            print(type(card))
            self.player.hand.add_card(Card(card))
            self.deck.remove_card(Card(card))
            # check if player has blackjack
        for card in dealer:
            print(card)
            print(type(card))
            self.dealer.hand.add_card(Card(card))
            self.deck.remove_card(Card(card))

    
    def play_game(self, action: int) -> None:
        # wins, losses = 0, 0
        
        # hand_index = 0
        # curr_hand = self.players[0].hand[0]
        # for i, element in enumerate(self.players[0].hand):
        #     if not element.complete:
        #         curr_hand = element
        #         hand_index = i
                
        match action:
            case 0:
                self.current_game.stay(curr_hand)
            case 1:
                self.current_game.hit(curr_hand)
            case 2:
                self.current_game.split(curr_hand)
            case 3:
                self.current_game.insure()
        
        # if curr_hand.complete and hand_index == len(self.players[0].hand) - 1:
        #     self.current_game.dealer_play()
        #     return self.current_game.end()
        
        # redundant = -10 if (not curr_hand.split_possible and action == 3) or (not curr_hand.insurance_possible and action == 2) else 0
        
        return self.get_state(curr_hand), False, False, False, {}
    
    def hit(self, hand: Hand) -> None:
        self.action_interface.execute("hit")
        sleep(1)
        action_interface.scan_cards()
        self.get_cards()
            
    def yield_cards(self, cards):
        for i in range(len(cards)):
            try :
                cards[i] = int(cards[i])
            except:
                if cards[i] == 'A':
                    cards[i] = 11
                else:
                    cards[i] = 10
        
    
    def stay(self, hand: Hand) -> None:
        self.action_interface.execute("stand")
        hand.complete = True
        # then do next game
        
    def get_state(self) -> tuple:
        # print("Hand value:", hand.get_value(), "Dealer's hand value:", self.dealer.hand.get_value(), "Current count:", self.deck.current_count, "Face tally:", self.deck.face_tally, "Deck length:", len(self.deck.cards), "Insurance possible:", self.dealer.hand.insurance_possible, "Split possible:", hand.split_possible)
        return [self.hand.get_value(), self.dealer.hand.get_value(one_card=True), self.deck.current_count, self.deck.face_tally, len(self.deck.cards), self.deck.card_arr] # , self.dealer.hand.insurance_possible, hand.split_possible
        
            
    
    
        
    