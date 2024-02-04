from .card import Card
import random


class Deck:
    
    def __init__(self) -> None:
        self.cards = [Card(s, v) for s in ["Spades", "Clubs", "Hearts",
                      "Diamonds"] for v in ["A", "2", "3", "4", "5", "6", 
                      "7", "8", "9", "10", "J", "Q", "K"]]
        self.cards = self.cards * 6
        self.number_of_decks = 6
        self.original_length = len(self.cards)
        self.cards.append(Card(None, "Yellow"))
        self.true_count = 0
        self.current_count = 0
        self.face_tally = 0
                
    def shuffle(self) -> None:
        random.shuffle(self.cards)
        
    def reset(self) -> None:
        self.cards = [Card(s, v) for s in ["Spades", "Clubs", "Hearts",
                      "Diamonds"] for v in ["A", "2", "3", "4", "5", "6", 
                      "7", "8", "9", "10", "J", "Q", "K"]]
        self.cards = self.cards * 6
        self.number_of_decks = 6
        self.original_length = len(self.cards)
        self.true_count = 0
        self.current_count = 0
        self.face_tally = 0
        self.shuffle()
        self.cards.insert(random.randrange(len(self.cards) // 2 + 1, len(self.cards)), Card(None, "Yellow"))
        
        
    def update_count(self, card: Card) -> None:
        if card.rank in ["2", "3", "4", "5", "6"]:
            self.current_count += 1
            cards_left = self.number_of_decks - ((self.original_length - len(self.cards))) 
            if cards_left != 0:
                self.true_count = self.current_count / (cards_left / 52)
        elif card.rank in ["10", "J", "Q", "K", "A"]:
            self.current_count -= 1
            self.true_count -= 1


    def deal(self) -> Card:
        if len(self.cards) < 1:
            self.reset()
            
        if not self.cards[-1].rank.isnumeric():
            self.face_tally += 1

        
        card = self.cards.pop()
        self.update_count(card)
        return card