import time
import random
from blackjack import Manager

def ai_action(valid_actions):
    return random.choice(valid_actions)

def human_action(valid_actions):
    while True:
        action = input(f"Choose action {valid_actions}: ")
        try:
            action = int(action)
            if action in valid_actions:
                return action
            else:
                print(f"Invalid action. Please choose one of {valid_actions}.")
        except ValueError:
            print("Please enter a valid number.")

def format_hand(cards):
    """Utility function to format the hand as a readable string."""
    return ', '.join(str(card) for card in cards)

def place_bet():
    """Prompts the player to place a bet and validates the input."""
    while True:
        try:
            bet = int(input("Enter your bet amount: "))
            if bet > 0:
                return bet
            else:
                print("Bet amount must be positive.")
        except ValueError:
            print("Please enter a valid number.")

def human_action(valid_actions):
    """Prompts the human player to choose an action and ensures it's valid."""
    while True:
        try:
            print(f"Choose action: {', '.join([f'{action_name} ({action})' for action, action_name in zip(valid_actions, ['Stay', 'Hit', 'Split', 'Insure', 'Double Down']) if action in valid_actions])}")
            action = int(input("Enter action number: "))
            if action in valid_actions:
                return action
            else:
                print(f"Invalid action. Choose from: {valid_actions}")
        except ValueError:
            print("Please enter a valid number.")

def play_game(instance, use_ai=False):
    # Place the bet, balance adjustments happen in the Game class
    bet = place_bet()

    # Start the game, bet deduction occurs inside `new_game`
    state, game_over = instance.new_game(bet)

    print("\n=== Game Start ===")
    
    while True:
        player_hand_value = state[0]
        player_cards = instance.player.hands[0].cards
        dealer_visible_card = instance.dealer.hand.cards[0]

        print(f"Your Hand: {format_hand(player_cards)} (Value: {player_hand_value})")
        print(f"Dealer's Visible Card: {dealer_visible_card}\n")
        
        if game_over:
            print("Blackjack!")
            print(f"You Win! Your reward: {bet * 2.5}")
            print(f"Final Player Balance: {instance.player.balance}")
            break

        valid_actions = instance.get_valid_actions()

        if use_ai:
            action = ai_action(valid_actions)
            print(f"AI chose: {action}")
        else:
            action = human_action(valid_actions)

        state, reward, done, truncated, info = instance.play_game(action)

        if done or truncated:
            print("\n--- Dealer's Turn ---")
            print(f"Dealer reveals their hidden card: {instance.dealer.hand.cards[1]}")

            print(f"Your Final Hand: {format_hand(instance.player.hands[0].cards)} (Value: {state[0]})")
            print(f"Dealer's Final Hand: {format_hand(instance.dealer.hand.cards)} (Value: {instance.dealer.hand.calculate_value()})")

            if reward > bet:
                print(f"\nYou Win! Your reward: {reward}")
            elif reward == bet:
                print(f"\nPush! No money lost.")
            else:
                print(f"\nYou Lose! Your loss: {bet}")

            print(f"Final Player Balance: {instance.player.balance}")
            input("Press Enter to start the next game... ")
            break



def main():
    instance = Manager()

    # Show player's starting balance
    print(f"Starting Balance: {instance.player.balance}")

    # Ask once whether it's human or AI
    while True:
        mode = input("Choose mode (1 for Human, 2 for AI): ")
        if mode == '1':
            use_ai = False
            break
        elif mode == '2':
            use_ai = True
            break
        else:
            print("Invalid choice. Please enter 1 for Human or 2 for AI.")

    # Loop over games, keeping balance between games
    while True:
        play_game(instance, use_ai)

if __name__ == "__main__":
    main()