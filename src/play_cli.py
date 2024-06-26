from blackjack import Manager


instance = Manager()
instance.new_game()

while True:
    
    action = input("Enter action: ")
    state, reward, done, truncated, _ = instance.play_game(int(action))
    if done or truncated:
        print("Game over!")
        instance.new_game()
        