from blackjack import Manager


instance = Manager()
instance.new_game()

while True:
    
    action = input("Enter action: ")
    state, reward, done, truncated, _ = instance.play_game(int(action))
    if done or truncated:
        print("Game over! User recieved reward of:", reward)
        print("Balance:", instance.player.balance)
        instance.new_game()
        