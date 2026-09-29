from blackjack import BlackjackGame

if __name__ == "__main__":
    game = BlackjackGame("models/weights/best.pt")
    
    game.run()