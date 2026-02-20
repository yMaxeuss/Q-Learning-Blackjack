import random
import os
import numpy as np
import matplotlib.pyplot as plt

# Toggle card counting feature
USE_CARD_COUNTING = False

class Card:
    faces = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
    houses = ["♥", "♦", "♣", "♠"]

    def __init__(self, face, house):
        self.face = face
        self.house = house
        
        if face in ["J", "Q", "K"]:
            self.value = 10
        elif face == "A":
            self.value = 11  # Aces default to 11, adjusted later if needed
        else:
            self.value = int(face)

    def __repr__(self):
        return f"{self.face}{self.house}"

class Deck:
    def __init__(self, deck_count=6):
        self.original_count = deck_count
        self.deck = []
        self.running_count = 0
        self.reset_deck()

    def __len__(self):
        return len(self.deck)

    def initial_cards(self):
        for _ in range(self.original_count):
            for house in Card.houses:
                for face in Card.faces:
                    self.deck.append(Card(face, house))

    def get_card_value(self, card):
        # Card Counting System Logic
        if card.value >= 2 and card.value <= 6:
            return 1
        elif card.value >= 10 or card.face == "A":
            return -1
        else:
            return 0
        
    def deal_card(self):
        if len(self.deck) == 0:
            self.reset_deck()
        card = self.deck.pop()
      
        self.running_count += self.get_card_value(card)     
        return card 

    def get_true_count(self):
        decks_remaining = len(self.deck) / 52
        if decks_remaining < 1: 
            decks_remaining = 1
            
        return round(self.running_count / decks_remaining)

    def shuffle_deck(self):
        random.shuffle(self.deck)

    def reset_deck(self):
        self.deck = []
        self.initial_cards()
        self.shuffle_deck()
        self.running_count = 0

class Hand:
    def __init__(self):
        self.hand_cards = []
        self.hand_value = 0
        self.has_usable_ace = False

    def add_card(self, card):
        if card:
            self.hand_cards.append(card)
            self.calculate_value()

    def calculate_value(self):
        self.hand_value = 0
        ace_count = 0

        for card in self.hand_cards:
            self.hand_value += card.value
            if card.face == "A":
                ace_count += 1

        self.has_usable_ace = False
        if ace_count > 0 and self.hand_value <= 21:
            self.has_usable_ace = True

        while self.hand_value > 21 and ace_count > 0:
            self.hand_value -= 10
            ace_count -= 1
            if ace_count == 0:
                self.has_usable_ace = False

        return self.hand_value

class BlackjackEnv:
    def __init__(self):
        self.deck = Deck(deck_count=6)
        self.player_hand = None
        self.dealer_hand = None

    def get_count_bucket(self):
        tc = self.deck.get_true_count()
        
        if tc <= 1: return 0     
        elif tc == 2: return 1    
        elif tc == 3: return 2   
        elif tc == 4: return 3 
        else: return 4 

    def get_state(self):
        # The AI only "sees" these three(four) numbers: (Player Score, Dealer Up-Card, Usable Ace, (Count Bucket))
        p_score = self.player_hand.hand_value
        d_score = self.dealer_hand.hand_cards[0].value # Only look at the first dealer card!
        has_ace = 1 if self.player_hand.has_usable_ace else 0

        if USE_CARD_COUNTING:
            count_bucket = self.get_count_bucket()
            return (p_score, d_score, has_ace, count_bucket)
        else:
            return (p_score, d_score, has_ace)

    def reset(self):
        if len(self.deck) < 52:
            self.deck.reset_deck()

        self.player_hand = Hand()
        self.dealer_hand = Hand()

        for _ in range(2):
            self.player_hand.add_card(self.deck.deal_card())
            self.dealer_hand.add_card(self.deck.deal_card())

        return self.get_state()

    def step(self, action):
        # Executes the AI's chosen move and returns: (Next State, Reward, Is_Done)
        # Actions: 1 = Hit, 0 = Stay
        
        p_val = self.player_hand.hand_value
        
        if action == 1:
            self.player_hand.add_card(self.deck.deal_card())
            new_p_val = self.player_hand.hand_value
            
            if new_p_val > 21:
                return self.get_state(), -1, True
            elif new_p_val == 21:
                return self.get_state(), 1, True
                
            return self.get_state(), 0, False      

        else:
            # The Dealer plays out their hand using standard blakjack rules (Hit until 17)
            while self.dealer_hand.hand_value < 17:
                self.dealer_hand.add_card(self.deck.deal_card())

            p_val = self.player_hand.hand_value
            d_val = self.dealer_hand.hand_value

            if d_val > 21:
                return self.get_state(), 1, True 
            elif p_val > d_val:
                return self.get_state(), 1, True 
            elif p_val == d_val:
                return self.get_state(), 0, True 
            else:
                return self.get_state(), -1, True 

def train_ai(env, episodes=50000):
    print(f"Training AI for {format_num(episodes)} episodes...")
    
    # Initialize the "Brain" (Q-Table) with zeros
    if USE_CARD_COUNTING:
        q_table = np.zeros((32, 12, 2, 5, 2)) # Dimensions: Player Score (32), Dealer Card (12), Usable Ace (2), Count Bucket(5), Actions (2)
    else:
        q_table = np.zeros((32, 12, 2, 2)) # Dimensions: Player Score (32), Dealer Card (12), Usable Ace (2), Actions (2)
    
    # Hyperparameters
    alpha = 0.1             # Learning Rate: How quickly it overrides old info
    gamma = 0.95            # Discount Factor: How much it cares about future rewards
    epsilon = 1.0           # Exploration Rate: Starts at 100% random choices
    min_epsilon = 0.01      # Never drops below 1% randomness

    # Gradually reduces randomness (We want epsilon to reach 0.01 when training is 80% complete)
    epsilon_decay = (min_epsilon / epsilon) ** (1 / (episodes * 0.8)) 

    for i in range(episodes):
        state = env.reset()
        done = False

        while not done:
            # 1. Choose Action (Epsilon-Greedy Strategy)
            if random.uniform(0, 1) < epsilon:
                action = random.choice([0, 1]) # Explore: Pick randomly
            else:
                action = np.argmax(q_table[state]) # Exploit: Use the best known move

            # 2. Execute Action in the Environment
            next_state, reward, done = env.step(action)

            # 3. Update the Q-Table using the Bellman Equation
            state_action = state + (action,)
            old_value = q_table[state_action]

            if done:
                next_max = 0 # No future value if the game is over
            else:
                next_max = np.max(q_table[next_state]) # Best possible future score

            # The Core Math: Update expected value based on the reward received
            new_value = old_value + alpha * (reward + (gamma * next_max) - old_value)
            q_table[state_action] = new_value
            
            state = next_state

        # At the end of every game, slightly reduce the chance of making random moves
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

    print("Training Complete!")
    return q_table

def test_ai(env, q_table, game_count=1000):
    print(f"\n--- Testing AI over {format_num(game_count)} games ---")
    wins, losses, draws = 0, 0, 0

    starting_bankroll = 10 * game_count 
    bankroll = starting_bankroll
    bankroll_history = [starting_bankroll]

    for _ in range(game_count):
        state = env.reset()
        done = False

        if USE_CARD_COUNTING:
            count_bucket = state[3] 
            
            if count_bucket == 4:      
                bet = 100
            elif count_bucket == 3:    
                bet = 50
            else:                      
                bet = 10
        else:
            bet = 10

        while not done:
            action = np.argmax(q_table[state])
            
            state, reward, done = env.step(action)

            if done:
                if reward == 1:
                    wins += 1
                    if state[0] == 21 and len(env.player_hand.hand_cards) == 2:
                        bankroll += (bet * 1.5)
                    else:
                        bankroll += bet
                elif reward == -1:
                    losses += 1
                    bankroll -= bet
                else:
                    draws += 1

        bankroll_history.append(bankroll)

    print(f"Wins:   {wins}")
    print(f"Losses: {losses}")
    print(f"Draws:  {draws}")
    
    win_rate = wins / (wins + losses) * 100
    print(f"Win Rate (excluding draws): {win_rate:.2f}%")
    print(f"Win Rate (with draws): {wins / game_count * 100:.2f}%\n")

    # Money Output
    print("-" * 30)
    print(f"Starting Bankroll: ${format_num(starting_bankroll)}")
    print(f"Final Bankroll:    ${format_num(bankroll)}")
    profit = bankroll - starting_bankroll
    
    if profit > 0:
        print(f"Profit: +${format_num(profit)}")
    else:
        print(f"Loss:   -${format_num(abs(profit))}")
    print("-" * 30, "\n")

    plt.figure(figsize=(10, 6))
    
    line_color = "green" if bankroll >= starting_bankroll else "red"
    plt.plot(bankroll_history, label="AI Bankroll", color=line_color, linewidth=1.5)
    
    plt.axhline(y=starting_bankroll, color='gray', linestyle='--', label="Starting Bankroll")
    
    strategy_name = "Card Counting" if USE_CARD_COUNTING else "Basic Strategy"
    plt.title(f"Blackjack AI Performance ({strategy_name})")
    plt.xlabel("Hands Played")
    plt.ylabel("Bankroll ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.show()

def check_strategy(q_table, p_sum, d_sum, have_a=0, c_bucket=2):
    """Query the Q-Table for specific strategic decisions."""
    
    if not (4 <= p_sum <= 21):
        raise ValueError(f"Impossible Player Score: {p_sum}. Must be 4-21.")
    if not (2 <= d_sum <= 11):
        raise ValueError(f"Impossible Dealer Card: {d_sum}. Must be 2-11.")
    if have_a not in [0, 1]:
        raise ValueError("Usable Ace must be 0 (False) or 1 (True).")
    if have_a == 1 and p_sum < 12:
        raise ValueError(f"Impossible Soft Hand: {p_sum}. A hand with a usable Ace must be at least 12.")
    
    if USE_CARD_COUNTING:
        state = (p_sum, d_sum, have_a, c_bucket)
        print(f"--- AI Strategy: You have {p_sum}, Dealer shows {d_sum} (Count Bucket {c_bucket}) ---")
    else:
        state = (p_sum, d_sum, have_a)
        print(f"--- AI Strategy: You have {p_sum}, Dealer shows {d_sum} ---")
    
    values = q_table[state] 
    print(f"Expected Value of Staying: {values[0]:.4f}")
    print(f"Expected Value of Hitting: {values[1]:.4f}")
    
    if values[0] > values[1]:
        print(">> AI Decision: STAY\n")
    else:
        print(">> AI Decision: HIT\n")

def format_num(num):
    if num >= 1_000_000:
        return f"{num / 1_000_000:g}M"
    elif num >= 1_000:
        return f"{num / 1_000:g}K"
    return str(num)


def main():

    env = BlackjackEnv()

    if USE_CARD_COUNTING:
        model_file = "q_table_counting.npy"
    else:
        model_file = "q_table_basic.npy"
    
    load_model = False
    if os.path.exists(model_file):
        choice = input(f"\nFound existing brain! Do you want to load '{model_file}'? (Y/N): ").lower()
        if choice in ["y", "yes"]:
            load_model = True

    if load_model:
        print(f"Loading '{model_file}'...")
        trained_q_table = np.load(model_file)
        
    else:
        print("\nTraining a new AI...")

        # Card counting works best if trained over 2M episodes (takes 15-30 secs)
        trained_q_table = train_ai(env, episodes=100000) 
        
        test_ai(env, trained_q_table, game_count=1000)
        
        save_choice = input(f"Do you want to save your new brain as '{model_file}' for next time? (Y/N): ").lower()
        if save_choice in ["y", "yes"]:
            np.save(model_file, trained_q_table)
            print(f"Brain successfully saved to {model_file}!")

    test_ai(env, trained_q_table, game_count=10000)
    
    # 3. Check specific scenarios
    check_strategy(trained_q_table, p_sum=16, d_sum=10, have_a=0, c_bucket=2) # Classic hard hand
    check_strategy(trained_q_table, p_sum=20, d_sum=7, have_a=0, c_bucket=2)  # Classic winning hand

if __name__ == "__main__":
    main()

