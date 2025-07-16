from dotenv import load_dotenv
load_dotenv()
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.core.tools import FunctionTool
from typing import List, Tuple, Dict
import random
from collections import Counter
from openai import OpenAI

from judgeval.tracer import Tracer, wrap

# Initialize Judgment tracer
judgment = Tracer(project_name="blackjack_agent")

# Initialize and wrap the OpenAI client
openai_client = wrap(OpenAI())

import logging
import sys


# ----------------------------
# Standard Deck Definition
# ----------------------------

STANDARD_DECK = Counter({
    '2': 4, '3': 4, '4': 4, '5': 4, '6': 4, '7': 4, '8': 4, '9': 4,
    '10': 4, 'J': 4, 'Q': 4, 'K': 4, 'A': 4
})


# ----------------------------
# Function 1: Parse Hand
# ----------------------------
@judgment.observe(span_type="tool")
def parse_hand(cards: List[str]) -> Tuple[int, bool]:
    values = []
    for card in cards:
        if card.upper() in ['J', 'Q', 'K']:
            values.append(10)
        elif card.upper() == 'A':
            values.append(11)
        else:
            values.append(int(card))
    total = hand_value(values)
    is_soft = 11 in values and total <= 21
    # print(f"[LOG] parse_hand({cards}) -> total={total}, is_soft={is_soft}")
    return total, is_soft


# ----------------------------
# Function 2: Hand Value
# ----------------------------
@judgment.observe(span_type="tool")
def hand_value(hand: List[int]) -> int:
    value = sum(hand)
    aces = hand.count(11)
    while value > 21 and aces:
        value -= 10
        aces -= 1
    # print(f"[LOG] hand_value({hand}) = {value}")
    return value


# ----------------------------
# Function 3: Simulate Best Decision
# ----------------------------

@judgment.observe(span_type="tool")
def simulate_best_decision(cards: List[str], dealer_card: str, num_simulations: int = 100) -> str:
    dealer_val = 11 if dealer_card.upper() == 'A' else 10 if dealer_card.upper() in ['J', 'Q', 'K'] else int(dealer_card)

    values = []
    for card in cards:
        if card.upper() in ['J', 'Q', 'K']:
            values.append(10)
        elif card.upper() == 'A':
            values.append(11)
        else:
            values.append(int(card))

    def draw_card():
        return random.choice([2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11])

    def simulate_dealer():
        hand = [dealer_val, draw_card()]
        while hand_value(hand) < 17:
            hand.append(draw_card())
        return hand_value(hand)

    # STAND scenario
    stand_total = hand_value(values)
    stand_results = {"win": 0, "loss": 0, "draw": 0}

    for _ in range(num_simulations):
        dealer_total = simulate_dealer()
        if dealer_total > 21 or stand_total > dealer_total:
            stand_results["win"] += 1
        elif stand_total < dealer_total:
            stand_results["loss"] += 1
        else:
            stand_results["draw"] += 1

    # HIT scenario
    hit_results = {"win": 0, "loss": 0, "draw": 0}

    for _ in range(num_simulations):
        player_hand = values.copy()
        player_hand.append(draw_card())
        while hand_value(player_hand) < 17:
            player_hand.append(draw_card())

        player_total = hand_value(player_hand)
        if player_total > 21:
            hit_results["loss"] += 1
            continue

        dealer_total = simulate_dealer()
        if dealer_total > 21 or player_total > dealer_total:
            hit_results["win"] += 1
        elif player_total < dealer_total:
            hit_results["loss"] += 1
        else:
            hit_results["draw"] += 1

    # Comparison
    stand_win = round(stand_results["win"] / num_simulations, 3)
    hit_win = round(hit_results["win"] / num_simulations, 3)

    recommendation = "HIT" if hit_win > stand_win else "STAND"
    msg = f"Simulated {num_simulations} games per move.\n"
    msg += f" - If you STAND: Win rate = {stand_win*100:.1f}%\n"
    msg += f" - If you HIT:   Win rate = {hit_win*100:.1f}%\n\n"
    msg += f"👉 Based on simulation, you should **{recommendation}**."

    print(f"[LOG] simulate_best_decision(cards={cards}, dealer_card={dealer_card}) = {recommendation}")
    return msg


# ----------------------------
# Function 4: Simulate Best Decision With Seen Cards
# ----------------------------

@judgment.observe(span_type="tool")
def simulate_best_decision_with_seen(cards: List[str], dealer_card: str, seen_cards: Dict[str, int], num_simulations: int = 1000) -> str:
    deck = STANDARD_DECK.copy()
    for card in cards + [dealer_card]:
        deck[card.upper()] -= 1
    for card, count in seen_cards.items():
        deck[card.upper()] -= count

    def value_from_card(c):
        return 11 if c.upper() == 'A' else 10 if c.upper() in ['J', 'Q', 'K'] else int(c)

    flat_deck = []
    for card, count in deck.items():
        val = value_from_card(card)
        flat_deck.extend([val] * max(count, 0))  # avoid negative counts

    def draw_card():
        return random.choice(flat_deck)

    def simulate_dealer():
        hand = [value_from_card(dealer_card), draw_card()]
        while hand_value(hand) < 17:
            hand.append(draw_card())
        return hand_value(hand)

    values = [value_from_card(c) for c in cards]

    stand_total = hand_value(values)
    stand_results = {"win": 0, "loss": 0, "draw": 0}
    for _ in range(num_simulations):
        dealer_total = simulate_dealer()
        if dealer_total > 21 or stand_total > dealer_total:
            stand_results["win"] += 1
        elif stand_total < dealer_total:
            stand_results["loss"] += 1
        else:
            stand_results["draw"] += 1

    hit_results = {"win": 0, "loss": 0, "draw": 0}
    for _ in range(num_simulations):
        hand = values.copy()
        hand.append(draw_card())
        while hand_value(hand) < 17:
            hand.append(draw_card())
        player_total = hand_value(hand)
        if player_total > 21:
            hit_results["loss"] += 1
            continue
        dealer_total = simulate_dealer()
        if dealer_total > 21 or player_total > dealer_total:
            hit_results["win"] += 1
        elif player_total < dealer_total:
            hit_results["loss"] += 1
        else:
            hit_results["draw"] += 1

    stand_win = round(stand_results["win"] / num_simulations, 3)
    hit_win = round(hit_results["win"] / num_simulations, 3)
    recommendation = "HIT" if hit_win > stand_win else "STAND"

    msg = f"Simulated {num_simulations} games with adjusted deck.\n"
    msg += f" - STAND win rate: {stand_win*100:.1f}%\n"
    msg += f" - HIT win rate:   {hit_win*100:.1f}%\n\n"
    msg += f"👉 With seen cards considered, you should **{recommendation}**."

    print(f"[LOG] simulate_best_decision_with_seen(cards={cards}, dealer_card={dealer_card}, seen_cards={seen_cards}) = {recommendation}")
    return msg

tools = [
    FunctionTool.from_defaults(
        fn=parse_hand, 
        description= "Given a list of cards (e.g. ['A', '6']), return the total hand value and whether the hand is soft (contains Ace as 11)."
    ),
    FunctionTool.from_defaults(
        fn=hand_value,
        description="Given a list of card values (e.g. [10, 11]), return the final hand value considering Aces."
    ),
    FunctionTool.from_defaults(
        fn=simulate_best_decision,
        description="Given a list of player's cards and a dealer's visible card, simulate hit vs stand outcomes and recommend the better move."
    ),
    FunctionTool.from_defaults(
        fn=simulate_best_decision_with_seen,
        description="Simulate hit vs stand while accounting for previously seen cards (single 52-card deck)."
    )
]


# Use llama_index OpenAI client with the model
llm = LlamaOpenAI(model="gpt-4o", temperature=0.0)
agent = ReActAgent.from_tools(tools,
            llm=llm, verbose=True, max_iterations=30)

prompt = """
You are a smart Blackjack decision agent.

You have access to the following tools:

- parse_hand(cards: List[str]) -> (int, bool): Parse a list of cards into the total value and whether the hand is soft.
- hand_value(hand: List[int]) -> int: Calculate the final value of a hand, taking Aces into account.
- simulate_best_decision(cards: List[str], dealer_card: str) -> str: Simulate both HIT and STAND outcomes and recommend the better move based on win rate.
- simulate_best_decision_with_seen(cards: List[str], dealer_card: str, seen_cards: Dict[str, int])

Please answer the user’s question using the tools step-by-step:
1. First, describe your plan.
2. Then call tools in the correct order.
3. Finally, explain your result.

Use parse_hand and hand_value before simulating if possible.
If the user mentions already played cards, use the simulation with seen cards.

Here is the user's question:
{user_input}
"""

user_input = input("\nYou: ")

@judgment.observe(span_type="function")
def run_agent(user_input: str) -> str:
    """
    This function runs the blackjack agent and is traced by judgeval.
    
    Note: The llama_index agent itself uses its own OpenAI client which is not wrapped by judgeval.
    However, we can still trace this function and all the tool functions.
    
    If you want to trace the actual LLM calls, you need to use the wrapped OpenAI client directly.
    """
    formatted_prompt = prompt.format(user_input=user_input)
    
    # Use the llama_index agent for the response
    agent_response = agent.chat(formatted_prompt)
    
    # Optional: You can also make a direct call with the wrapped OpenAI client
    # to trace the LLM call itself
    # openai_response = openai_client.chat.completions.create(
    #     model="gpt-4o",
    #     messages=[{"role": "user", "content": formatted_prompt}]
    # )
    
    return agent_response.response

# Run the agent
response = run_agent(user_input)
print("\n🤖 Agent:\n", response)