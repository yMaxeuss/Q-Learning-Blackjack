# Reinforcement Learning Blackjack Agent

An Artificial Intelligence built from scratch using Python and **Q-Learning**. This agent learns to play Blackjack by playing millions of simulated hands, and features a dynamic **Card Counting** toggle that allows it to adjust its betting strategy based on the True Count.

## Features
* **Custom Blackjack Environment:** A fully built object-oriented Blackjack engine.
* **Q-Learning Algorithm:** The AI uses the Bellman Equation to map out expected values for every possible game state.
* **Card Counting Integration:** Toggleable True Count tracking (Hi-Lo system) that splits the AI's "brain" to understand deck composition.
* **Dynamic Betting System:** The AI mathematically sizes its wagers based on the statistical advantage provided by the True Count.
* **Data Visualization:** Uses `matplotlib` to chart the AI's bankroll over tens of thousands of hands to visualize the law of large numbers and variance.

## Dependent classes
* Numpy
* Matplotlib
