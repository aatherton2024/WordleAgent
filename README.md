## Wordle

This implementation of a Wordle Agent comes with an example agent RandomAgent. To see RandomAgent in action go to the initialize_agent method
in agent.py and uncomment the line of code that returns a RandomAgent.

To see the agent in action:

python3 game.py -a data/allowed.txt -p data/possible.txt

To see the agent run with a histogram for guess count distribution:

python3 game.py -a data/allowed.txt -p data/possible.txt -m histogram
