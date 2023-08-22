import sys
from abc import ABC, abstractmethod
from collections import defaultdict
from numpy import mean
from random import choice, sample, shuffle
from tqdm import tqdm
from util import *
from queue import Queue


def initialize_agent(allowed, possible):
    #return RandomAgent(allowed, possible)
    return MyWordleAgent(allowed, possible)

class WordleAgent(ABC):

    def __init__(self, allowed, possible):
        self.allowed = allowed
        self.possible = possible

    @abstractmethod
    def first_guess(self):
        """Makes the first guess of a Wordle game.

        A WordleGame will call this method to get the agent's first guess of the game.
        This is an implicit signal to the agent that a new game has begun. Subsequent
        guess requests during the same game will use the .next_guess method.

        Returns
        -------
        str
            The first guess of a game of Wordle
        """
        ...

    @abstractmethod
    def next_guess(self):
        """Makes the next guess of an in-progress Wordle game.

        A WordleGame will call this method to get the agent's next guess of an
        in-progress game.

        Returns
        -------
        str
            The next guess of the agent, during an in-progress game of Wordle
        """
        ...

    @abstractmethod
    def report_feedback(self, guess, feedback):
        """Provides feedback to the agent after a guess.

        After the agent makes a guess, a WordleGame calls this method to deliver
        feedback to the agent about the guess. No return value is expected from the
        method call.

        Feedback takes the form of a list of colors, corresponding to the letters
        of the guess:
        - "green" means the guessed letter is in the target word, and in the specified position
        - "yellow" means the guessed letter is in the target word, but not in the specified position
        - "gray" means the guessed letter is not in the target word

        For instance, if the WordleGame calls:
            agent.report_feedback("HOUSE", ["gray", "green", "gray", "gray", "yellow"])
        Then the agent can infer that:
            - the target word has the letter "O" in position 2 (counting from 1)
            - the target word contains the letter "E", but not in position 5
            - the target word does not contain letters "H", "U", or "S"
        An example target word that fits this feedback is "FOYER".

        There are some important special cases when the guess contains the same letter in
        multiple positions. Suppose the letter X appears M times in the guess and N times
        in the target:
            - the K appearances of X in a correct position will be "GREEN"
            - if M <= N, then all other appearances of X will be "YELLOW"
            - if M > N, then N-K of the other appearances of X (selected arbitrarily) will
            be "YELLOW". The remaining appearances of X will be "GRAY"

        Parameters
        ----------
        guess : str
            The guess made by the agent
        feedback : list[str]
            A list of colors (expressed as strings "green", "yellow", "gray") corresponding
            to the letters in the guess
        """
        ...

class RandomAgent(WordleAgent):
    """A WordleAgent that guesses (randomly) from among words that satisfy the accumulated feedback."""

    def __init__(self, allowed, possible):
        super().__init__(allowed, possible)
        self.pool = self.possible

    def first_guess(self):
        self.pool = self.possible
        shuffle(self.pool)
        return self.next_guess()

    def next_guess(self):
        shuffle(self.pool)
        return self.pool[0]

    def report_feedback(self, guess, feedback):
        self.pool = filter_possible_words(guess, feedback, self.pool)
        print(self.pool)
        print(len(self.pool))

class Tree():
    """
    A class to represent an expectimax tree. Tree has depth 5 and branching factor of 3.
    The leftwards node at level one of the tree are all words in pool with same first letter
    as the word stored in the root node. The middle node at this level is all words that have
    the first letter of the root node word appear somewhere that isn't their first letter. The 
    rightwards node is then all other words. The leftwards node at level two is all words with 
    the same first two letters as the root node word. Observe this pattern continues and is
    illustrated below to level 2 (obviously extends further):
    
    guess --- green ------------------- green, green ---- ...
          |                          |                                               
          |                          |-- green, yellow --- ...
          |                          |
          |                          |-- green, grey --- ...
          |                         
          |-- yellow-------------------- yellow, green --- ...  
          |                          |                         
          |                          |-- yellow, yellow --- ...
          |                          |
          |                          |-- yellow, grey --- ...                  
          |                          
          |-- grey-------------------- grey, green --- ...  
                                     |                         
                                     |-- grey, yellow --- ...
                                     |
                                     |-- grey, grey --- ... 

    Attributes:
    val -- str
        root word of tree
    rootNode -- Node
        root node of tree
    """
    def __init__(self, word, pool):
        """
        Initialize Tree

        Keyword arguments:
        word -- root node word the tree will be built around
        pool -- pool of words that will compose tree
        """
        self.val = word
        self.root_node = Node(pool)

class Node():
    """
    A class to represent a node used in Tree class
    
    Attributes:
    pool -- list[str]
        pool of possible words at this node
    green -- Node
        child node of this node, words in the pool of green have green feedback at current position
    yellow -- Node
        child node of this node, words in the pool of green have yellow feedback at current position
    grey -- Node
        child node of this node, words in the pool of green have grey feedback at current position
    """
    def __init__(self, pool):
        """
        Initialize Node

        Keyword arguments:
        pool -- pool of words stored in this node
        """
        self.pool = pool
        self.green = None
        self.yellow = None
        self.grey = None

class MyWordleAgent(WordleAgent):
    """
    A class to represent a search-based Wordle AI agent. Class uses caching and the expectimax search
    algorithm to determine the best guess at every stage of a wordle game.
    
    Attributes:
    pool -- list[str]
        pool of possible wordle answers
    canGuess -- list[str]
        pool of possible wordle guesses
    cache -- dictionary(str : str)
        stores strings of feedback patterns and their optimal associated guess
    feedbackSoFar -- str
        string of feedback received so far in current wordle game
    guess -- str
        the first guess to bed used in every game, has to be calculated once
    """

    def __init__(self, allowed, possible):
        """
        Initialize Wordle Agent

        Keyword arguments:
        allowed -- the list of allowable guesses
        possible -- the list of possible answers
        """
        super().__init__(allowed, possible)
        self.pool = self.possible
        self.can_guess = self.allowed
        self.cache = dict()
        self.feedback_so_far = None
        self.guess = None
        
    def first_guess(self):
        """
        Method to get the first guess in a game of wordle

        Returns:
        Optimal first guess
        """
        self.pool = self.possible
        self.can_guess = self.allowed
        self.feedback_so_far = ""
        if self.guess is None: self.guess = self.find_guess().val
        return self.guess
    
    def next_guess(self):
        """
        Method to get the optimal next guess in a game of wordle

        Returns:
        Optimal next guess
        """
        GUESS_RANDOM_POOL_SIZE = 3

        if len(self.pool) <= GUESS_RANDOM_POOL_SIZE: return self.pool[0]
        if self.feedback_so_far in self.cache: return self.cache.get(self.feedback_so_far)
        return self.find_guess().val
    
    def report_feedback(self, guess, feedback):
        """
        Show guess feedback and reduce pool of possible answers

        Keyword arguments:
        guess -- the guess used
        feedback -- feedback from the guess used
        """
        self.pool = filter_possible_words(guess, feedback, self.pool)
        self.feedback_so_far += ''.join(feedback)
    
    def find_guess(self):
        """
        Helper method to find optimal guess in cases where optimal guess is not 
        cached or first guess undetermined. Builds expectimax trees for every 
        guessable word.
        
        Returns:
        The expectimax tree with the lowest score (least expected words remaining if guess used)
        """
        best_tree = None
        best_score = float('inf')

        for guess in self.can_guess:
            cur_tree, score = self.recursive_build_tree(guess)
            if score < best_score: 
                best_tree = cur_tree
                best_score = score

        if len(self.feedback_so_far) != 0: self.cache.setdefault(self.feedback_so_far, best_tree.val)
        return best_tree
    
    def recursive_build_tree(self, guess):
        """
        Recursively build expectimax tree

        Keyword arguments:
        guess -- the guess used

        Returns:
        A tuple of the expectimax tree and its associated score
        """
        tree = Tree(guess, self.pool)
        score = self.calculate_score(guess, 0, tree.root_node)
        return (tree, score)
    
    def calculate_score(self, guess, pos, node):
        """
        Helper method to build expectimax tree reursively and calculate its score

        Keyword arguments:
        guess -- the guess used
        pos -- position of the current letter being considered
        node -- current tree node being considered

        Returns:
        The score of the expectimax tree (expected words remaining if guess used)
        """
        MIN_POOL_SIZE = 1
        LAST_POSITION = 4

        pool_len = len(node.pool)
        if pool_len <= MIN_POOL_SIZE: return 1
        if pos > LAST_POSITION: return pool_len

        green, yellow, grey = self.get_pool(guess, pos, node.pool)

        node.green = Node(green)
        node.yellow = Node(yellow)
        node.grey = Node(grey)
    
        green_score = (len(node.green.pool) / pool_len) * self.calculate_score(guess, pos + 1, node.green)
        yellow_score = (len(node.yellow.pool) / pool_len) * self.calculate_score(guess, pos + 1, node.yellow)
        grey_score = (len(node.grey.pool) / pool_len) * self.calculate_score(guess, pos + 1, node.grey)

        return green_score + yellow_score + grey_score

    def get_pool(self, guess, pos, pool):
        """
        Helper method to find the three descendant pools of the current node

        Keyword arguments:
        guess -- the guess used
        pos -- position of the current letter being considered
        pool -- the pool of words that will be split into thruple of lists

        Returns:
        A thruple ([green], [yellow], [grey]) where green is words in pool with green 
        feedback with guess at pos, etc...
        """
        ret = ([], [], [])
        for word in pool:
            if guess[pos] == word[pos]: ret[0].append(word)
            elif guess[pos] in word: ret[1].append(word)
            else: ret[2].append(word)
        return ret
