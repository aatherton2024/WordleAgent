from agent import initialize_agent, MyWordleAgent
import unittest

class Test_recursive_build_tree(unittest.TestCase):
    def runTest(self):
        guesses = ["ebony"]
        answers = ["birch", "beech", "cedar", "ebony", "maple"]
        agent = initialize_agent(guesses, answers)
        score = agent.recursive_build_tree(guesses[0])[1]
        assert score == 1.40, "incorrect expectimax score"

class Test_first_guess(unittest.TestCase):
    def runTest(self):
        answers = ["birch", "beech", "cedar", "ebony", "maple"]
        agent = initialize_agent(answers, answers)
        first_guess = agent.first_guess()
        assert first_guess in ["birch", "beech"], "incorrect first guess"

class Test_get_pool(unittest.TestCase):
    def runTest(self):
        answers = ["birch", "beech", "cedar", "ebony", "maple"]
        agent = initialize_agent(answers, answers)
        pool = agent.get_pool(answers[0], 0, answers)
        assert pool[0] == ["birch", "beech"] and pool[1] == ["ebony"] and pool[2] == ["cedar", "maple"], "incorrect pool generation"


unittest.main()