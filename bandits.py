import math
import typing as T

###############
#UCB1 bandit Algorithm
###############
class UCB1:
    """
    Implementing an Upper Confidence Bound Algorithm, 
    which is a RL method (balances exploration and exploitation when choosing between multiple arms/choices)
    """
    def __init__(self, arm_ids: T.List[str]):
        """
        Initializing bandit with all possible arms
        Each arm starts with 0 pulls (no trials), and a total reward of 0
        """
        self.counts = {a: 0 for a in arm_ids}
        self.totals = {a: 0.0 for a in arm_ids}
        self.t = 0

    def select(self) -> str:
        """
        Function used to choose which arm to test next via the UCB1 Rule
        """
        #Counting total # of trials
        self.t+= 1
        #Step 1: Making sure each arm is tried atleast once before using UCB formula
        for a, c in self.counts.items():
            if c == 0:
                return a
        #Step 2: Computing UCB score for each arm
        ucbs = {}
        for a in self.counts:
            #Average reward
            avg = self.totals[a] / self.counts[a]
            #Confidence Interval
            bonus = math.sqrt(2.0 * math.log(self.t) / self.counts[a])
            #Combining both --> reward + uncertainity
            ucbs[a] = avg + bonus
        return max(ucbs.items(), key=lambda kv: kv[1])[0]

    def update(self, arm_id: str, reward_0_1: float):
        """
        After running trial with specific arm, update its statistics with the new reward (0-1)
        """
        #Limiting reward for 0 to 1
        r = max(0.0, min(1.0, reward_0_1))
        #Increase count and add new reward to the running total
        self.counts[arm_id]+= 1
        self.totals[arm_id]+= r