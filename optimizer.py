import time
import random
import typing as T
from .types import Arm, Trial
from .bandits import UCB1
from .evals import accuracy_fuzzy, coherence_simple

def _truncate(s: str, n: int = 42) -> str:
    """
    Helper function to shorten long string so fits better in print
    """
    return s if len(s) <= n else s[: n - 1] + "…"

def _rule(char: str = "─", n: int = 80) -> str:
    #Helper function to create a horizontal divider line, which makes output more clear
    return char * n

###############
#Optimizer
###############
class Optimizer:
    """
    Runs the optimization process
    Manages testing of arms, scoring performance, and learning best setup with UCB1 Bandit Algorithm
    """
    def __init__(self, model, arms: T.List[Arm], w_acc = 0.85, w_coh = 0.15, w_cost = 0.01):
        """
        Initializing the optimizer with model, all arms, and weight values for accuracy, coherence, and cost
        """
        self.model = model
        self.arms = {a.arm_id: a for a in arms}
        self.bandit = UCB1(list(self.arms.keys()))
        self.w_acc, self.w_coh, self.w_cost = w_acc, w_coh, w_cost
        #List of all trials (history of runs)
        self.replay: T.List[Trial] = []
        #Cache used to avoid computing same input + arm results twice
        self.cache: dict[tuple[str, str], tuple[str, dict]] = {}

    def _score(self, pred: str, ref: str, usage: dict) -> T.Tuple[float, float, float, float]:
        """
        Private helper function for calculating performance score for 1 model response
        Combines accuracy, coherence and cost, while returning them as well
        """
        acc = accuracy_fuzzy(pred, ref)
        coh = coherence_simple(pred)
        cost = (usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)) / 1000.0
        score = self.w_acc * acc + self.w_coh * coh - self.w_cost * cost
        return acc, coh, cost, score

    def run_example(self, inp: str, ref: str) -> Trial:
        """
        Runs a single (input, reference) example with one selected arm
        Output: Returns a trial object containing all details of this run
        """
        #Selecting which arm to try + Filling base template with actual question
        arm_id = self.bandit.select()
        arm = self.arms[arm_id]
        prompt = arm.template.format(input=inp)
        #Checking if result already in cache
        key = (inp, arm_id)
        start = time.time()
        if key in self.cache:
            out, usage = self.cache[key]
        else:
            out, usage = self.model.generate(prompt, arm.params)
            self.cache[key] = (out, usage)
        #Measuring model time
        latency = time.time() - start
        #Evaluating outputs performance and giving reward feedback to UCB1 bandit
        acc, coh, cost, score = self._score(out, ref, usage)
        self.bandit.update(arm_id, score)
        #Recording trial informaton
        trial = Trial(inp, ref, arm_id, out, acc, coh, cost, score, latency)
        self.replay.append(trial)
        return trial

    def fit_stream(self, data: T.List[dict], epochs = 1):
        """
        Training optimizer on dataset of input + reference pairs
        Input: data (list of questions and answers), epochs are amount of passes to make through the dataset
        """
        for e in range(epochs):
            #Shuffling order for variety
            random.shuffle(data)
            print(f"\nEpoch {e}  | Trials")
            print(_rule())
            print(f"{'ARM':<8} {'PHRASING':<12} {'TEMP':>5}  {'ACC':>5}  {'COH':>5}  {'SCORE':>6}  {'LAT(s)':>6}  {'INPUT':<42}")
            print(_rule())
            for row in data:
                #Running a training example, obtaining details for arm used, and printing the trial summary
                t = self.run_example(row["input"], row["reference"])
                a = self.arms[t.arm_id]
                print(
                    f"{t.arm_id:<8} {a.meta['phrasing']:<12} {a.meta['temperature']:>5.1f}  "
                    f"{t.accuracy:>5.2f}  {t.coherence:>5.2f}  {t.score:>6.2f}  {t.latency_s:>6.2f}  "
                    f"{_truncate(row['input'])}"
                )

    def best_arm(self) -> Arm:
        """
        Finds and returns the best performing arm based on average score across all trials
        If there are no trials yet, then a random arm is returned
        """
        if not self.replay:
            return random.choice(list(self.arms.values()))
        #Aggregating all scores for each arm
        agg: dict[str, list[float]] = {}
        for t in self.replay:
            agg.setdefault(t.arm_id, []).append(t.score)
        #Selecting arm with higest average score
        best_id = max(agg.items(), key=lambda kv: sum(kv[1]) / len(kv[1]))[0]
        return self.arms[best_id]

    def best_arm_per_input(self) -> dict[str, tuple[Arm, float]]:
        """
        Returns a dict mapping each unique input_text to its best performing arm and corresponding average score
        Winner
        """
        if not self.replay:
            #If no trials yet
            return {}

        #Step 1: Aggregate all trial scores by (input_text, arm_id)
        per_input_scores: dict[str, dict[str, list[float]]] = {}
        for t in self.replay:
            per_input_scores.setdefault(t.input_text, {}).setdefault(t.arm_id, []).append(t.score)

        #Step 2: For each input, find the arm with highest avg score
        result: dict[str, tuple[Arm, float]] = {}
        for inp, arm_map in per_input_scores.items():
            best_id, scores = max(
                arm_map.items(),
                #Computing avg per ar
                key=lambda kv: sum(kv[1]) / len(kv[1])
            )
            avg_score = sum(scores) / len(scores)
            #Save arm object, avg score for certain input
            result[inp] = (self.arms[best_id], avg_score)
        return result

    def topk_arms_per_input(self, k: int = 3) -> dict[str, list[tuple[Arm, float]]]:
        """
        Returns top k performing arms for each input, sorted by avg score in desc order
        Leaderboard (top k --> 3)
        """
        if not self.replay:
            return {}

        #Step 1: Aggregate trial scores as before
        per_input_scores: dict[str, dict[str, list[float]]] = {}
        for t in self.replay:
            per_input_scores.setdefault(t.input_text, {}).setdefault(t.arm_id, []).append(t.score)
        #Step 2: For each input, compute avg scores per arm
        out: dict[str, list[tuple[Arm, float]]] = {}
        for inp, arm_map in per_input_scores.items():
            ranked = []
            for arm_id, scores in arm_map.items():
                avg_score = sum(scores) / len(scores)
                ranked.append((self.arms[arm_id], avg_score))
            #Step 3: Sort arms by score, where highest is first
            ranked.sort(key=lambda x: x[1], reverse=True)
            #Keep only top k arms
            out[inp] = ranked[:k]
        return out