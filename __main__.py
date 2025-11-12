from .config import USE_DUMMY_MODEL, MODEL_NAME, OPENAI_API_KEY, W_ACC, W_COH, W_COST
from .models import DummyClient, OpenAIClient
from .prompts import build_arms
from .optimizer import Optimizer, _rule

###############
#Demo Dataset (for testing purposes)
###############
#Testing dataset for how well the optimizer will learn
#Each item contains an input quest for model, and a reference for correct answer expected
DATA = [
    {"input": "Who wrote the Harry Potter Books?", "reference": "J.K. Rowling"},
    {"input": "Compute 13 + 31", "reference": "44"},
    {"input": "Capital of France?", "reference": "Paris"},
    {"input": "Derivative of x^3 + 3x^2?", "reference": "3x^2 + 6x"},
]


def main():
    #Can be Dummy client or OpenAI client
    model: ModelClient
    if USE_DUMMY_MODEL:
        model = DummyClient()
    else:
        assert OPENAI_API_KEY, "Set OPENAI_API_KEY to use OpenAI"
        model = OpenAIClient(MODEL_NAME, OPENAI_API_KEY)
    #Building all prompt variations, where each arm is a combo of phrasing style + temp
    arms = build_arms()
    #Initializing optimizer with model, the arms, and scoring weights
    opt = Optimizer(model, arms, w_acc=0.9, w_coh=0.1, w_cost=0.01)

    #Training optimizer on the demo dataset as it loops through dataset 3 times for learning best arms
    opt.fit_stream(DATA, epochs = 5)

    #Best Prompt per Question
    perq = opt.best_arm_per_input()
    print("\n=== Best Prompt per Question ===")
    print(_rule())
    print(f"{'INPUT':<42} {'PHRASING':<12} {'TEMP':>5}  {'AVG_SCORE':>9}")
    print(_rule())
    for inp, (arm, avg_score) in perq.items():
        print(f"{_truncate(inp):<42} {arm.meta['phrasing']:<12} {arm.meta['temperature']:>5.1f}  {avg_score:>9.2f}")

    #Displaying best performing arm/prompt setup thus far (Overall)
    best = opt.best_arm()
    print("\n=== Best Prompt (so far) ===")
    print(f"Phrasing: {best.meta['phrasing']}  |  Temperature: {best.meta['temperature']:.1f}")
    print(_rule())

    #Showing template with a real question filled in
    sample_input = DATA[0]["input"] if DATA else "Sample question"
    print("Prompt preview (filled):\n")
    print(best.template.format(input=sample_input))

    #Testing best arm on a new question thats not in the training data to see its performance
    test = {"input": "What is 12 * 11?", "reference": "132"}
    t = opt.run_example(test["input"], test["reference"])
    print("\n=== Fresh Trial ===")
    print(_rule())
    print(f"Question: {test['input']}")
    print(f"Output  : {t.output_text}")
    print(f"Score   : {t.score:.2f} (acc={t.accuracy:.2f}, coh={t.coherence:.2f}, cost={t.cost:.3f})")


if __name__ == "__main__":
    main()