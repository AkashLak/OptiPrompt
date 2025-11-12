import typing as T
from .types import Arm, Params

###############
#CoT-only base template + phrasing variants + temperature grid
###############

#Main chain of thought instructions for model to think step-by-step internally
#but only shows the final answer, not reasoning
BASE_COT = (
    "Think step by step to reach the answer, but output only the final result.\n"
    "Return only the final answer prefixed by 'Final:'.\n\n"
    "Question: {input}\n"
)
#Different versions for the base prompt, 
#where each one adds a small hint that changes how the middle thinks internally
#Plain: no extra hint (default CoT), unit_hint: model includes units, 
#bullet_ref: organized, bullet point style reasoning, math_hint: doubles check arithmetics
PHRASINGS = [
    ("plain", lambda s: s),
    ("units_hint", lambda s: s + "\nIf relevant, include units in the final answer."),
    ("bullet_pref", lambda s: s + "\nInternally organize reasoning as bullet points."),
    ("math_hint", lambda s: s + "\nWhen arithmetic appears, compute carefully."),
]

#List of temperature values to test (creativity/randomness of model)
TEMP_GRID = [0.0, 0.2, 0.4, 0.7]


def build_arms() -> T.List[Arm]:
    """
    returns a list of Arm objects that define every testable prompt setup (combination of phrasing styles, and temperatures)
    """
    #Stores all arms
    arms: T.List[Arm] = []
    idx = 0
    #Looping through every phrasing style
    for tag, mutate in PHRASINGS:
        #Phrase modification based on base CoT prompt and each temperature value
        tmpl = mutate(BASE_COT)
        for temp in TEMP_GRID:
            arms.append(Arm(
                #Unique label for arms
                arm_id = f"arm_{idx}",
                #Storing modified template of prompt
                template = tmpl,
                params = Params(temperature = temp, max_tokens = 256, top_p = 1.0),
                #Metadata for the phrasing
                meta = {"phrasing": tag, "temperature": temp},
            ))
            idx+= 1
    return arms

