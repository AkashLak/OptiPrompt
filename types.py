from dataclasses import dataclass

###############
#Data structures (with Data Classes)
###############
@dataclass
class Params:
    """
    Stores the model parameters for a given test run 
    Temp: Creativity, Max_Tokens: Limit of tokens model can generate in a response, top_p: Broadness of models word choice
    """
    temperature: float = 0.0
    max_tokens: int = 256
    top_p: float = 1.0

@dataclass
class Arm:
    """
    Represents one prompt variation that the optimizer can try
    arm_id: Unique ID of test, template: Actual prompt text template, params: Model parameters from previous class, meta: Extra info about this arm -> Ex: {"phrasing": "math_hint}
    """
    arm_id: str
    template: str            
    params: Params
    meta: dict               

@dataclass
class Trial:
    """
    Stores result of one expirement (single model run using a specific arm id)
    """
    input_text: str
    #Correct answer for accuracy
    reference: str
    arm_id: str
    output_text: str
    #Closeness of models answer to correct reference (0-1)
    accuracy: float
    #Clearness of models text (0-1)
    coherence: float
    #Token cost for model run
    cost: float
    #Overall score: accuracy + coherence - cost
    score: float
    #How long the model took to respond (seconds)
    latency_s: float