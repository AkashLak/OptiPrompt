from rapidfuzz import fuzz

###############
#Evaluators
###############
def accuracy_fuzzy(pred: str, ref: str) -> float:
    """
    Measures how close the models output (pred) is to the correct reference answer using fuzzy string matching
    Input: Models predicted output + correct reference answer
    Output: Returns value between 0 and 1 (perfect match)
    """
    return fuzz.token_set_ratio(pred, ref) / 100.0


def coherence_simple(pred: str) -> float:
    """
    Evaluates coherence/how well formed the models output is
    Input: Models predicted output
    Output: Returns value between 0 and 1
    """
    #Splitting output text into tokens/words
    toks = pred.split()
    if not toks:
        #If no output then coherence just 0
        return 0.0
    #Measures the word repetition, where longer text increase score by up to 50 tokens 
    uniq = len(set(toks)) / len(toks)
    length_bonus = min(1.0, len(toks) / 50.0)
    #Returns a weighted average
    return max(0.0, min(1.0, 0.5 * uniq + 0.5 * length_bonus))