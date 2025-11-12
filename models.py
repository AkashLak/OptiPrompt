import time
import random
import re
import typing as T
from .types import Params

###############
#Model adapters (Optimizer --> Different models)
###############

class ModelClient:
    """
    Base case for all the model adapters (real or models must follow)
    """
    def generate(self, prompt: str, params: Params) -> T.Tuple[str, dict]:
        """
        Every model adapter implements the generate method
        Input: Prompt (full text prompt), params (model settings, like temp, and max_tokens)
        Output: Tuple that contains (model_output_text, usage_information)
        """
        raise NotImplementedError


class DummyClient(ModelClient):
    """
    Offline simulator for fast, and free testing of the optimizer loop
    """
    def generate(self, prompt: str, params: Params):
        """
        Main function that optimizer calls to generate a response from the fake model
        """
        #Simulating a small delay to mimic a real API call
        time.sleep(0.02)
        #Creating a fake quality curve where the accuracy peaks around temperature of 0.3 --> Similar to real LLMs
        base = 0.64 - 0.08 * abs(params.temperature - 0.3)
        #Using RegEx to make the phrasing matter
        is_math_q = bool(re.search(r"\d+\s*[\+\-\*/]\s*\d+", prompt))
        if "When arithmetic appears, compute carefully." in prompt and is_math_q:
            #Math hint actually matters for mathematical expressions
            base = min(0.99, base + 0.20)
        if "Internally organize reasoning as bullet points." in prompt:
            #Coherence score increase due to longer text
            pass
        if "If relevant, include units in the final answer." in prompt and not is_math_q:
            #Penalty if hint is irrelevant to question
            base = max(0.10, base - 0.02)

        #Picking correct answer based on prompt text
        if "Harry Potter Books" in prompt:
            gold = "Final: J.K. Rowling"
        elif "13 + 31" in prompt:
            gold = "Final: 44"
        elif "12 * 11" in prompt:
            gold = "Final: 132"
        elif "Capital of France" in prompt:
            gold = "Final: Paris"
        elif "Derivative of x^3 + 3x^2" in prompt:
            gold = "Final: 3x^2 + 6x"
        else:
            #Default value (so no error produced)
            gold = "Final: 40"
        #Randomly decides if the model answers correctly based on the "base" + accuracy
        answer = gold if random.random() < base else "Final: Not sure"
        #Creating a fake usage report for apprxomiate token count
        usage = {
            "prompt_tokens": len(prompt) // 4, 
            "completion_tokens": 32}
        return answer, usage


class OpenAIClient(ModelClient):
    """
    The adaptor for real OpenAI GPT models (live calls)
    """
    def __init__(self, model: str, api_key: str):
        try:
            import openai
        except Exception as e:
            raise RuntimeError("Install openai: pip install openai") from e
        #Saving key + client for later usage
        self.openai = openai
        self.openai.api_key = api_key
        self.model = model

    def generate(self, prompt: str, params: Params):
        """
        Function used to send given prompt + model settings to real OpenAI API, and gets the generated response,
        returning both the text output and usage info (tokens used)
        """
        #Conversation format (system + user messages)
        messages = [
            {"role": "system", "content": "You are a careful, precise assistant."},
            {"role": "user", "content": prompt},
        ]
        #Sending a request to OpenAI Chat API, providing the creativity level, word sampling, and response length limit
        resp = self.openai.chat.completions.create(
            model = self.model,
            messages = messages,
            temperature = params.temperature,
            top_p = params.top_p,
            max_tokens = params.max_tokens,
        )
        #Extracting the models generated text output
        text = resp.choices[0].message.content.strip()
        #Extracting the responses token usage info for cost computation
        u = resp.usage
        usage = {
            "prompt_tokens": getattr(u, "prompt_tokens", 0),
            "completion_tokens": getattr(u, "completion_tokens", 0),
        }
        return text, usage