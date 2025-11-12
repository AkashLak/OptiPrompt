import os

###############
#Configuring
###############
#Currently set to True since using a created Dummy Model (as dont have OpenAI API key/credits)
#Can change to False if want to use OpenAI API (need to provide a key)
USE_DUMMY_MODEL = True
MODEL_NAME = "gpt-4o-mini"         
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

#Scoring weights
W_ACC = 0.9
W_COH = 0.1
W_COST = 0.01