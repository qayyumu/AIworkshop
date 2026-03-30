### This example is about using Openai GPT model and Dspy features to create a simple question-answering agent.
#### pip install dspy
### Dspy (Declarative Self-improving Python) is a library for creating declarative self-improving Python programs.
import dspy
import os
from dotenv import load_dotenv
load_dotenv()
lm = dspy.LM('openai/gpt-4o-mini', api_key=os.getenv('OPENAI_API_KEY'))

print(lm("Write about the latest trends in AI!", temperature=0.7))  # => ['This is a test!']
# print(lm(messages=[{"role": "user", "content": "Say this is a test!"}]))  # => ['This is a test!']
# print(lm.history[-1].keys())  # access the last call to the LM, with all metadata

########################################################
#### Dspy features
dspy.configure(lm=lm)
# Define a module (ChainOfThought) and assign it a signature (return an answer, given a question).
qa = dspy.ChainOfThought('question -> answer')
# Run with the default LM configured with `dspy.configure` above.
response = qa(question="How many provinces are in Pakistan?")
print(response.answer)

sentence = "it's a charming and often affecting journey to Lahore."  
classify = dspy.Predict('sentence -> sentiment: bool')  # 
print(sentence, "-> Sentiment: ", classify(sentence=sentence).sentiment)





