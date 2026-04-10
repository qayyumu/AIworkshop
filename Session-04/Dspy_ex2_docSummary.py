### This example is loading a document and summarizing it using Dspy features.
import dspy
import os
from dotenv import load_dotenv
load_dotenv()
lm = dspy.LM('openai/gpt-4o-mini', api_key=os.getenv('OPENAI_API_KEY'))

with open('./Session-04/Dspy_ex1.py', 'r') as file:
    document = file.read()
# print(document)
########################################################
#### Dspy features
dspy.configure(lm=lm)

summarize = dspy.ChainOfThought('document -> summary')
summary = summarize(document=document)
print("Document Summary: ",summary.summary)

# print("\n📋 GENERATED PROMPT (last LLM call):")
# print("-" * 70)
# lm.inspect_history(n=1)



