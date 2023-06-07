from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "bigscience/bloomz"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint)

inputs = tokenizer.encode("Translate to English: Je t’aime.", return_tensors="pt")
outputs = model.generate(inputs)
print(tokenizer.decode(outputs[0]))


