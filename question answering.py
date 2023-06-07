from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
language_model = "bigscience/bloomz-1b1"

model = AutoModelForCausalLM.from_pretrained(language_model)
tokenizer = AutoTokenizer.from_pretrained(language_model)

context = "Albert Einstein was a German-born theoretical physicist. He developed the theory of relativity, one of the two pillars of modern physics. Einstein is best known for his mass-energy equivalence formula E = mc^2."

question = "What is Albert Einstein known for?"

encoding = tokenizer.encode_plus(question, context, return_tensors="pt")
input_ids = encoding["input_ids"]
attention_mask = encoding["attention_mask"]

start_scores, end_scores = model(input_ids, attention_mask=attention_mask)

start_index = torch.argmax(start_scores, dim=1).item()
end_index = torch.argmax(end_scores, dim=1).item()

answer_tokens = input_ids[0][start_index:end_index+1]
answer = tokenizer.decode(answer_tokens)


print("Question:", question)
print("Answer:", answer)

