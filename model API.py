from transformers import AutoModelForCausalLM, AutoTokenizer

language_model = "bigscience/bloomz-1b1"

model = AutoModelForCausalLM.from_pretrained(language_model)
tokenizer = AutoTokenizer.from_pretrained(language_model)

prompt = "BLOOM 是一个"
#BLOOM是一个基于云计算和人工智能的开放式平台，为客户提供从数据分析到智能决策的全流程解决方案。
input_ids = tokenizer.encode(prompt, return_tensors="pt")

output = model.generate(input_ids, max_length=100, num_return_sequences=1, no_repeat_ngram_size=2, repetition_penalty=1.5)

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(generated_text)
