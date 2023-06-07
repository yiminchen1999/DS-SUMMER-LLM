from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM


tokenizer = AutoTokenizer.from_pretrained('bigscience/bloom-1b3')
model = AutoModelForCausalLM.from_pretrained('bigscience/bloom-1b3')

generator = pipeline(task="text-generation", model=model, tokenizer=tokenizer)

# eng
generator("biostatistics is ", max_length=30, num_return_sequence=5)


#[{'generated_text': 'bioinformatics is  a branch of computer science that deals with the analysis of data and the design of algorithms that can be used to solve problems in'}]

# Chinese
generator("生物统计学是一门")

#[{'generated_text': '生物信息学是一门新兴的交叉学科，它涉及生物信息学、计算机科学、数学'}]
