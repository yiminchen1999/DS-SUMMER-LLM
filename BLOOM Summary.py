

summary_instruction = f"\n\nSummarize the previous text in three sentences:\n\n"

total_prompt = prompt + summary_instruction

input_ids = tokenizer(prompt + f"\n\n" + summary_instruction, return_tensors="pt").to(0)
sample = model.generate(**input_ids, max_length=5000,  top_k=1, temperature=0.9, repetition_penalty = 2.0)

result_string = tokenizer.decode(sample[0], truncate_before_pattern=[r"\n\n^#", "^'", "\n\n\n"])

print(result_string[len(total_prompt)::])