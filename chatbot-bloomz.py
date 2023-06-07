from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import torch

language_model = "bigscience/bloomz-1b1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForCausalLM.from_pretrained(language_model, use_cache=True).to(device)
tokenizer = AutoTokenizer.from_pretrained(language_model)

while True:

    user_question = input("User: ")

    #
    summary_instruction = f"\nAnswer:\n"

    # Combine
    prompt = user_question + summary_instruction

    # Generate the answer
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    sample = model.generate(input_ids, max_length=5000, top_k=1, temperature=0.3, repetition_penalty=2.0)
    #
    # max_length: The maximum length of the generated output. In this case, it is set to 5000 tokens.
    # top_k: The number of highest probability choices to consider at each generation step. Setting top_k to 1 means only the top choice is considered.
    # temperature: The temperature parameter controls the randomness of the generated text. Higher values (e.g., 1.0) make the output more diverse, while lower values (e.g., 0.3) make it more focused and deterministic.
    # repetition_penalty: The repetition penalty encourages the model to avoid repeating the same tokens in its output. A higher value, such as 2.0, makes the model more likely to avoid repetitions.
    generated_text = tokenizer.decode(sample[0], skip_special_tokens=True)

    print("ChatBot:", generated_text)

    #end the conversation
    if user_question.lower() == "exit":
        break

