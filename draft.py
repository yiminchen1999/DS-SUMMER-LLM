from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import torch
model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"

# Load the model and tokenizer
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

while True:
    # Prompt the user for a question
    user_question = input("User: ")

    # Encode the question and context
    inputs = tokenizer(user_question,return_tensors="pt")

    #
    with torch.no_grad():
        outputs = model(**inputs)
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits


    start_index = torch.argmax(start_logits)
    end_index = torch.argmax(end_logits) + 1
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start_index:end_index]))

    print("ChatBot:", answer)

