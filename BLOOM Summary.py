from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import torch

language_model = "bigscience/bloomz-1b1"

# Enable CUDA if desired
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForCausalLM.from_pretrained(language_model, use_cache=True).to(device)
tokenizer = AutoTokenizer.from_pretrained(language_model)

# Source text
prompt = "BLOOMwastrainedontheROOTScorpus(Lauren¸conetal.,2022),acompositecollection of498HuggingFacedatasets(Lhoestetal.,2021)amountingto1.61terabytesoftextthat span46naturallanguagesand13programminglanguages.Ahigh-leveloverviewofthis datasetcanbeseeninFigure3,whileadetaileditemizedlistofeverylanguagealong withitslinguisticgenus,familyandmacroareaispresentedinTable1.Beyondthecorpus itself,theprocessresultedinthedevelopmentandreleaseofanumberoforganizational andtechnicaltools,includingthoseillustratedinFigure2."


summary_instruction = f"\n\nSummarize the previous text in three sentences:\n\n"

total_prompt = prompt + summary_instruction

input_ids = tokenizer.encode(total_prompt, return_tensors="pt").to(device)
sample = model.generate(input_ids, max_length=5000, top_k=1, temperature=0.9, repetition_penalty=2.0)

result_string = tokenizer.decode(sample[0], skip_special_tokens=True)

print(result_string)
#summary = "A new database has been created to help researchers understand how humans communicate across languages"