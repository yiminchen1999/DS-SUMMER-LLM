from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-1b1")
model = AutoModelForCausalLM.from_pretrained("bigscience/bloomz-1b1")

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# English
input_text = "biostatistics is "
num_sequences = 1

for _ in range(num_sequences):
    generated = generator(input_text, max_length=20, num_return_sequences=1) #ValueError: num_return_sequences has to be 1, but is 4 when doing greedy search.
    print(generated[0]['generated_text'])

#"greedy" for more accurate completion e.g. math/history/translations (but which may be repetitive/less inventive)
# "sampling" for more imaginative completions e.g. story/poetry (but which may be less accurate)
# Chinese
input_text = "生物统计学是一门"

for _ in range(num_sequences):
    generated = generator(input_text, max_length=20, num_return_sequences=1)
    print(generated[0]['generated_text'])
#对BLOOM预训练之后，我们应用相同的大规模多任务微调，使BLOOM具有多语言zero-shot任务泛化能力。我们称得到的模型为BLOOMZ。为了训练BLOOMZ，我们扩展了P3来包含非英语中新数据集和新任务，例如翻译。这产生了xP3，它是83个数据集的提升集合，覆盖46种语言和16中任务。正如上图4所述，xP3反映了ROOTS的语言分布。xP3中的任务包含跨语言和单语言。我们使用PromptSource来收集这些prompts，为prompt添加额外的元数据，例如输入和目标语言。为了研究多语言prompt的重要性，我们还将xP3中的英语提示用机器翻译为相应的数据集语言，来生成一个称为xP3mt的集合。
