#!pip install torch==2.2.1
#!pip install transformers==4.44.0 accelerate
#!pip install swanlab modelscope datasets peft pandas tiktoken

from modelscope import snapshot_download, AutoTokenizer
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
import torch
import pandas as pd

#model_id = "ZhipuAI/glm-4-9b-chat"    
#model_dir = "/kaggle/input/glm4-model/ZhipuAI/glm-4-9b-chat/"

# 在modelscope上下载GLM4模型到本地目录下
#model_dir = snapshot_download(model_id, cache_dir="/kaggle/working", revision="master")
#print("snapshot_download done")

def get_device_map_dict(num_gpus=2):
    device_map_dict = {'transformer.embedding.word_embeddings': 0,
                       'transformer.rotary_pos_emb': 0,
                        'transformer.encoder.final_layernorm': 0,
                        'transformer.output_layer': 0,
                        'transformer.encoder.layers.39': 0}
    num_trans_layers = 39
    gpu_target = 0
    for index in range(num_trans_layers):
        if index % num_gpus != 0:
            gpu_target += 1
        else:
            gpu_target = 0
        device_map_dict[f'transformer.encoder.layers.{index}'] = gpu_target
    print(device_map_dict)
    return device_map_dict

def get_lora_device_map_dict(num_gpus=2):
    device_map_dict = {'base_model.model.transformer.embedding.word_embeddings': 0,
                        'base_model.model.transformer.rotary_pos_emb': 0,
                        'base_model.model.transformer.encoder.final_layernorm': 0,
                        'base_model.model.transformer.output_layer': 0,
                        'base_model.model.transformer.encoder.layers.39': 0}
    num_trans_layers = 39
    gpu_target = 0
    for index in range(num_trans_layers):
        if index % num_gpus != 0:
            gpu_target += 1
        else:
            gpu_target = 0
        device_map_dict[f'base_model.model.transformer.encoder.layers.{index}'] = gpu_target
    print(device_map_dict)
    return device_map_dict

train_jsonl_new_path = "/kaggle/input/glm4-test/llm_train_data.json"
total_df = pd.read_json(train_jsonl_new_path, lines=True)
train_df = total_df[int(len(total_df) * 0.1):]  # 取90%的数据做训练集
test_df = total_df[:int(len(total_df) * 0.1)].sample(n=20)  # 随机取10%的数据中的20条做测试集
#train_df = train_df[:10]
print("train test data split done")


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
#model_id = "/kaggle/input/glm-4/transformers/glm-4-9b-chat/1"
#online_model_id = "THUDM/glm-4-9b-chat"
online_model_id = "/kaggle/input/glm4-model/ZhipuAI/glm-4-9b-chat/"
tokenizer = AutoTokenizer.from_pretrained(online_model_id, use_fast=False, trust_remote_code=True, cache_dir='/kaggle/working/chatglm_cache')
tokenizer.pad_token = tokenizer.eos_token

device_map = get_device_map_dict(num_gpus=2)

model = AutoModelForCausalLM.from_pretrained(
    online_model_id,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    #quantization_config=quantization_config,
    #device_map="auto", 
    device_map = device_map,
    #torch_dtype=torch.float16,
    torch_dtype=torch.bfloat16,
    #cache_dir='/kaggle/working/chatglm_cache'
)

def process_func(example):
    """
    将数据集进行预处理, 处理成模型可以接受的格式
    """

    MAX_LENGTH = 384 
    input_ids, attention_mask, labels = [], [], []
    system_prompt = """你是一个地址实体元素解析领域的专家，你需要从给定的地址文本中提取 兴趣点; 道路; 楼号; 房间号 实体. 以 json 格式输出, 如 {"entity_text": "洪文六里", "entity_label": "道路"} 注意: 1. 输出的每一行都必须是正确的 json 字符串. 2. 找不到任何实体时, 输出"没有找到任何实体". """
    
    instruction = tokenizer(
        f"<|system|>\n{system_prompt}<|endoftext|>\n<|user|>\n{example['input']}<|endoftext|>\n<|assistant|>\n",
        add_special_tokens=False,
    )
    response = tokenizer(f"{example['output']}", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = (
        instruction["attention_mask"] + response["attention_mask"] + [1]
    )
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}   


from datasets import Dataset

train_ds = Dataset.from_pandas(train_df)
train_dataset = train_ds.map(process_func, remove_columns=train_ds.column_names)

from peft import LoraConfig, TaskType, get_peft_model

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["query_key_value", "dense", "dense_h_to_4h", "activation_func", "dense_4h_to_h"],
    inference_mode=False,  # 训练模式
    r=8,  # Lora 秩
    lora_alpha=32,  # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1,  # Dropout 比例
)

model = get_peft_model(model, config)
print("peft model done")

from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, GenerationConfig
import os
os.environ["WANDB_DISABLED"]="true"
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)

#!mkdir /kaggle/working/glm_output
args = TrainingArguments(
    output_dir="/kaggle/working/glm_output",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    logging_strategy="steps",
    logging_steps=5,
    num_train_epochs=2,
    save_strategy="no",
    save_total_limit=1,
    #save_steps=100,
    learning_rate=1e-5,
    save_on_each_node=True,
    gradient_checkpointing=True,
    #load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)

trainer.train()

#trainer.save_model("/kaggle/working/glm_output/best.pth")
peft_model_id = "/kaggle/working/glm_output/best.pth"
trainer.model.save_pretrained(peft_model_id)
tokenizer.save_pretrained(peft_model_id)
print("model save done")

