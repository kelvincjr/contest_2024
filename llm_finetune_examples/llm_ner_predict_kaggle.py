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

def get_test_data():
    df = pd.read_csv('/kaggle/input/glm4-test/test_data.csv', header=0)
    data = []
    data_count = 0
    for i in range(len(df)):
        id_ = df.iloc[i].at['id']
        u_address = df.iloc[i].at['N_standard_address']
        data.append((id_,u_address))
        data_count += 1
    return data

#GLM模型加载
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

model = model.eval()

#训练好的LORA模型加载
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
lora_dir = "/kaggle/input/dc-peft-model/"
device_map = get_lora_device_map_dict(num_gpus=2)
model = PeftModel.from_pretrained(model, lora_dir, device_map=device_map, torch_type=torch.float16)
#model = PeftModel.from_pretrained(model, model_id=lora_dir)
print("peft model load done")

#读测试数据
test_data = get_test_data()

def predict(messages, model, tokenizer):
    device = "cuda"

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    #with torch.no_grad():
    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=384, do_sample=True, temperature=0.1)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response

#input_text = "洪文村洪文五里75号\3545\莲前洪文村洪文社区|莲前|103室"
import json
json_data = []
for item in test_data:
    #input_text = "莲前西路怡富花园一、二期258#401室"
    id_=item[0]
    if id_ < 1800:
        continue
    input_text = item[1]
    test_texts = {
        "instruction": """你是一个地址实体元素解析领域的专家，你需要从给定的地址文本中提取 兴趣点; 道路; 楼号; 房间号 实体. 以 json 格式输出, 如 {"entity_text": "洪文六里", "entity_label": "道路"} 注意: 1. 输出的每一行都必须是正确的 json 字符串. 2. 找不到任何实体时, 输出"没有找到任何实体". """,
        "input": f"文本:{input_text}"
    }

    instruction = test_texts['instruction']
    input_value = test_texts['input']

    messages = [
        {"role": "system", "content": f"{instruction}"},
        {"role": "user", "content": f"{input_value}"}
    ]

    response = predict(messages, model, tokenizer)
    print("id: {}, u_address: {}".format(id_,input_text))
    print(response)
    
    entity_items = response.split("}")
    entities = []
    for entity_item in entity_items:
        entity_item = entity_item.strip()
        #print("entity_item: ", entity_item)
        if entity_item.startswith("{") and entity_item.endswith("\""):
            try:
                entity = json.loads(entity_item + "}")
                entities.append(entity)
            except:
                pass
    json_item = {"id":str(id_), "u_address": input_text, "entities":entities}
    json_data.append(json_item)
    if id_ == 2000:
        print("{} items processed".format(id_))
        with open("/kaggle/working/test_data_entities_{}.json".format(id_), "w", encoding="utf-8") as f:
            for json_item in json_data:
                f.write(json.dumps(json_item, ensure_ascii=False) + "\n")
        json_data = []
        print("{} items save done".format(id_))
        break

