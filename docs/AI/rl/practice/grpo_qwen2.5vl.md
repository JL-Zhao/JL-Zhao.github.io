---
comments: true
---

!!! abstract "这里是基于 trl 库实现对 Qwen2.5-VL 系列模型 GRPO 强化学习的流程。"

??? warning "下面是我使用的库的版本，可以作为参考信息（~~虽然感觉没啥用~~）"

    Python=3.10.19  
    torch=2.9.0  
    transformers=5.0.0.dev0  
    trl=0.26.0.dev0  

本项目的目标是使用 TRL 库来实现对 Qwen2.5-VL-3B-Instruct（你换成 Qwen2.5-VL 系列的其他模型也行）的 GRPO 强化学习，旨在增强其对图像表格的推理能力。

## 数据准备

为了能够更好的训练模型，我提前用 DeepSeek-v3 在 WTQ 数据集上蒸馏了一批用来训练模型的数据，保存为 `grpo_train.jsonl` 文件，其中每一行的 json 中包含如下字段：

- question_id：问题编号
- image：图像名
- category：类别
- question：具体问题
- table：字典形式的表格
- answer：列表形式的正确答案
- response：DeepSeek-v3 的回答

如果想要提高数据的质量，可以提前使用答案匹配或者任何你认为合适的方式对数据进行过滤或者增强等。

接下来是对数据文件的处理与加载

```python
import re
import json
from datasets import load_dataset
from transformers import AutoProcessor
from PIL import Image

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct") # 分词器加载
def prepare_dataset(data_path):
    dataset = load_dataset('json', data_files=data_path, split='train')
    
    def make_conversation(example):
        answer = example["answer"]
        image = "/path/to/your/image/folder" + example["image"] # 路径转换

        question = example['question']
        
        prompt = INSTRUCTION_TEMPLATE.format(question=question) # 生成提示词
        
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        
        return {
            "prompt": conversation,
            "image": image,
            "solution": answer
        }

    dataset = dataset.map(make_conversation) # 应用上面的数据处理函数
    
    return dataset
```

在数据准备阶段，我们主要是转换数据的格式，以便可以和 trl 库中的使用方式对齐。

## 奖励函数准备

奖励函数是强化学习的重要组成部分，是你期望训练模型需要提升的能力的具体体现点。

对于表格任务的问答，我这里是没有使用原 GRPO 中的格式规范加答案准确性的范式，而是使用了最终答案部分正确的 F1 得分和全部正确的得分。其中部分正确的情况还进行加权，确保得分与全部正确的情况有一定的差距。

```python
def reward_function(completions: list[list[dict[str, str]]], solution: list[str], **kwargs):
    rewards = []
    
    for completion, answer in zip(completions, solution):
        completion = completion[0]['content']
        predict_answer_list = extract_answer_list(completion)
        
        predict_set = {normalize_value(x) for x in predict_answer_list if x is not None}
        gold_set = {normalize_value(x) for x in answer if x is not None}
        
        if not gold_set:
            rewards.append(0.0)
            continue
        if not predict_set:
            rewards.append(0.0)
            continue
        
        intersection = predict_set.intersection(gold_set)
        overlap_count = len(intersection)
        
        if predict_set == gold_set:
            rewards.append(1.0)
            continue
        
        precision = overlap_count / len(predict_set)
        recall = overlap_count / len(gold_set)
        if (precision + recall) == 0:
            f1_score = 0.0
        else:
            f1_score = 2 * (precision * recall) / (precision + recall)
        final_score = f1_score * 0.8
        rewards.append(final_score)
        
    return rewards
```

## 模型训练

准备完了数据集和奖励函数，就可以开始模型的正式训练了。

**1. 首先是加载数据集并且过滤其中与训练无关的部分**

remove_columns() 函数用于移除数据集中的指定列，这里我们移除了与训练无关的列，包括 question_id、category、question、table、answer、markdown_table、response、image。

```python
train_dataset = prepare_dataset("/path/to/your/grpo_train.jsonl")
train_dataset = train_dataset.remove_columns(["question_id", "category", "question", "table", "answer", "markdown_table", "response", "image"])
```

**2. 接着加载基线模型**

```python
from transformers import Qwen2_5_VLForConditionalGeneration

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct", 
    torch_dtype=torch.bfloat16, 
    device_map="auto"
)
```

**3. 配置 LoRA 设置并且应用到模型中**

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], # 这里是你想要微调的组件名
    task_type="CAUSAL_LM",
    bias="none",
    lora_dropout=0.05,
)
model = get_peft_model(model, lora_config)
```

在这一步，你也可以查看自己通过 LoRA 配置之后，需要更新的参数量以及后续修改的参数量占原模型参数量的比例。

```python
model.print_trainable_parameters()
```

**4. GRPO 配置**

```python
from trl import GRPOConfig

training_args = GRPOConfig(
    output_dir="训练参数的输出路径",
    learning_rate="学习率，1e-6 即可",
    remove_unused_columns=False, # 是否保留未使用的列，选择保留，避免报错
    num_train_epochs="训练轮数",
    bf16=True, # 参数精度为 bfloat16
    per_device_train_batch_size="每个设备训练几个批次，根据自己的显卡决定，A100 可以使用 8，4090 建议 2-4",
    max_completion_length="最大输出长度",
    num_generations=8,  # 每个输入生成 8 个输出，这部分需要和 per_device_train_batch_size 对应
    max_prompt_length="输入 prompt 最大长度",
    report_to=["tensorboard"],
    logging_steps=10,
    save_strategy="steps",
    save_steps=4000,
)

```

**5. 正式训练并保存训练后的模型**

```python
from trl import GRPOTrainer

trainer = GRPOTrainer(
    model=model,
    processing_class=processor,
    reward_funcs=[tarpo_manager.reward_function],
    args=training_args,
    train_dataset=train_dataset,
)

print("Starting training with TARPO...")
trainer.train()

trainer.save_model("/path/to/save/model")
print(f"Training finished. Model saved to {CONFIG['output_dir']}")
```

至此，模型的训练就完成了。并且 LoRA 相关参数也已经存储在了 `/path/to/save/model` 目录下。
