import datetime
import json
import os
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments, DataCollatorWithPadding
from peft import LoraConfig, get_peft_model
import gc
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import load_dataset

# ===================== 固定标签（请根据实际数据修改） =====================
# 注意：请确保 ALL_LABELS 中包含数据集中出现的所有标签，例如 "3", "8", "10" 等
ALL_LABELS = ["1", "2", "4", "6", "7", "9", "11", "13", "14", "15", "16", "17", "19", "20",
              "21", "22", "24", "25", "26", "27", "28", "30", "32", "33",  "34", "35", "36", "37", "39", "40"]
LABEL2ID = {label: idx for idx, label in enumerate(ALL_LABELS)}
ID2LABEL = {idx: label for idx, label in enumerate(ALL_LABELS)}
NUM_CLASSES = len(ALL_LABELS)


# ===================== LoRA 配置 =====================
def apply_lora(model):
    config = LoraConfig(
        task_type="FEATURE_EXTRACTION",
        r=8,
        lora_alpha=32,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        lora_dropout=0.05,
        bias="none"
    )
    return get_peft_model(model, config)
# ===================== 均值池化 =====================
class ReferencePositionModel(nn.Module):
    def __init__(self, base_model_path):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(
            base_model_path,
            dtype=torch.bfloat16,
            trust_remote_code=True
        )
        self.classifier = nn.Linear(self.backbone.config.hidden_size, NUM_CLASSES, dtype=torch.bfloat16)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden)
        mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        pooled_output = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1)
        pooled_output = pooled_output.to(torch.bfloat16)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(label_smoothing=0.1)
            loss = loss_fct(logits.view(-1, NUM_CLASSES), labels.view(-1))

        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}


# ===================== 数据处理 =====================
def get_train_data(data_file, tokenizer):
    dataset = load_dataset("json", data_files=data_file)["train"]
    train_valid_split = dataset.train_test_split(test_size=0.2, seed=42)
    print(f"all: {len(dataset)}")  
    def preprocess_function(examples):
        processed_inputs = []
        for rsrp_str, label in zip(examples["input"], examples["output"]):
            try:
                rsrp_values = [float(x) for x in rsrp_str.split(',')]
            except:
                rsrp_values = [0.0] * 110

            rsrp_str_vals = [f"{v:.2f}" for v in rsrp_values]
            input_text = "RSRP sequence: " + " ".join(rsrp_str_vals)
            processed_inputs.append(input_text)

        tokenized_inputs = tokenizer(
            processed_inputs,
            padding=False,
            truncation=True,
            max_length=512   # 足够的长度容纳110个数值
        )
        tokenized_inputs["labels"] = [LABEL2ID[label] for label in examples["output"]]
        return tokenized_inputs

    tokenized_dataset = train_valid_split.map(
        preprocess_function, batched=True, remove_columns=["input", "output"]
    )
    print(f"train: {len(tokenized_dataset['train'])}")  # 输出: 12000
    print(f"val: {len(tokenized_dataset['test'])}")  # 输出: 3000
    return tokenized_dataset


# ===================== 评估指标 =====================
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = predictions.argmax(axis=-1)
    accuracy = accuracy_score(labels, preds)
    if preds.shape[0] == 0:
        return {"accuracy": 0.0}
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


# ===================== 显存释放 =====================
def release_gpu_memory():
    torch.cuda.empty_cache()
    gc.collect()


# ===================== 主训练流程 =====================
def main():
    data_file = "/home/zlx/LLaMA-Factory/data/train.json"
    base_model_path = "/home/zlx/models/Llama-3.2-3B-Instruct"
    output_lora_path = f"/home/zlx/LLaMA-Factory/saves/lorac/Llama-3.2-3B-Instruct_(8,32)_o"
    os.makedirs(output_lora_path, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 加载数据
    tokenized_dataset = get_train_data(data_file, tokenizer)

    # 初始化模型
    model = ReferencePositionModel(base_model_path)
    model.backbone = apply_lora(model.backbone)
    model = model.to("cuda")
    model.backbone.print_trainable_parameters()

    # 数据整理器（动态padding）
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=output_lora_path,
        eval_strategy="epoch",
        learning_rate=2e-4,
        weight_decay=0.01,
        max_grad_norm=1.0,
        optim="adamw_torch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        warmup_ratio=0.05,
        save_strategy="epoch",
        save_total_limit=2,
        logging_steps=20,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        bf16=True,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )

    print("Start training on GPU")
    trainer.train()

    # 保存模型
    model.backbone.save_pretrained(output_lora_path)
    tokenizer.save_pretrained(output_lora_path)
    torch.save(model.classifier.state_dict(), os.path.join(output_lora_path, "classifier.pth"))
    print(f"? Model saved: {output_lora_path}")

    release_gpu_memory()


if __name__ == "__main__":
    main()