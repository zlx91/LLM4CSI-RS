from transformers import AutoTokenizer, AutoModel
from peft import PeftModel
import torch
import numpy as np
import os
import torch.nn as nn
import json
from tqdm import tqdm

# 设置 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ===================== 固定标签（与训练时完全一致） =====================
ALL_LABELS = ["1", "2", "4", "6", "7", "9", "11", "13", "14", "15", "16", "17", "19", "20",
              "21", "22", "24", "25", "26", "27", "28", "30", "32", "33",  "34", "35", "36", "37", "39", "40"]
ID2LABEL = {idx: label for idx, label in enumerate(ALL_LABELS)}


# ===================== 模型结构（均值池化） =====================
class ReferencePositionModel(nn.Module):
    def __init__(self, base_model_path):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(
            base_model_path,
            dtype=torch.bfloat16,
            trust_remote_code=True
        )
        self.classifier = nn.Linear(self.backbone.config.hidden_size, len(ALL_LABELS), dtype=torch.bfloat16)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        pooled_output = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1)
        pooled_output = pooled_output.to(torch.bfloat16)
        logits = self.classifier(pooled_output)
        return logits

# ===================== 数据预处理（原始数值序列，一位小数） =====================
def preprocess_for_inference(rsrp_str):
    try:
        rsrp_values = [float(x) for x in rsrp_str.split(',')]
    except:
        return "RSRP sequence: -120.0"
    rsrp_str_vals = [f"{v:.2f}" for v in rsrp_values]
    return "RSRP sequence: " + " ".join(rsrp_str_vals)


# ===================== 批处理预测（带温度缩放） =====================
def predict_batch(prompts, tokenizer, model, device, top_k=4, temperature=1.5):
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)

    with torch.no_grad():
        model.eval()
        logits = model(**inputs)
        scaled_logits = logits / temperature          # 温度缩放，使概率分布更平滑
        probs = torch.nn.functional.softmax(scaled_logits.float(), dim=-1).cpu().numpy()

    all_results = []
    for i in range(len(prompts)):
        prob = probs[i]
        top_indices = np.argsort(prob)[-top_k:][::-1]
        all_results.append([(ID2LABEL[idx], round(float(prob[idx]), 4)) for idx in top_indices])

    return all_results


# ===================== 主推理与保存流程（输出加权融合所需格式） =====================
def run_inference_and_save(data_list, tokenizer, model, device, json_path="result.json", batch_size=32):
    total = len(data_list)
    print(f"?? 开始推理，总样本数：{total}")

    with open(json_path, "w", encoding="utf-8") as f:
        for i in tqdm(range(0, total, batch_size), desc="推理进度", total=(total + batch_size - 1) // batch_size,
                      colour="blue"):
            batch_data = data_list[i: i + batch_size]

            prompts = []
            valid_indices = []   # 存储全局样本索引

            for idx_in_batch, item in enumerate(batch_data):
                if "input" in item:
                    prompt = preprocess_for_inference(item["input"])
                    prompts.append(prompt)
                    valid_indices.append(i + idx_in_batch)   # 全局索引

            if prompts:
                batch_results = predict_batch(prompts, tokenizer, model, device, top_k=4, temperature=1.5)

                for global_idx, top4 in zip(valid_indices, batch_results):
                    top4_str = " ".join([f"{label}({prob})" for label, prob in top4])
                    true_label = data_list[global_idx].get("output", "unknown")
                    out_record = {
                        "sample_index": global_idx,
                        "true_label": true_label,
                        "top4_prediction": top4_str
                    }
                    f.write(json.dumps(out_record, ensure_ascii=False) + "\n")

    print(f"\n? 推理完成！结果已保存到：{json_path}")


# ===================== 主程序 =====================
if __name__ == "__main__":
    base_model_path = "/home/zlx/models/Llama-3.2-1B-Instruct"
    lora_model_path = "/home/zlx/LLaMA-Factory/saves/lorac/Llama-3.2-1B-Instruct_(8,32)"
    test_data_file = "/home/zlx/LLaMA-Factory/data/test.json"

    device = torch.device("cuda")

    # 1. 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 2. 加载模型
    print("正在加载模型...")
    model = ReferencePositionModel(base_model_path)
    model.backbone = PeftModel.from_pretrained(model.backbone, lora_model_path).merge_and_unload()
    model.classifier.load_state_dict(torch.load(f"{lora_model_path}/classifier.pth", map_location="cpu"))
    model = model.to(device).eval()
    print(" 模型加载完成！")

    # 3. 加载测试数据（JSON 数组）
    print(" 正在加载测试数据...")
    with open(test_data_file, 'r', encoding='utf-8') as f:
        test_data_list = json.load(f)
    print(f"数据加载完成，共 {len(test_data_list)} 条")

    # 4. 开始推理
    run_inference_and_save(test_data_list, tokenizer, model, device, json_path="result.json", batch_size=32)