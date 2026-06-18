import os
import json
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm import load
from mlx.utils import tree_flatten
import time
import csv
import numpy as np
import random
from pathlib import Path

# Get path relative to the script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_PATH = os.path.normpath(os.path.join(SCRIPT_DIR, "../data/train_chat.jsonl"))
DEFAULT_OUTPUT_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "../lora_trading_policy_mlx"))
DEFAULT_LOG_PATH = os.path.normpath(os.path.join(SCRIPT_DIR, "../training_log.csv"))

# 1. LoRA 레이어 정의
class LoRALinear(nn.Module):
    def __init__(self, quantized_module, rank=16, alpha=32):
        super().__init__()
        self.quantized_module = quantized_module
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        el_per_int = 32 // quantized_module.bits
        in_features = quantized_module.weight.shape[1] * el_per_int
        out_features = quantized_module.weight.shape[0]
        
        self.lora_a = mx.random.normal((in_features, rank)) * 0.01
        self.lora_b = mx.zeros((rank, out_features))
        
    def __call__(self, x):
        y = self.quantized_module(x)
        # LoRA 경로는 float32로 계산
        y += (x.astype(mx.float32) @ self.lora_a @ self.lora_b) * self.scaling
        return y

# 2. 공식 스타일 데이터 배치 반복자 (Raw 및 Chat 포맷 동시 지원)
def iterate_batches(dataset, tokenizer, batch_size, max_tokens):
    while True:
        indices = np.random.permutation(len(dataset))
        for i in range(0, len(indices) - batch_size + 1, batch_size):
            batch = []
            for j in range(batch_size):
                sample = dataset[indices[i + j]]
                if "messages" in sample:
                    # 토크나이저의 apply_chat_template을 사용하여 공식 템플릿 적용
                    text = tokenizer.apply_chat_template(
                        sample["messages"],
                        tokenize=False,
                        add_generation_prompt=False
                    )
                else:
                    text = (
                        "### Instruction:\n" f"{sample['instruction']}\n\n"
                        "### Input:\n" f"{json.dumps(sample['input'], ensure_ascii=False)}\n\n"
                        "### Output:\n" f"{json.dumps(sample['output'], ensure_ascii=False)}"
                    )
                tokens = tokenizer.encode(text)
                if len(tokens) > max_tokens:
                    tokens = tokens[:max_tokens]
                batch.append(tokens)

            lengths = [len(x) for x in batch]
            max_len = max(lengths)
            
            batch_arr = np.zeros((batch_size, max_len), np.int32)
            for j, tokens in enumerate(batch):
                batch_arr[j, :lengths[j]] = tokens
            
            batch_mx = mx.array(batch_arr)
            yield batch_mx[:, :-1], batch_mx[:, 1:], mx.array(lengths)

# 3. 수정된 Loss 함수 (params를 첫 번째 인자로 받음)
def loss_fn(params, model, x, y, lengths):
    model.update(params) # 가중치 업데이트 반영
    outputs = model(x)
    # 반환값이 1개일 때와 2개일 때 모두 대응
    logits = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
    logits = logits.astype(mx.float32) 

    # 차원 정렬
    if logits.ndim == 2:
        logits = logits.reshape(x.shape[0], x.shape[1], -1)
    elif logits.shape[0] != x.shape[0]:
        logits = logits.transpose(1, 0, 2)

    length_mask = mx.arange(x.shape[1])[None, :] < lengths[:, None]
    ce = nn.losses.cross_entropy(logits, y) * length_mask
    return ce.sum() / length_mask.sum()

# 4. LoRA 적용 함수 (unfreeze 포함)
def apply_lora(model, config):
    applied_count = 0
    model.freeze() # 일단 전체 고정
    
    num_layers = len(model.model.layers)
    start_layer = max(0, num_layers - config["lora_layers"])
    
    for i in range(start_layer, num_layers):
        layer = model.model.layers[i]
        for name in config["target_modules"]:
            if hasattr(layer.self_attn, name):
                mod = getattr(layer.self_attn, name)
                # LoRA 교체
                lora_mod = LoRALinear(mod, config["lora_rank"], config["lora_alpha"])
                setattr(layer.self_attn, name, lora_mod)
                lora_mod.unfreeze() # 학습 가능하도록 명시적 해제
                applied_count += 1
    print(f"Total LoRA layers applied & unfrozen: {applied_count}")
    return model

# 5. 메인 학습 루프
def train_model(model, tokenizer, config):
    print("\n=== STARTING TRAINING ===\n")
    
    # [핵심 수정] 학습 가능 파라미터 추출
    trainable_params = model.trainable_parameters()
    if not trainable_params:
        raise ValueError("No trainable parameters found. Check apply_lora unfreeze logic.")

    with open(config["data_path"], 'r') as f:
        samples = [json.loads(line) for line in f]
    
    optimizer = optim.AdamW(learning_rate=config["learning_rate"])
    
    # [핵심 수정] 미분 함수 정의 (첫 인자인 params에 대해 미분)
    vg_fn = nn.value_and_grad(model, loss_fn)

    log_file = open(config["log_path"], 'w', newline='')
    log_writer = csv.writer(log_file)
    log_writer.writerow(['step', 'loss'])

    start_time = time.time()
    batch_iter = iterate_batches(samples, tokenizer, config["batch_size"], config["max_tokens"])
    
    for step in range(config["iters"]):
        try:
            inputs, targets, lengths = next(batch_iter)
            
            # [핵심 수정] trainable_params를 명시적으로 전달
            (loss, _), grads = vg_fn(trainable_params, model, inputs, targets, lengths)
            
            # 업데이트 시에도 trainable_params 기준
            optimizer.update(trainable_params, grads)
            model.update(trainable_params)
            
            # LoRA 파라미터와 loss만 평가 (uint32 에러 방지)
            mx.eval(trainable_params, optimizer.state, loss)
            
            if (step + 1) % 10 == 0:
                print(f"[Step {step+1}] Loss: {loss.item():.4f}, Time: {time.time()-start_time:.1f}s")
                log_writer.writerow([step + 1, loss.item()])
                log_file.flush()

            if (step + 1) % 10 == 0:
                mx.clear_cache() # MLX의 미사용 메모리 즉시 해제
                
            if (step + 1) % 100 == 0:
                # 더 강력한 동기화로 메모리 파편화 방지
                mx.eval(trainable_params)

        except Exception as e:
            print(f"Error at step {step + 1}: {e}")
            continue

    log_file.close()
    
    save_path = Path(config["output_dir"]) / "adapters.npz"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    mx.savez(str(save_path), **dict(tree_flatten(model.trainable_parameters())))
    print(f"Saved adapters to {save_path}")
    
    return model

if __name__ == "__main__":
    config = {
        "model_path": "mlx-community/gpt-oss-20b-MXFP4-Q8",
        "data_path": DEFAULT_DATA_PATH,
        "output_dir": DEFAULT_OUTPUT_DIR,
        "batch_size": 1, 
        "learning_rate": 1e-5,
        "iters": 1000,
        "max_tokens": 64,
        "lora_rank": 4,
        "lora_alpha": 8,
        "lora_layers": 2,
        "target_modules": ["q_proj", "v_proj"],
        "log_path": DEFAULT_LOG_PATH
    }

    model, tokenizer = load(config["model_path"])
    model = apply_lora(model, config)
    train_model(model, tokenizer, config)