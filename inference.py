from vllm import SamplingParams
from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel) # patch GRPO Algo

max_seq_length = 1024
lora_rank = 64   

# 从 HuggingFace 加载 Qwen2.5-3B-Instruct 模型
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "meta-llama/meta-Llama-3.1-8B-Instruct",
    max_seq_length = max_seq_length,
    load_in_4bit = True, # False for LoRA 16bit
    fast_inference = True, # Enable vLLM fast inference
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.6, # Reduce if out of memory
)

model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank,            # LoRA秩
    target_modules = [         # 应用LoRA的目标模块
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha = lora_rank,    # LoRA缩放系数
    use_gradient_checkpointing = "unsloth",  # 启用梯度检查点以支持长序列
    random_state = 666,       # 随机种子
)

SYSTEM_PROMPT = """
请使用中文按以下格式回答问题:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""


def test_ruozhi(prompt):
  text = tokenizer.apply_chat_template([
    {"role" : "system", "content" : SYSTEM_PROMPT},
    {"role" : "user", "content" : prompt},
  ], tokenize = False, add_generation_prompt = True)


  sampling_params = SamplingParams(
    temperature = 0.8,  # 生成温度（越高越随机）
    top_p = 0.95,   # 核采样阈值
    max_tokens = 1024, # 最大生成token数
  )
  output = model.fast_generate(
    text,
    sampling_params = sampling_params,
    lora_request = model.load_lora("grpo_saved_lora"),
  )[0].outputs[0].text
  print("\n")
  print(output)