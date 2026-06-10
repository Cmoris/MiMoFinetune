import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any

import torch
import torch.nn.functional as F
import transformers
from torch import nn
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    logging,
)


from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# TODO: 改成你的 MiMo 文件名
# 例如你的第一个代码文件如果叫 model/mimo.py，就写：
# from model.mimo import MiMoAudioForCausalLM, MiMoAudioConfig, MiMoAudioArguments
from mimo_model import MiMoAudioForCausalLM, MiMoAudioConfig, MiMoAudioArguments

# TODO: 换成你的 MiMo 数据集
# 这个 dataset 应该返回 input_ids / labels / attention_mask
from data.mimo_dataset import MiMoStreamingDataset


logger = logging.get_logger(__name__)
local_rank = 0


def rank0_print(*args):
    if local_rank in [0, -1]:
        print(*args)


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


@dataclass
class DataArguments:
    annotation_dir: str = field(default="")
    token_root: str = field(default="")
    max_length: int = field(default=2048)

    # MiMo 一般不是直接吃 wav，而是吃离散 speech token
    # 如果你仍然从 wav 开始，需要在 dataset 里先用 codec/tokenizer 转成 speech token
    query: Optional[str] = field(default=None)


@dataclass
class ModelArguments:
    pretrained_model_name_or_path: str = field(default="")

    # MiMoAudioForCausalLM 初始化需要这些特殊 token id
    # 这里建议你和原始 checkpoint 的 tokenizer/config 保持一致
    sosp_idx: int = field(default=0)
    eosp_idx: int = field(default=1)
    sostm_idx: int = field(default=2)
    eostm_idx: int = field(default=3)
    eot_idx: int = field(default=4)
    empty_idx: int = field(default=5)

    freeze_modules: Optional[List[str]] = field(default=None)

    # 如果你要额外加入对话事件 token
    add_special_tokens: str = field(
        default="",
        metadata={
            "help": "Comma-separated extra tokens, e.g. <ts>,<te>,<speaker_A>,</speaker_A>"
        },
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    output_dir: str = field(default="./output/mimo_finetuned")
    overwrite_output_dir: bool = field(default=True)

    model_max_length: int = field(default=2048)

    bits: Optional[int] = field(default=None)
    double_quant: bool = field(default=True)
    quant_type: str = field(default="nf4")

    lora_enable: bool = field(default=False)
    lora_r: int = field(default=16)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.05)
    lora_bias: str = field(default="none")

    # MiMo 里建议至少覆盖 global transformer 的 q/k/v/o
    # 如果你也想训 local transformer，可以加 local_transformer 相关模块
    target_modules: str = field(default="q_proj,k_proj,v_proj,o_proj")

    # loss 权重
    text_loss_weight: float = field(default=1.0)
    speech_loss_weight: float = field(default=1.0)

    # 是否训练 speech local transformer 的 audio-token loss
    train_speech_loss: bool = field(default=True)


def normalize_mimo_ids(
    input_ids: torch.Tensor,
    audio_channels: int,
    group_size: int,
) -> torch.Tensor:
    """
    支持两种输入：
    1. [B, C+1, T*group_size]
    2. [B, T*group_size*(C+1)] flatten format

    统一转成 [B, C+1, T*group_size]
    """
    if input_ids.dim() == 3:
        return input_ids

    if input_ids.dim() != 2:
        raise ValueError(f"Unexpected input_ids shape: {input_ids.shape}")

    B, L = input_ids.shape
    step = (audio_channels + 1) * group_size

    if L % step != 0:
        raise ValueError(
            f"Flatten input length {L} is not divisible by "
            f"(audio_channels + 1) * group_size = {step}"
        )

    # flat: [B, T, group_size*(C+1)]
    # -> [B, T, group_size, C+1]
    # -> [B, C+1, T*group_size]
    ids = input_ids.view(B, -1, group_size, audio_channels + 1)
    ids = ids.permute(0, 3, 1, 2).contiguous()
    ids = ids.view(B, audio_channels + 1, -1)
    return ids


def build_model_and_tokenizer(
    model_args: ModelArguments,
    training_args: TrainingArguments,
):
    compute_dtype = (
        torch.float16
        if training_args.fp16
        else torch.bfloat16
        if training_args.bf16
        else torch.float32
    )

    bnb_model_from_pretrained_args = {}

    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig

        bnb_model_from_pretrained_args.update(
            dict(
                device_map={"": training_args.device}
                if training_args.device
                else "auto",
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=training_args.bits == 4,
                    load_in_8bit=training_args.bits == 8,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=training_args.double_quant,
                    bnb_4bit_quant_type=training_args.quant_type,
                    bnb_4bit_quant_storage=compute_dtype,
                ),
            )
        )

    rank0_print(f"Loading tokenizer from {model_args.pretrained_model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.pretrained_model_name_or_path,
        padding_side="right",
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    extra_tokens = []
    if model_args.add_special_tokens:
        extra_tokens = [
            x.strip()
            for x in model_args.add_special_tokens.split(",")
            if x.strip()
        ]

    if extra_tokens:
        rank0_print(f"Adding extra text tokens: {extra_tokens}")
        tokenizer.add_tokens(extra_tokens, special_tokens=False)

    mimo_audio_args = MiMoAudioArguments(
        model_name_or_path=model_args.pretrained_model_name_or_path,
        sosp_idx=model_args.sosp_idx,
        eosp_idx=model_args.eosp_idx,
        sostm_idx=model_args.sostm_idx,
        eostm_idx=model_args.eostm_idx,
        eot_idx=model_args.eot_idx,
        empty_idx=model_args.empty_idx,
    )

    rank0_print(f"Loading MiMo model from {model_args.pretrained_model_name_or_path}")

    model = MiMoAudioForCausalLM.from_pretrained(
        model_args.pretrained_model_name_or_path,
        args=mimo_audio_args,
        torch_dtype=compute_dtype,
        trust_remote_code=True,
        **bnb_model_from_pretrained_args,
    )

    if extra_tokens:
        model.resize_token_embeddings(len(tokenizer))

    if training_args.bits in [4, 8]:
        model.config.torch_dtype = compute_dtype
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=training_args.gradient_checkpointing,
        )

    if training_args.lora_enable:
        target_modules = [
            x.strip()
            for x in training_args.target_modules.split(",")
            if x.strip()
        ]

        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )

        rank0_print(f"Adding LoRA adapters to: {target_modules}")
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    if model_args.freeze_modules is not None:
        rank0_print(f"Freezing modules: {model_args.freeze_modules}")

        named_params = (
            model.base_model.model.named_parameters()
            if training_args.lora_enable and hasattr(model, "base_model")
            else model.named_parameters()
        )

        for name, param in named_params:
            if any(name.startswith(m) for m in model_args.freeze_modules):
                param.requires_grad = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(
                make_inputs_require_grad
            )

    return model, tokenizer


def train():
    global local_rank

    set_seed(42)

    parser = HfArgumentParser(
        (TrainingArguments, ModelArguments, DataArguments)
    )
    training_args, model_args, data_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank

    model, tokenizer = build_model_and_tokenizer(
        model_args=model_args,
        training_args=training_args,
    )

    rank0_print("Preparing MiMo dataset...")

    annotation_paths = [
        str(x)
        for x in Path(data_args.annotation_dir).glob("*.jsonl")
    ]

    train_dataset = MiMoStreamingDataset(
        annotation_paths=annotation_paths,
        tokenizer=tokenizer,
        token_root=data_args.token_root,
        max_length=data_args.max_length,
        group_size=model.config.group_size,
        audio_channels=model.config.audio_channels,
        empty_idx=model.args.empty_idx,
        query=data_args.query,
    )

    rank0_print("Starting training...")

    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    trainer = MiMoTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=train_dataset.data_collator,
    )

    checkpoint_dir = Path(training_args.output_dir)
    checkpoints = list(checkpoint_dir.glob("checkpoint-*"))

    if checkpoints:
        rank0_print("Resuming from checkpoint...")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    rank0_print("Saving final model...")
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train()