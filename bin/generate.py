import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoTokenizer
from torchaudio.transforms import MelSpectrogram

from model.mimo_audio.modeling_mimo_audio import (
    MiMoAudioArguments,
    MiMoAudioForCausalLM,
)
from model.mimo_audio_tokenizer.modeling_audio_tokenizer import MiMoAudioTokenizer
from model.mimo_audio.process_speechdata import InputSegment

from data.dataset import AudioDataSet


SPECIAL_TOKENS = [
    "<|sosp|>",
    "<|eosp|>",
    "<|empty|>",
    "<|Human|>",
    "<|SpeechLM|>",
    "<|sostm|>",
    "<|eostm|>",
    "<|eot|>",
]


@dataclass
class InferDataArguments:
    data_path: str = field(default=None)
    validate_path: str = field(default=None)


def load_model(
    model_path: str,
    audio_tokenizer_path: str,
    checkpoint_path: str | None,
    speech_loss_weights: str,
    device: str,
    use_lora: bool,
):
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    for token in SPECIAL_TOKENS:
        if token not in tokenizer.get_vocab():
            tokenizer.add_tokens([token], special_tokens=True)

    model_args = MiMoAudioArguments(
        model_name_or_path=model_path,
        sosp_idx=tokenizer.convert_tokens_to_ids("<|sosp|>"),
        eosp_idx=tokenizer.convert_tokens_to_ids("<|eosp|>"),
        empty_idx=tokenizer.convert_tokens_to_ids("<|empty|>"),
        sostm_idx=tokenizer.convert_tokens_to_ids("<|sostm|>"),
        eostm_idx=tokenizer.convert_tokens_to_ids("<|eostm|>"),
        eot_idx=tokenizer.convert_tokens_to_ids("<|eot|>"),
        speech_loss_weights=speech_loss_weights,
    )

    # 全量微调时，checkpoint_path 就是模型目录
    load_path = model_path if use_lora or checkpoint_path is None else checkpoint_path

    model = MiMoAudioForCausalLM.from_pretrained(
        load_path,
        args=model_args,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
    )

    if use_lora:
        if checkpoint_path is None:
            raise ValueError("使用 LoRA 时必须提供 --checkpoint_path")

        non_lora_path = Path(checkpoint_path) / "non_lora_trainables.bin"
        if non_lora_path.exists():
            state_dict = torch.load(non_lora_path, map_location="cpu")
            model.load_state_dict(state_dict, strict=False)

        model = PeftModel.from_pretrained(model, checkpoint_path)

    model.eval()
    model.tokenizer = tokenizer

    audio_tokenizer = MiMoAudioTokenizer.from_pretrained(audio_tokenizer_path)
    audio_tokenizer.to(device=device, dtype=torch.bfloat16)
    audio_tokenizer.eval()

    mel_transform = MelSpectrogram(
        sample_rate=audio_tokenizer.config.sampling_rate,
        n_fft=audio_tokenizer.config.nfft,
        hop_length=audio_tokenizer.config.hop_length,
        win_length=audio_tokenizer.config.window_size,
        f_min=audio_tokenizer.config.fmin,
        f_max=audio_tokenizer.config.fmax,
        n_mels=audio_tokenizer.config.n_mels,
        power=1.0,
        center=True,
    )

    return model, tokenizer, audio_tokenizer, mel_transform


def build_infer_dataset_helper(
    tokenizer,
    audio_tokenizer,
    mel_transform,
    model,
    use_lora: bool,
):
    """
    创建一个只用于调用 preprocess_input/get_input_ids 的 dataset helper。
    不真正读取 jsonl。
    """
    helper = AudioDataSet.__new__(AudioDataSet)
    helper.tokenizer = tokenizer
    helper.mimo_audio_tokenizer = audio_tokenizer
    helper.mel_transform = mel_transform
    helper.data_args = InferDataArguments()

    helper.model = model.model if use_lora else model
    helper.speech_zeroemb_idx = helper.model.speech_empty_ids
    helper.ignore_index = -100
    return helper


def build_prompt_input_ids(
    conversation,
    helper: AudioDataSet,
):
    """
    使用与训练 dataset 基本相同的 InputSegment 构造方式。

    conversation 中可以包含 system/user，
    assistant 内容会被忽略，只保留 assistant 开始标记。
    """
    segments = []

    for item in conversation:
        role = item["role"]

        # 推理时不能把参考答案放进 prompt
        if role == "assistant":
            break

        if role not in {"system", "user"}:
            continue

        segments.append(
            InputSegment(
                text=f"<|im_start|>{role}\n",
                speech_zeroemb_idx=helper.speech_zeroemb_idx,
                text_zeroemb_idx=helper.model.args.empty_idx,
            )
        )

        content = item["content"]
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]

        for content_item in content:
            if content_item["type"] == "text":
                segments.append(
                    InputSegment(
                        text=content_item["text"],
                        speech_zeroemb_idx=helper.speech_zeroemb_idx,
                        text_zeroemb_idx=helper.model.args.empty_idx,
                    )
                )
            elif content_item["type"] == "audio":
                speech_tokens = helper.preprocess_input(content_item)
                segments.append(
                    InputSegment(
                        text="",
                        audio=speech_tokens,
                        speech_zeroemb_idx=helper.speech_zeroemb_idx,
                        text_zeroemb_idx=helper.model.args.empty_idx,
                    )
                )

        segments.append(
            InputSegment(
                text="<|im_end|>\n",
                speech_zeroemb_idx=helper.speech_zeroemb_idx,
                text_zeroemb_idx=helper.model.args.empty_idx,
            )
        )

    # 与训练代码中的 assistant 前缀保持一致
    segments.append(
        InputSegment(
            text="<|im_start|>assistant\n<|sostm|>",
            speech_zeroemb_idx=helper.speech_zeroemb_idx,
            text_zeroemb_idx=helper.model.args.empty_idx,
        )
    )

    return helper.get_input_ids(segments)


def decode_text(tokenizer, generated_ids: torch.Tensor) -> str:
    """
    MiMo input_ids 通常第 0 个通道是文本 token，
    其余通道是 speech token。
    """
    if generated_ids.ndim == 3:
        text_ids = generated_ids[0, 0]
    elif generated_ids.ndim == 2:
        text_ids = generated_ids[0]
    else:
        text_ids = generated_ids

    text = tokenizer.decode(text_ids.tolist(), skip_special_tokens=False)

    # 只保留 assistant 生成部分
    marker = "<|im_start|>assistant\n<think>\n\n</think>\n"
    if marker in text:
        text = text.split(marker, 1)[1]

    if "<|im_end|>" in text:
        text = text.split("<|im_end|>", 1)[0]

    return text.strip()


@torch.inference_mode()
def infer_one(
    model,
    tokenizer,
    helper,
    conversation,
    device,
    max_new_tokens,
    temperature,
    top_p,
):
    input_ids = build_prompt_input_ids(conversation, helper)
    input_ids = input_ids.unsqueeze(0).to(device)
    input_ids = input_ids.T.reshape(1, -1) # [B, flattened(T, audio_channels + 1)]
    breakpoint()
    seq_len = input_ids.shape[-1] // helper.model.group_size
    attention_mask = torch.ones(
        (1, seq_len),
        dtype=torch.long,
        device=device,
    )

    generate_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "use_cache": True,
    }

    if temperature > 0:
        generate_kwargs["temperature"] = temperature
        generate_kwargs["top_p"] = top_p

    output = model.generate(**generate_kwargs)

    if hasattr(output, "sequences"):
        output = output.sequences

    return decode_text(tokenizer, output)


def load_json_or_jsonl(path: str):
    path = Path(path)

    if path.suffix == ".jsonl":
        samples = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                obj = json.loads(line)

                # 训练 jsonl 最后一行可能是 seek offset 列表
                if isinstance(obj, list) and obj and all(
                    isinstance(x, int) for x in obj
                ):
                    continue

                samples.append(obj)
        return samples

    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    return obj if isinstance(obj, list) else [obj]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", required=True)
    parser.add_argument("--audio_tokenizer_path", required=True)
    parser.add_argument("--checkpoint_path", default=None)
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", default="infer_result.jsonl")

    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument(
        "--speech_loss_weights",
        default="100-12-8-6-4-2-2-1-1",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.9)

    args = parser.parse_args()

    model, tokenizer, audio_tokenizer, mel_transform = load_model(
        model_path=args.model_path,
        audio_tokenizer_path=args.audio_tokenizer_path,
        checkpoint_path=args.checkpoint_path,
        speech_loss_weights=args.speech_loss_weights,
        device=args.device,
        use_lora=args.use_lora,
    )

    helper = build_infer_dataset_helper(
        tokenizer=tokenizer,
        audio_tokenizer=audio_tokenizer,
        mel_transform=mel_transform,
        model=model,
        use_lora=args.use_lora,
    )

    samples = load_json_or_jsonl(args.input_path)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as fout:
        for index, sample in enumerate(samples):
            # 兼容 {"conversation": [...]} 和直接 [...]
            conversation = sample.get("conversation", sample) if isinstance(sample, dict) else sample

            prediction = infer_one(
                model=model,
                tokenizer=tokenizer,
                helper=helper,
                conversation=conversation,
                device=args.device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )

            result = {
                "index": index,
                "prediction": prediction,
            }

            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            fout.flush()

            print(f"[{index}] {prediction}")


if __name__ == "__main__":
    main()
