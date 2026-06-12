import argparse
import json
from pathlib import Path

import torch
import torchaudio
from peft import PeftModel
from transformers import AutoTokenizer
from torchaudio.transforms import MelSpectrogram

from model.mimo_audio.modeling_mimo_audio import (
    MiMoAudioArguments,
    MiMoAudioForCausalLM,
)
from model.mimo_audio_tokenizer.modeling_audio_tokenizer import MiMoAudioTokenizer
from model.mimo_audio.process_speechdata import InputSegment

# 复用训练 dataset 中的音频编码和 MiMo input_ids 构造逻辑
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


def load_model(
    model_path,
    audio_tokenizer_path,
    checkpoint_path,
    speech_loss_weights,
    device,
    use_lora,
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

    load_path = (
        model_path
        if use_lora or checkpoint_path is None
        else checkpoint_path
    )

    model = MiMoAudioForCausalLM.from_pretrained(
        load_path,
        args=model_args,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
    )

    if use_lora:
        if checkpoint_path is None:
            raise ValueError("LoRA 推理必须提供 --checkpoint_path")

        non_lora_path = Path(checkpoint_path) / "non_lora_trainables.bin"
        if non_lora_path.exists():
            state_dict = torch.load(non_lora_path, map_location="cpu")
            model.load_state_dict(state_dict, strict=False)

        model = PeftModel.from_pretrained(model, checkpoint_path)

    model.eval()
    model.tokenizer = tokenizer

    audio_tokenizer = MiMoAudioTokenizer.from_pretrained(
        audio_tokenizer_path
    )
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


def make_dataset_helper(
    tokenizer,
    audio_tokenizer,
    mel_transform,
    model,
    use_lora,
):
    """
    不读取训练集，只复用 AudioDataSet 中的：
      preprocess_input()
      get_input_ids()
      encode_batch()
    """
    helper = AudioDataSet.__new__(AudioDataSet)
    helper.tokenizer = tokenizer
    helper.mimo_audio_tokenizer = audio_tokenizer
    helper.mel_transform = mel_transform
    helper.model = model.model if use_lora else model
    helper.speech_zeroemb_idx = helper.model.speech_empty_ids
    helper.ignore_index = -100
    return helper


def load_audio(audio_path, target_sr):
    wav, sr = torchaudio.load(audio_path)

    if wav.ndim == 2:
        wav = wav.mean(dim=0)

    if sr != target_sr:
        wav = torchaudio.functional.resample(
            wav,
            orig_freq=sr,
            new_freq=target_sr,
        )

    return wav


def split_equal_chunks(wav, sample_rate, chunk_seconds):
    chunk_samples = int(round(sample_rate * chunk_seconds))

    if chunk_samples <= 0:
        raise ValueError("--chunk_seconds 必须大于 0")

    for start in range(0, wav.numel(), chunk_samples):
        chunk = wav[start:start + chunk_samples]

        if chunk.numel() == 0:
            continue

        yield start / sample_rate, chunk


def make_segment(helper, text="", audio=None):
    return InputSegment(
        text=text,
        audio=audio,
        speech_zeroemb_idx=helper.speech_zeroemb_idx,
        text_zeroemb_idx=helper.model.args.empty_idx,
    )


def build_chunk_prompt(helper, wav_chunk):
    """
    每个 chunk 都按照训练时的一轮 user -> assistant 构造：

      <|im_start|>user
      [audio chunk]
      <|im_end|>
      <|im_start|>assistant
      <think>

      </think>

    assistant 的真实答案不放入输入，由模型生成。
    """
    speech_tokens = helper.preprocess_input(wav_chunk)

    segments = [
        make_segment(helper, text="<|im_start|>user\n"),
        make_segment(helper, audio=speech_tokens),
        make_segment(helper, text="<|im_end|>\n"),
        make_segment(
            helper,
            text="<|im_start|>assistant\n<think>\n\n</think>\n",
        ),
    ]

    return helper.get_input_ids(segments)


def get_generated_suffix(sequences, prompt_length):
    """
    返回 generate() 新生成的部分。
    兼容 [B, C, L] 和 [B, L]。
    """
    if sequences.ndim == 3:
        return sequences[:, :, prompt_length:]

    if sequences.ndim == 2:
        return sequences[:, prompt_length:]

    raise ValueError(
        f"Unexpected sequences shape: {tuple(sequences.shape)}"
    )


def decode_text(tokenizer, generated_ids):
    # MiMo 多通道 token 的第 0 通道是文本 token
    if generated_ids.ndim == 3:
        text_ids = generated_ids[0, 0]
    elif generated_ids.ndim == 2:
        text_ids = generated_ids[0]
    else:
        text_ids = generated_ids

    text = tokenizer.decode(
        text_ids.detach().cpu().tolist(),
        skip_special_tokens=False,
    )

    if "<|im_end|>" in text:
        text = text.split("<|im_end|>", 1)[0]

    return text.strip()


@torch.inference_mode()
def streaming_infer(
    model,
    tokenizer,
    helper,
    audio_path,
    chunk_seconds,
    max_new_tokens,
    temperature,
    top_p,
    device,
):
    target_sr = helper.mimo_audio_tokenizer.config.sampling_rate
    wav = load_audio(audio_path, target_sr)

    past_key_values = None
    total_group_length = 0
    results = []

    for chunk_index, (start_time, wav_chunk) in enumerate(
        split_equal_chunks(wav, target_sr, chunk_seconds)
    ):
        new_input_ids = build_chunk_prompt(helper, wav_chunk)
        new_input_ids = new_input_ids.unsqueeze(0).to(device)

        # MiMo 的真实时间长度按 group_size 计算
        new_group_length = (
            new_input_ids.shape[-1] // helper.model.group_size
        )
        total_group_length += new_group_length

        # 有 KV cache 时，attention_mask 要覆盖：
        # 历史 cache + 当前 chunk
        attention_mask = torch.ones(
            (1, total_group_length),
            dtype=torch.long,
            device=device,
        )

        prompt_length = new_input_ids.shape[-1]

        generate_kwargs = {
            "input_ids": new_input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "use_cache": True,
            "return_dict_in_generate": True,
        }

        if temperature > 0:
            generate_kwargs["temperature"] = temperature
            generate_kwargs["top_p"] = top_p

        output = model.generate(**generate_kwargs)

        sequences = (
            output.sequences
            if hasattr(output, "sequences")
            else output
        )

        generated_ids = get_generated_suffix(
            sequences,
            prompt_length=prompt_length,
        )
        prediction = decode_text(tokenizer, generated_ids)

        # generate 后的 cache 已经包含：
        # 历史内容 + 当前 chunk prompt + 当前生成结果
        past_key_values = getattr(output, "past_key_values", None)

        if past_key_values is None:
            raise RuntimeError(
                "model.generate() 没有返回 past_key_values。"
                "请确认 MiMoAudioForCausalLM.generate 支持 "
                "return_dict_in_generate=True 和 use_cache=True。"
            )

        generated_group_length = (
            generated_ids.shape[-1] // helper.model.group_size
            if generated_ids.ndim == 3
            else generated_ids.shape[-1]
        )
        total_group_length += generated_group_length

        end_time = min(
            start_time + chunk_seconds,
            wav.numel() / target_sr,
        )

        result = {
            "chunk_index": chunk_index,
            "start": round(start_time, 3),
            "end": round(end_time, 3),
            "prediction": prediction,
        }
        results.append(result)

        print(
            f"[chunk {chunk_index}] "
            f"{start_time:.2f}-{end_time:.2f}s: "
            f"{prediction}"
        )

    return results


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", required=True)
    parser.add_argument("--audio_tokenizer_path", required=True)
    parser.add_argument("--checkpoint_path", default=None)
    parser.add_argument("--audio_path", required=True)
    parser.add_argument("--output_path", default="stream_result.jsonl")

    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--chunk_seconds", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--speech_loss_weights",
        default="100-12-8-6-4-2-2-1-1",
    )

    args = parser.parse_args()

    model, tokenizer, audio_tokenizer, mel_transform = load_model(
        model_path=args.model_path,
        audio_tokenizer_path=args.audio_tokenizer_path,
        checkpoint_path=args.checkpoint_path,
        speech_loss_weights=args.speech_loss_weights,
        device=args.device,
        use_lora=args.use_lora,
    )

    helper = make_dataset_helper(
        tokenizer=tokenizer,
        audio_tokenizer=audio_tokenizer,
        mel_transform=mel_transform,
        model=model,
        use_lora=args.use_lora,
    )

    results = streaming_infer(
        model=model,
        tokenizer=tokenizer,
        helper=helper,
        audio_path=args.audio_path,
        chunk_seconds=args.chunk_seconds,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        device=args.device,
    )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for result in results:
            f.write(
                json.dumps(result, ensure_ascii=False) + "\n"
            )


if __name__ == "__main__":
    main()
