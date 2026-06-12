import argparse
import json
import os
from pathlib import Path

import torch
import torchaudio
from transformers import GenerationConfig

from model.mimo_audio.mimo_audio import MimoAudio

from model.mimo_audio.process_speechdata import InputSegment
from model.mimo_audio.modeling_mimo_audio import MiMoSampler, MiMoStopper


def read_last_line(path: str):
    with open(path, "rb") as f:
        f.seek(-2, 2)
        while f.read(1) != b"\n":
            f.seek(-2, 1)
        return f.readline()


def load_jsonl_by_seek(path: str):
    seeks = json.loads(read_last_line(path))
    samples = []

    with open(path, "r", encoding="utf-8") as f:
        for seek in seeks:
            f.seek(seek)
            samples.append(json.loads(f.readline()))

    return samples


class StreamingMimoAudioModel(MimoAudio):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.stream_past_key_values = None
        self.stream_total_length = 0

    def clear_stream_cache(self):
        self.stream_past_key_values = None
        self.stream_total_length = 0

    def make_segment(self, text="", audio=None):
        return InputSegment(
            text=text,
            audio=audio,
            speech_zeroemb_idx=self.speech_zeroemb_idx,
            text_zeroemb_idx=self.empty_token,
        )

    def build_prompt_from_conversation(
        self,
        conversation,
        append_generation_prompt=True,
    ):

        prompt = []

        for item in conversation:
            role = item["role"]

            if role == "assistant":
                break

            if role not in ("system", "user"):
                continue

            prompt.append(
                self.make_segment(text=f"<|im_start|>{role}\n")
            )

            content = item["content"]
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]

            for content_item in content:
                if content_item["type"] == "text":
                    prompt.append(
                        self.make_segment(
                            text=content_item.get("text", "")
                        )
                    )

                elif content_item["type"] == "audio":
                    audio = self.load_audio_item(content_item)
                    audio_tokens = self.preprocess_input(audio)
                    prompt.append(
                        self.make_segment(audio=audio_tokens)
                    )

            prompt.append(
                self.make_segment(text="<|im_end|>\n")
            )

        if append_generation_prompt:
            prompt.extend(
                [
                    self.make_segment(
                        text="<|im_start|>assistant\n"
                    ),
                    self.make_segment(
                        text="<think>\n\n</think>\n"
                    ),
                ]
            )

        return self.get_input_ids(prompt)

    def load_audio_item(self, item):
        """
        支持训练 JSON 中的：
        {
            "type": "audio",
            "audio": "...wav",
            "start": 1.2,
            "end": 3.4
        }
        """
        path = item["audio"]
        info = torchaudio.info(path)

        sr = info.sample_rate
        start = float(item.get("start", 0.0))
        end = item.get("end")

        frame_offset = int(round(start * sr))

        if end is None:
            num_frames = -1
        else:
            num_frames = max(
                0,
                int(round(float(end) * sr)) - frame_offset,
            )

        wav, loaded_sr = torchaudio.load(
            path,
            frame_offset=frame_offset,
            num_frames=num_frames,
        )

        if wav.ndim == 2:
            wav = wav.mean(dim=0)

        target_sr = self.mimo_audio_tokenizer.config.sampling_rate
        if loaded_sr != target_sr:
            wav = torchaudio.functional.resample(
                wav,
                orig_freq=loaded_sr,
                new_freq=target_sr,
            )

        return wav

    def normal_infer(self, conversation, max_new_tokens=256):
        """
        普通非流式推理，直接复用原 forward()。
        """
        input_ids = self.build_prompt_from_conversation(
            conversation,
            append_generation_prompt=True,
        )

        stopping_criteria = [
            MiMoStopper(
                stop_tokens=[
                    self.tokenizer.eos_token_id,
                    self.im_end_idx,
                ],
                group_size=self.group_size,
                audio_channels=self.audio_channels,
            )
        ]

        return self.forward(
            input_ids,
            stopping_criteria=stopping_criteria,
            max_new_tokens=max_new_tokens,
            task_name="spoken_dialogue",
        )

    @torch.no_grad()
    def streaming_forward(
        self,
        input_ids,
        max_new_tokens=128,
        task_name="spoken_dialogue",
    ):
        """
        真正的 KV-cache 流式解码。

        input_ids:
            只包含当前新 chunk 对应的 MiMo 输入，
            shape 与 get_input_ids() 返回一致。

        注意：
            这要求 MiMoAudioForCausalLM.generate() 支持：
              past_key_values
              use_cache
              return_dict_in_generate
            如果自定义 generate 没把 cache 返回出来，
            就需要进一步改 modeling_mimo_audio.py。
        """
        task_sampler = self.get_task_sampler(task_name)

        # 与原 forward 完全一样的 flatten 方式
        flat_input_ids = input_ids.T.reshape(1, -1).to(self.device)

        current_prompt_length = (
            flat_input_ids.shape[1] // (self.audio_channels + 1)
        )

        generation_config = GenerationConfig(
            max_length=(
                self.stream_total_length
                + current_prompt_length // self.group_size
                + max_new_tokens
            ),
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        try:
            output = self.model.generate(
                input_ids=flat_input_ids,
                generation_config=generation_config,
                past_key_values=self.stream_past_key_values,
                use_cache=True,
                return_dict_in_generate=True,
                global_sampler=task_sampler["global"],
                local_sampler=task_sampler["local"],
            )
        except TypeError as e:
            raise RuntimeError(
                "当前 MiMoAudioForCausalLM.generate() "
                "不接受 past_key_values 或 return_dict_in_generate。"
                "需要在 modeling_mimo_audio.py 的自定义 generate 中"
                "把这两个参数继续传给 model.forward()。"
            ) from e

        sequences = (
            output.sequences
            if hasattr(output, "sequences")
            else output
        )

        past_key_values = getattr(
            output,
            "past_key_values",
            None,
        )

        if past_key_values is None:
            raise RuntimeError(
                "generate() 没有返回 past_key_values。"
                "需要修改 modeling_mimo_audio.py，"
                "让生成结果保留最后一步的 KV cache。"
            )

        self.stream_past_key_values = past_key_values

        # sequences 通常包含当前 prompt + 新生成结果
        reshaped = sequences.int().cpu().reshape(
            -1,
            self.audio_channels + 1,
        ).T

        generated = reshaped[:, current_prompt_length:]

        generated_group_length = (
            generated.shape[1] // self.group_size
        )
        self.stream_total_length += (
            current_prompt_length // self.group_size
            + generated_group_length
        )

        text_ids = generated[0, ::self.group_size]

        if text_ids.numel() > 0:
            text_ids = text_ids[:-1]

        text = self.tokenizer.decode(
            text_ids,
            skip_special_tokens=False,
        )
        text = text.strip().replace("<|empty|>", ".")

        if "<|im_end|>" in text:
            text = text.split("<|im_end|>", 1)[0]

        return text

    def build_stream_chunk_prompt(
        self,
        wav_chunk,
        first_chunk=False,
    ):
        """
        每个 chunk 作为一轮新的 user 输入。

        首个 chunk:
          <|im_start|>user
          [audio]
          <|im_end|>
          <|im_start|>assistant
          <think>...</think>

        后续 chunk 也使用相同格式，但历史通过 KV cache 传递。
        """
        audio_tokens = self.preprocess_input(wav_chunk)

        prompt = [
            self.make_segment(text="<|im_start|>user\n"),
            self.make_segment(audio=audio_tokens),
            self.make_segment(text="<|im_end|>\n"),
            self.make_segment(
                text="<|im_start|>assistant\n"
            ),
            self.make_segment(
                text="<think>\n\n</think>\n"
            ),
        ]

        return self.get_input_ids(prompt)

    @torch.no_grad()
    def stream_audio(
        self,
        audio_item,
        chunk_seconds=1.0,
        max_new_tokens=128,
    ):
        """
        把一条音频等长切成多个 chunk，
        每个 chunk 调用 streaming_forward()。
        """
        self.clear_stream_cache()

        wav = self.load_audio_item(audio_item)
        sr = self.mimo_audio_tokenizer.config.sampling_rate

        chunk_samples = int(round(chunk_seconds * sr))
        results = []

        for chunk_index, start in enumerate(
            range(0, wav.numel(), chunk_samples)
        ):
            chunk = wav[start:start + chunk_samples]

            if chunk.numel() == 0:
                continue

            input_ids = self.build_stream_chunk_prompt(
                chunk,
                first_chunk=(chunk_index == 0),
            )

            prediction = self.streaming_forward(
                input_ids,
                max_new_tokens=max_new_tokens,
                task_name="spoken_dialogue",
            )

            result = {
                "chunk_index": chunk_index,
                "start": round(start / sr, 3),
                "end": round(
                    min(start + chunk.numel(), wav.numel()) / sr,
                    3,
                ),
                "prediction": prediction,
            }
            results.append(result)

            print(
                f"[chunk {chunk_index}] "
                f"{result['start']:.2f}-{result['end']:.2f}: "
                f"{prediction}"
            )

        return results


def find_first_user_audio(conversation):
    for item in conversation:
        if item.get("role") != "user":
            continue

        content = item.get("content", [])
        if isinstance(content, list):
            for x in content:
                if (
                    isinstance(x, dict)
                    and x.get("type") == "audio"
                ):
                    return x

    raise ValueError("样本中没有找到 user audio")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", required=True)
    parser.add_argument("--audio_tokenizer_path", required=True)
    parser.add_argument("--lora_path", default=None)
    parser.add_argument("--jsonl_path", required=True)
    parser.add_argument("--output_path", default="result.jsonl")

    parser.add_argument(
        "--mode",
        choices=["normal", "streaming"],
        default="normal",
    )
    parser.add_argument("--chunk_seconds", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None)

    args = parser.parse_args()

    model = StreamingMimoAudioModel(
        model_path=args.model_path,
        mimo_audio_tokenizer_path=args.audio_tokenizer_path,
        lora_path=args.lora_path,
        device=args.device,
    )

    samples = load_jsonl_by_seek(args.jsonl_path)

    if args.limit is not None:
        samples = samples[:args.limit]

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as fout:
        for sample_index, conversation in enumerate(samples):
            if args.mode == "normal":
                prediction = model.normal_infer(
                    conversation,
                    max_new_tokens=args.max_new_tokens,
                )

                result = {
                    "index": sample_index,
                    "prediction": prediction,
                }

            else:
                audio_item = find_first_user_audio(conversation)

                chunks = model.stream_audio(
                    audio_item,
                    chunk_seconds=args.chunk_seconds,
                    max_new_tokens=args.max_new_tokens,
                )

                result = {
                    "index": sample_index,
                    "chunks": chunks,
                    "prediction": "".join(
                        x["prediction"] for x in chunks
                    ),
                }

            fout.write(
                json.dumps(result, ensure_ascii=False) + "\n"
            )
            fout.flush()


if __name__ == "__main__":
    main()
