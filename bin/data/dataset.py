import re
import os
import json
import numpy as np
from dataclasses import dataclass, field
from typing import Union, Optional, Any, List, Dict, Tuple

import torch
import torchaudio
from torch.utils.data import Dataset

import transformers
from transformers import logging

from model.mimo_audio.process_speechdata import InputSegment, StreamingInputSegment


logger = logging.get_logger(__name__)

DEFAULT_SAMPLE_RATE = 16000

def read_audio(ele: dict) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    使用 torchaudio 读取、裁剪并重采样音频。

    Args:
        ele:
            {
                "audio": "/path/to/audio.wav",
                "audio_start": 1.2,  # 可选，单位为秒
                "audio_end": 3.5,    # 可选，单位为秒
            }

    Returns:
        clip:
            重采样后的单通道音频，shape 为 [num_samples]。
        clip_pts:
            clip 中每个采样点对应的原始音频绝对时间，单位为秒，
            shape 为 [num_samples]。
        audio_sr:
            原始音频采样率。
    """
    audio_path = ele["audio"]

    info = torchaudio.info(audio_path)

    audio_sr = info.sample_rate
    total_source_frames = info.num_frames
    audio_duration = total_source_frames / audio_sr

    audio_start = ele.get("audio_start")
    audio_end = ele.get("audio_end")

    # 注意：不能用 `if not audio_start`，因为 0.0 是合法值
    if audio_start is None:
        audio_start = 0.0

    if audio_end is None:
        audio_end = audio_duration

    audio_start = max(0.0, float(audio_start))
    audio_end = min(float(audio_end), audio_duration)

    if audio_end <= audio_start:
        raise ValueError(
            f"Invalid audio range: start={audio_start}, end={audio_end}, "
            f"duration={audio_duration}"
        )

    # 采用左闭右开区间 [audio_start, audio_end)
    frame_offset = int(round(audio_start * audio_sr))
    end_frame = int(round(audio_end * audio_sr))

    frame_offset = min(frame_offset, total_source_frames)
    end_frame = min(end_frame, total_source_frames)

    num_frames = max(0, end_frame - frame_offset)

    waveform, loaded_sr = torchaudio.load(
        audio_path,
        frame_offset=frame_offset,
        num_frames=num_frames,
    )

    if loaded_sr != audio_sr:
        raise RuntimeError(
            f"Unexpected sample rate: info={audio_sr}, loaded={loaded_sr}"
        )

    # 多通道音频转为单通道
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if audio_sr != DEFAULT_SAMPLE_RATE:
        waveform = torchaudio.functional.resample(
            waveform,
            orig_freq=audio_sr,
            new_freq=DEFAULT_SAMPLE_RATE,
        )

    clip = waveform.squeeze(0)

    # 每个输出采样点对应的绝对音频时间
    clip_pts = (
        torch.arange(
            clip.numel(),
            device=clip.device,
            dtype=torch.float64,
        )
        / DEFAULT_SAMPLE_RATE
        + audio_start
    )

    return clip, clip_pts, audio_sr

def readlastline(path: str):
    with open(path, "rb") as f:
        f.seek(-2, 2) # avoid last \n
        while f.read(1) != b"\n":  
            f.seek(-2, 1)
        return f.readline()

def build_conversation(
    conversation: List[Dict[str, Any]],
    strip_text: bool = True,
) -> List[Dict[str, str]]:

    user_audio_path = None
    assistant_text = None

    for turn in conversation:
        role = turn.get("role")
        content = turn.get("content")

        if role == "user":
            if isinstance(content, list):
                for x in content:
                    if isinstance(x, dict) and x.get("type") == "audio":
                        user_audio_path = x.get("audio")
                        break
            elif isinstance(content, str):
                # already MiMo-style
                user_audio_path = content

        elif role == "assistant":
            if isinstance(content, list):
                texts = []
                for x in content:
                    if isinstance(x, dict) and x.get("type") == "text":
                        text = x.get("text", "")
                        texts.append(text)
                assistant_text = "".join(texts)
            elif isinstance(content, str):
                # already MiMo-style
                assistant_text = content

    if user_audio_path is None:
        raise ValueError(f"No user audio found in conversation: {conversation}")

    if assistant_text is None:
        raise ValueError(f"No assistant text found in conversation: {conversation}")

    if strip_text:
        assistant_text = assistant_text.strip()

    return [
        {
            "role": "user",
            "content": user_audio_path,
        },
        {
            "role": "assistant",
            "content": assistant_text,
        },
    ]

class AudioDataSet(Dataset):
    def __init__(self, tokenizer, mimo_audio_tokenizer, mel_transform, path_item, data_args, model, lora_enable=False, ignore_index=-100):
        super(AudioDataSet, self).__init__()
        self.tokenizer = tokenizer
        self.mimo_audio_tokenizer = mimo_audio_tokenizer
        self.mel_transform = mel_transform
        self.path_item = path_item
        self.data_args = data_args
        if lora_enable:
            self.model = model.model
        else:
            self.model = model

        self.speech_zeroemb_idx = self.model.speech_empty_ids
        self.ignore_index = ignore_index
        

        if path_item == 'train':
            handles = []
            assert data_args.data_path.endswith('.jsonl'), f"Please organize the annotations in JSONL format, with each data sample on a separate line, and the last line stores the seek indices"
            logger.warning(f'Load {data_args.data_path}. Please ensure its last line stores the seek indices...')
            seeks = json.loads(readlastline(data_args.data_path))
            handles.extend(zip([data_args.data_path] * len(seeks), seeks))
            logger.warning(f'Successfully loaded {data_args.data_path}')
        else:
            handles = []
            assert data_args.validate_path.endswith('.jsonl'), f"Please organize the annotations in JSONL format, with each data sample on a separate line, and the last line stores the seek indices"
            logger.warning(f'Load {data_args.validate_path}. Please ensure its last line stores the seek indices...')
            seeks = json.loads(readlastline(data_args.validate_path))
            handles.extend(zip([data_args.validate_path] * len(seeks), seeks))
            logger.warning(f'Successfully loaded {data_args.validate_path}')

        self.list_data_dict = handles

    def load_conversation(self, index):
        annotation_path, seek = self.list_data_dict[index]
        with open(annotation_path) as f:
            f.seek(seek)
            line = f.readline()
        line = json.loads(line)
        return line

    def __len__(self):
        return len(self.list_data_dict)

    def wav2mel(self, wav):
        spec = self.mel_transform(wav[None, :])
        return torch.log(torch.clip(spec, min=1e-7)).squeeze()

    def resample_audio_if_needed(self, wav_tensor: torch.Tensor, original_sr: int):
        target_sr = self.mimo_audio_tokenizer.config.sampling_rate
        if original_sr != target_sr:
            wav_tensor = torchaudio.functional.resample(
                wav_tensor, original_sr, target_sr
            )
        return wav_tensor

    def group_by_length(self, features: torch.Tensor, lengths: torch.Tensor, max_length: int):
        if features.size(0) != lengths.sum().item():
            raise ValueError(f"Feature size mismatch: {features.size(0)} vs {lengths.sum().item()}")
        
        split_points = []
        current_sum = 0
        
        for i, seq_len in enumerate(lengths):
            if current_sum + seq_len > max_length and current_sum > 0:
                split_points.append(i)
                current_sum = seq_len.item()
            else:
                current_sum += seq_len.item()
        
        # Convert split points to group sizes
        group_sizes = []
        prev = 0
        for point in split_points:
            group_sizes.append(point - prev)
            prev = point
        if prev < len(lengths):
            group_sizes.append(len(lengths) - prev)
        
        len_groups = torch.split(lengths, group_sizes)
        feature_sizes = [group.sum().item() for group in len_groups]
        feature_groups = torch.split(features, feature_sizes)
        
        return feature_groups, len_groups

    def encode_batch(self, input_features: torch.Tensor, input_lens: torch.Tensor, max_length: int = 256000):
        input_features = input_features.to(device=self.mimo_audio_tokenizer.device, dtype=torch.bfloat16)
        input_lens = input_lens.to(device=self.mimo_audio_tokenizer.device)
        feature_groups, len_groups = self.group_by_length(input_features, input_lens, max_length)
        
        encoded_parts = []
        for features, lengths in zip(feature_groups, len_groups):
            with torch.no_grad():
                codes, _ = self.mimo_audio_tokenizer.encoder.encode(
                    input_features=features,
                    input_lens=lengths, 
                    return_codes_only=True
                )
                encoded_parts.append(codes)
        
        return torch.cat(encoded_parts, dim=-1).cpu()

    def get_input_ids(self, prompt):
        input_ids = [
            seg.to_input_id(
                self.tokenizer, 
                self.model.group_size, 
                self.model.audio_channels,
            )
            for seg in prompt
        ]
        input_ids = torch.cat(input_ids, dim=1)
        return input_ids

    def preprocess_input(
        self,
        input: Union[None, str, torch.Tensor] = None,
    ):
        if (isinstance(input, torch.Tensor) 
        or (isinstance(input, str) and os.path.isfile(input)) 
        or (isinstance(input, dict) and os.path.isfile(input["audio"]))):
            if isinstance(input, torch.Tensor):
                wav = input
            else:
                # wav, sr = torchaudio.load(input)
                # if wav.ndim == 2:
                #     wav = wav.mean(dim=0)
                # wav = self.resample_audio_if_needed(wav, sr)
                wav, _, _ = read_audio(input)
                
            
            mel = self.wav2mel(wav).transpose(0, 1)  # (seq_len, n_mels)

            input_len = mel.size(0)
            segment_size = 3000
            input_len_seg = [segment_size] * (input_len // segment_size)
            if input_len % segment_size > 0:
                input_len_seg.append(input_len % segment_size)

            codes_packed = self.encode_batch(
                input_features=mel, 
                input_lens=torch.tensor(input_len_seg)
            )
            
            codes = codes_packed.transpose(0, 1).detach().cpu()
            audio_codes = codes[:, :self.model.audio_channels]

            # Pad the sequence to be a multiple of group_size by repeating the last frame
            num_timesteps = audio_codes.shape[0]
            if num_timesteps % self.model.config.group_size != 0:
                padding_needed = self.model.config.group_size - (num_timesteps % self.model.config.group_size)
                last_tokens = audio_codes[-1:, :] # Keep dim for repeat
                padding_tokens = last_tokens.repeat(padding_needed, 1)
                audio_codes = torch.cat([audio_codes, padding_tokens], dim=0)
            
            audio_tokenized = audio_codes.reshape(-1)

            return audio_tokenized
        else:
            text = input
            if (
                text.isupper() or text.islower()
            ):  # If the text only contains upper-case or lower-case letters, capitalize it.
                text = text.capitalize()
            return text

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sample = self.load_conversation(i)

        input_segments = []
        labels_with_loss_segments = [[0]]
        for item in sample:
            if item['role'] == 'system':
                input_segments.append(InputSegment(
                    text="<|im_start|>system\n",
                    speech_zeroemb_idx=self.speech_zeroemb_idx,
                    text_zeroemb_idx=self.model.args.empty_idx,
                ))
                for content_item in item['content']:
                    if content_item['type'] == 'text':
                        input_segments.append(InputSegment(
                            text=content_item['text'],
                            speech_zeroemb_idx=self.speech_zeroemb_idx,
                            text_zeroemb_idx=self.model.args.empty_idx,
                        ))
                    elif content_item['type'] == 'audio':
                        speech_tokens = self.preprocess_input(content_item)
                        input_segments.append(InputSegment(
                            text="",
                            audio=speech_tokens,
                            speech_zeroemb_idx=self.speech_zeroemb_idx,
                            text_zeroemb_idx=self.model.args.empty_idx,
                        ))
                input_segments.append(
                    InputSegment(
                        text="<|im_end|>\n",
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.model.args.empty_idx,
                    )
                )
            elif item['role'] == 'user':
                input_segments.append(InputSegment(
                    text="<|im_start|>user\n",
                    speech_zeroemb_idx=self.speech_zeroemb_idx,
                    text_zeroemb_idx=self.model.args.empty_idx,
                ))
                for content_item in item['content']:
                    if content_item['type'] == 'text':
                        input_segments.append(InputSegment(
                            text=content_item['text'],
                            speech_zeroemb_idx=self.speech_zeroemb_idx,
                            text_zeroemb_idx=self.model.args.empty_idx,
                        ))
                    elif content_item['type'] == 'audio':
                        speech_tokens = self.preprocess_input(content_item)
                        input_segments.append(InputSegment(
                            text="",
                            audio=speech_tokens,
                            speech_zeroemb_idx=self.speech_zeroemb_idx,
                            text_zeroemb_idx=self.model.args.empty_idx,
                        ))
                input_segments.append(
                    InputSegment(
                        text="<|im_end|>\n",
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.model.args.empty_idx,
                    )
                )
            elif item['role'] == 'assistant':
                input_segments.append(InputSegment(
                    text="<|im_start|>assistant\n",
                    speech_zeroemb_idx=self.speech_zeroemb_idx,
                    text_zeroemb_idx=self.model.args.empty_idx,
                ))
                item['thinking'] = False
                if item['thinking']:
                    input_segments.append(
                        InputSegment(
                            text="<think>\n",
                            speech_zeroemb_idx=self.speech_zeroemb_idx,
                            text_zeroemb_idx=self.model.args.empty_idx,
                        )
                    )
                    current_input_length = self.get_input_ids(input_segments).size(-1)
                    labels_with_loss_segments[-1].append(current_input_length)
                    input_segments.append(
                        InputSegment(
                            text=item['Chain-of-thought'],
                            speech_zeroemb_idx=self.speech_zeroemb_idx,
                            text_zeroemb_idx=self.model.args.empty_idx,
                        ),
                        InputSegment(
                            text="</think>\n",
                            speech_zeroemb_idx=self.speech_zeroemb_idx,
                            text_zeroemb_idx=self.model.args.empty_idx,
                        )
                    )
                elif item['thinking'] == False:
                    input_segments.append(
                        InputSegment(
                            text="<think>\n\n</think>\n",
                            speech_zeroemb_idx=self.speech_zeroemb_idx,
                            text_zeroemb_idx=self.model.args.empty_idx,
                        )
                    )
                    current_input_length = self.get_input_ids(input_segments).size(-1)
                    labels_with_loss_segments[-1].append(current_input_length)
                elif item['thinking'] == None:
                    current_input_length = self.get_input_ids(input_segments).size(-1)
                    labels_with_loss_segments[-1].append(current_input_length)

                if 'audio' in item['content'][-1]:
                    speech_tokens = self.preprocess_input(item['content'][-1])
                    input_segments.append(StreamingInputSegment(
                        text=item['content'][-2]['text'],
                        audio=speech_tokens,
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.model.args.empty_idx,
                        tokenizer=self.model.tokenizer,
                        group_size=self.model.group_size,
                        audio_channels=self.model.audio_channels,
                    ))
                else:
                    input_segments.append(InputSegment(
                        text=item['content'][0]['text'],
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.model.args.empty_idx,
                    ))
                input_segments.append(
                    InputSegment(
                        text="<|im_end|>\n",
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.model.args.empty_idx,
                    )
                )
                labels_with_loss_segments.append([self.get_input_ids(input_segments).size(-1)])
        
        input_ids = self.get_input_ids(input_segments)
        labels = input_ids.clone()
        
        for i in range(len(labels_with_loss_segments[:-1])):
            labels[:, labels_with_loss_segments[i][0]:labels_with_loss_segments[i][1]] = self.ignore_index

        attention_mask = torch.ones(input_ids.shape[-1] // self.model.group_size).int()
        return dict(
            input_ids=input_ids.transpose(0, 1),
            labels=labels.transpose(0, 1),
            attention_mask=attention_mask,
        )

@dataclass
class DataCollatorLLMsTraining(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    ignore_index: int
    model: torch.nn.Module

    def __call__(self, instances, return_tensors="pt"):
        input_ids, attention_mask, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "attention_mask", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=0)
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_mask,
            batch_first=True,
            padding_value=0)
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=self.ignore_index)
        
        batch = dict(
            input_ids=input_ids.transpose(1, 2),
            attention_mask=attention_mask,
            labels=labels.transpose(1, 2),
        )
        return batch

    
def make_dialogue_module(tokenizer,
                        mimo_audio_tokenizer,
                        mel_transform,
                        data_args,
                        model,
                        lora_enable) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    if data_args.data_path is not None:
        train_dataset = AudioDataSet(tokenizer=tokenizer,
                                mimo_audio_tokenizer=mimo_audio_tokenizer,
                                mel_transform=mel_transform,
                                path_item='train',
                                data_args=data_args,
                                model=model,
                                lora_enable=lora_enable
                                )
    else:
        raise ValueError("data_args.data_path is None")
    
    if data_args.validate_path is not None:
        validate_dataset = AudioDataSet(tokenizer=tokenizer,
                                mimo_audio_tokenizer=mimo_audio_tokenizer,
                                mel_transform=mel_transform,
                                path_item='validate',
                                data_args=data_args,
                                model=model,
                                lora_enable=lora_enable
                            )
    else:
        validate_dataset = None
    
    data_collator = DataCollatorLLMsTraining(tokenizer=tokenizer,
                                             model=model,
                                             ignore_index=-100)
    
    return dict(train_dataset=train_dataset,
                eval_dataset=validate_dataset,
                data_collator=data_collator)

if __name__ == "__main__":
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    from model.mimo_audio.modeling_mimo_audio import (
        MiMoAudioArguments,
        MiMoAudioForCausalLM,
        MiMoSampler,
        MiMoStopper,
    )
    from model.mimo_audio_tokenizer.modeling_audio_tokenizer import MiMoAudioTokenizer
    from transformers import (
        AutoTokenizer,
        GenerationConfig
    )
    from torchaudio.transforms import MelSpectrogram
    from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

    model_path = "XiaomiMiMo/MiMo-Audio-7B-Instruct"
    tokenizer_path = "XiaomiMiMo/MiMo-Audio-Tokenizer"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(
            model_path
        )
    special_tokens = [
            "<|sosp|>",
            "<|eosp|>",
            "<|empty|>",
            "<|Human|>",
            "<|SpeechLM|>",
            "<|sostm|>",
            "<|eostm|>",
            "<|eot|>",
        ]
    for token in special_tokens:
        if token not in tokenizer.get_vocab():
            print(f"Add special tokens {token} to tokenizer.vocab")
            tokenizer.add_tokens([token], special_tokens=True)

    mimo_audio_tokenizer = MiMoAudioTokenizer.from_pretrained(tokenizer_path)
    mimo_audio_tokenizer.to(device, dtype=torch.bfloat16)
    sosp_idx = tokenizer.convert_tokens_to_ids("<|sosp|>")
    eosp_idx = tokenizer.convert_tokens_to_ids("<|eosp|>")
    empty_token = tokenizer.convert_tokens_to_ids("<|empty|>")
    sostm_idx = tokenizer.convert_tokens_to_ids("<|sostm|>")
    eostm_idx = tokenizer.convert_tokens_to_ids("<|eostm|>")
    eot_idx = tokenizer.convert_tokens_to_ids("<|eot|>")
    im_start_idx = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end_idx = tokenizer.convert_tokens_to_ids("<|im_end|>")

    model_args = MiMoAudioArguments(
            model_name_or_path=model_path,
            sosp_idx=sosp_idx,
            eosp_idx=eosp_idx,
            empty_idx=empty_token,
            sostm_idx=sostm_idx,
            eostm_idx=eostm_idx,
            eot_idx=eot_idx,
            speech_loss_weights="100-12-8-6-4-2-2-1-1"
        )

    model = MiMoAudioForCausalLM.from_pretrained(
        model_path,
        args=model_args,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
    )

    mel_transform = MelSpectrogram(
            sample_rate=mimo_audio_tokenizer.config.sampling_rate,
            n_fft=mimo_audio_tokenizer.config.nfft,
            hop_length=mimo_audio_tokenizer.config.hop_length,
            win_length=mimo_audio_tokenizer.config.window_size,
            f_min=mimo_audio_tokenizer.config.fmin,
            f_max=mimo_audio_tokenizer.config.fmax,
            n_mels=mimo_audio_tokenizer.config.n_mels,
            power=1.0,
            center=True,
        )

    @dataclass
    class DataArguments:
        data_path: str = field(default="/home/m-wu/proj/MiMo/bin/dataprocessing/jsonl_data/train.jsonl",
                            metadata={"help": "Path to the training data."})
        validate_path: str = field(default=None,
                            metadata={"help": "Path to the validation data."})
    data_args = DataArguments()
    data = make_dialogue_module(
            tokenizer=tokenizer, 
            mimo_audio_tokenizer=mimo_audio_tokenizer,
            mel_transform=mel_transform,
            data_args=data_args,
            model=model,
            lora_enable=False
        )
    
    train = data["train_dataset"]
    eval = data["eval_dataset"]
    collator = data["data_collator"]
    
    train_dataloader = DataLoader(train, batch_size=4, shuffle=False, collate_fn=collator)

    for batch in tqdm(train_dataloader):
        breakpoint()
        pass