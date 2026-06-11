import re
import os
import json
import numpy as np
from dataclasses import dataclass
from typing import Union, Optional, Any, List, Dict

import torch
import torchaudio
from torch.utils.data import Dataset
from torchcodec.decoders import AudioDecoder

import transformers
from transformers import logging

from model.mimo_audio.process_speechdata import InputSegment, StreamingInputSegment


logger = logging.get_logger(__name__)

DEFAULT_SAMPLE_RATE = 16000

def read_audio(ele: dict):
    audio_decoder = AudioDecoder(source=ele['audio'], sample_rate=DEFAULT_SAMPLE_RATE)
    audio_sr = audio_decoder.metadata.sample_rate
    audio_duration = audio_decoder.metadata.duration_seconds_from_header
    total_frames = int(audio_duration*audio_sr)
    audio_pts = np.linspace(1/audio_sr, audio_duration, total_frames)
    audio_start = ele.get("audio_start", None)
    audio_end = ele.get("audio_end", None)
    clip_idxs = None
    if audio_start is not None or audio_end is not None:
        audio_start = audio_pts[0] if not audio_start else audio_start
        audio_end = audio_pts[-1] if not audio_end else audio_end
        clip_idxs = ((audio_start <= audio_pts) & (audio_pts <= audio_end)).nonzero()[0]
        clip_pts = audio_pts[clip_idxs]
        total_frames = len(clip_pts)
    else:
        audio_start = 0
        audio_end = audio_duration
        
    nframes = int(total_frames/audio_sr*DEFAULT_SAMPLE_RATE)
    nframes_idxs = np.linspace(0, total_frames - 1, nframes).round().astype(int)
    clip_idxs = nframes_idxs if clip_idxs is None else clip_idxs[nframes_idxs]
    clip_pts = audio_pts[clip_idxs]
    clip = audio_decoder.get_samples_played_in_range(start_seconds=audio_start, stop_seconds=audio_end+1/DEFAULT_SAMPLE_RATE).data

    return clip.squeeze(0), clip_pts, audio_sr

def _read_last_line(path: str, buf: int = 4096) -> str:
    with open(path, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        pos, last = size, b""
        while pos > 0:
            read_sz = min(buf, pos)
            pos -= read_sz
            f.seek(pos)
            chunk = f.read(read_sz)
            lines = (chunk + last).split(b"\n")
            last  = lines[0]
            non_empty = [l for l in lines[1:] if l.strip()]
            if non_empty:
                return non_empty[-1].decode("utf-8")
    return last.decode("utf-8")

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
            list_data_dict = json.load(open(data_args.data_path, "r"))
        else:
            list_data_dict = json.load(open(data_args.validate_path, "r"))
        self.list_data_dict = list_data_dict

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
        if isinstance(input, torch.Tensor) or (isinstance(input, str) and os.path.isfile(input)):
            if isinstance(input, torch.Tensor):
                wav = input
            else:
                wav, sr = torchaudio.load(input)
                if wav.ndim == 2:
                    wav = wav.mean(dim=0)
                wav = self.resample_audio_if_needed(wav, sr)
            wav = wav
            
            mel = self.wav2mel(wav).transpose(0, 1)  # (seq_len, n_mels)
            mel = mel

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
        sample = self.list_data_dict[i]

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
                        speech_tokens = self.preprocess_input(content_item['audio'])
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
                        speech_tokens = self.preprocess_input(content_item['audio'])
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
                    speech_tokens = self.preprocess_input(item['content'][-1]['audio'])
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