import re
import os
import json
import numpy as np
from typing import Union, Optional, Any, List, Dict

import torch
import torchaudio
from torch.utils.data import Dataset
from torchcodec.decoders import AudioDecoder

from transformers import logging

from model.mimo_audio.process_speechdata import InputSegment

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
    """
    Convert one Qwen2Audio-style single-turn conversation to MiMo-style message.

    Input example:
    [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Transcribe the audio"},
                {"type": "audio", "audio": "...wav", "start": 0.0, "end": 2.344}
            ]
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "保费 您 是 帐户 扣 的 还是 刷卡 缴费 "}
            ]
        }
    ]

    Output:
    [
        {"role": "user", "content": "...wav"},
        {"role": "assistant", "content": "保费 您 是 帐户 扣 的 还是 刷卡 缴费"}
    ]
    """
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

class MiMoStreamingDataset(Dataset):
    def __init__(
                self,
                annotation_paths: list[str]
            ):
        super().__init__()
        self.handles: list[tuple[str, int]] = []
        for ap in annotation_paths:
            ap = str(ap)
            if ap.endswith(".jsonl"):
                # last line stores seek indices
                seeks = json.loads(_read_last_line(ap))
                self.handles.extend([(ap, sk) for sk in seeks])
                # logger.warning(f"Loaded {ap} ({len(seeks)} samples)")
            elif ap.endswith(".json"):
                # single-record JSON; seek=0 sentinel handled in load_record
                self.handles.append((ap, -1))
                logger.warning(f"Loaded single-record {ap}")
            else:
                raise ValueError(f"Unsupported annotation format: {ap}")
            
        self.group_size=self.model.config.group_size
        self.audio_channels=self.model.config.audio_channels
        self.delay_pattern = self.model.config.delay_pattern
        self.vocab_size = self.model.config.vocab_size

        self.speech_zeroemb_idx = self.model.speech_empty_ids

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
            wav = wav.to(self.device)
            
            mel = self.wav2mel(wav).transpose(0, 1)  # (seq_len, n_mels)

            input_len = mel.size(0)
            segment_size = 6000
            input_len_seg = [segment_size] * (input_len // segment_size)
            if input_len % segment_size > 0:
                input_len_seg.append(input_len % segment_size)

            codes_packed = self.encode_batch(
                input_features=mel, 
                input_lens=torch.tensor(input_len_seg),
            )
            
            codes = codes_packed.transpose(0, 1).detach().cpu()
            audio_codes = codes[:, :self.audio_channels]

            # Pad the sequence to be a multiple of group_size by repeating the last frame
            num_timesteps = audio_codes.shape[0]
            if num_timesteps % self.group_size != 0:
                padding_needed = self.group_size - (num_timesteps % self.group_size)
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

    def get_s2t_dialogue_sft_multiturn_prompt(self, message_list, thinking=False):
        lm_prompt = []
        for i in range(len(message_list)):
            if message_list[i]['role'] == 'user':
                lm_prompt += [
                    InputSegment(
                        text=f"<|im_start|>user\n",
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    ),
                    InputSegment(
                        audio=self.preprocess_input(message_list[i]['content']),
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    ),
                    InputSegment(
                        text=f"<|im_end|>\n",
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    )
                ]
            elif message_list[i]['role'] == 'assistant':
                lm_prompt += [
                    InputSegment(
                        text=f"<|im_start|>assistant\n",
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    ),
                    InputSegment(
                        text=message_list[i]['content'],
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    ),
                    InputSegment(
                        text=f"<|im_end|>\n",
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    )
                ]
            else:
                raise ValueError(f"Invalid role: {message_list[i]['role']}")
        
        lm_prompt.append(
            InputSegment(
                text=f"<|im_start|>assistant\n",
                speech_zeroemb_idx=self.speech_zeroemb_idx,
                text_zeroemb_idx=self.empty_token,
            )
        )
        if not thinking:
            lm_prompt.append(
                InputSegment(
                    text="<think>\n\n</think>\n",
                    speech_zeroemb_idx=self.speech_zeroemb_idx,
                    text_zeroemb_idx=self.empty_token,
                )
            )
        else:
            lm_prompt.append(
                InputSegment(
                    text="<think>\n",
                    speech_zeroemb_idx=self.speech_zeroemb_idx,
                    text_zeroemb_idx=self.empty_token,
                )
            )
        input_ids = self.get_input_ids(lm_prompt)
        return input_ids
    
    def load_record(self, index: int) -> dict:
        path, seek = self.handles[index]
        if seek == -1:                          # single .json file
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        with open(path, encoding="utf-8") as f:
            f.seek(seek)
            return json.loads(f.readline())
    
    def getitem(self, index:int):
        record = self.load_record(index)
        conversation = build_conversation(record)
        return record

    def __getitem__(self, index):
        return self.getitem(index)
    
    def __len__(self):
        return len(self.handles)
    
if __name__ == "__main__":
    path = ["/home/m-wu/proj/MiMo/bin/dataprocessing/jsonl_data/train.jsonl"]
    dataset = MiMoStreamingDataset(annotation_paths=path)
    