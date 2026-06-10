import re
import json
from pathlib import Path

from sklearn.model_selection import train_test_split

def load_trs_for_asr(trs_file):
    datas = []

    pattern = re.compile(
        r'(\S+)\s+(\S+)\s+"(.*?)"\s+-from\s+([\d.]+)\s+-to\s+([\d.]+)'
    )

    with open(trs_file, encoding="utf-8") as f:
        idx = 0

        for line in f:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            m = pattern.match(line)
            if m is None:
                continue

            _, wav_path, text, start, end = m.groups()

            datas.append({
                "id": idx,
                "audio_path": wav_path,
                "start": float(start),
                "end": float(end),
                "text": text,
            })
            
            idx += 1

    return datas

def build_conversations(datas: list):
    conversations = []
    for data in datas:
        user_msg = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Transcribe the audio"
                },
                {
                    "type": "audio", 
                    "audio": data["audio_path"], 
                    "start": data["start"],
                    "end": data["end"]
                }
            ]
        }

        assistant_msg = {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": data["text"]
                }
            ]
        }

        conversations.append([user_msg, assistant_msg])

    return conversations

def save_conversations(conversations: list, output_path):
    with open(output_path, 'w') as f:
        for conversation in conversations:
            line = json.dumps(conversation) + '\n'
            f.write(line)

    print(f"Saved to: {output_path}")

def generate_dataset(trs_dir:str, output_dir:str):
    output = Path(output_dir)
    output.mkdir(exist_ok=True, parents=True)
    trs_files = [x for x in Path(trs_dir).glob("*.trs")]
    all_conversation = []
    for trs_file in trs_files:
        datas = load_trs_for_asr(str(trs_file))
        conversations = build_conversations(datas)
        all_conversation.extend(conversations)

    train_conv, test_conv = train_test_split(all_conversation, test_size=0.4)
    test_conv, dev_conv = train_test_split(test_conv, test_size=0.5)
    
    output_path = output / ("train.jsonl")
    save_conversations(train_conv, output_path)
    output_path = output / ("test.jsonl")
    save_conversations(test_conv, output_path)
    output_path = output / ("dev.jsonl")
    save_conversations(dev_conv, output_path)

if __name__ == "__main__":
    trs_dir = "/ctd/Works/c-zheng/End2End/Chinese_general_AddNoise_RoomSimu/00trs_ok_202502_small"
    output_dir = "/home/m-wu/proj/MiMo/bin/dataprocessing/jsonl_data"
    generate_dataset(trs_dir, output_dir)