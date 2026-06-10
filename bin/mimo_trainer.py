from typing import List, Optional, Dict, Any

import torch

from transformers import Trainer
from transformers.modeling_outputs import BaseModelOutputWithPast

class MiMoTrainer(Trainer):
    """
    这个 Trainer 解决两个问题：

    1. MiMo 原 forward 只返回最后一步 logits，不适合直接 teacher-forcing。
    2. MiMo 有两层预测：
       - global transformer 预测 text token
       - local transformer 预测 speech/audio token
    """

    def compute_loss(
        self,
        model: MiMoAudioForCausalLM,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        **kwargs,
    ):
        input_ids = inputs["input_ids"]
        labels = inputs.get("labels", None)

        if labels is None:
            labels = input_ids.clone()

        group_size = model.config.group_size
        audio_channels = model.config.audio_channels
        empty_idx = model.args.empty_idx

        input_ids = normalize_mimo_ids(
            input_ids=input_ids,
            audio_channels=audio_channels,
            group_size=group_size,
        )

        labels = normalize_mimo_ids(
            input_ids=labels,
            audio_channels=audio_channels,
            group_size=group_size,
        )

        input_ids = input_ids.to(model.device)
        labels = labels.to(model.device)

        B = input_ids.shape[0]
        T_group = input_ids.shape[-1] // group_size

        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is None:
            attention_mask = torch.ones(
                B,
                T_group,
                dtype=torch.long,
                device=model.device,
            )
        else:
            attention_mask = attention_mask.to(model.device)

        if attention_mask.dim() == 3:
            # 万一 dataset 给的是 token-level mask: [B, C+1, T*group]
            attention_mask = attention_mask[:, 0, ::group_size]

        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)

        # ========== Global transformer teacher-forcing ==========
        # _prepare_input_embeds: [B, C+1, T*group] -> [B, T_group, hidden]
        inputs_embeds = model._prepare_input_embeds(input_ids)

        outputs: BaseModelOutputWithPast = model.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            return_dict=True,
        )

        hidden_states = outputs.last_hidden_state  # [B, T_group, H]

        # text channel 每个 group 只取第 0 个位置
        text_labels = labels[:, 0, ::group_size]  # [B, T_group]

        # causal LM: hidden[t] 预测 label[t+1]
        text_logits = model.lm_head(hidden_states[:, :-1, :])  # [B, T-1, vocab]
        shift_text_labels = text_labels[:, 1:].contiguous()    # [B, T-1]

        text_loss = F.cross_entropy(
            text_logits.reshape(-1, text_logits.shape[-1]),
            shift_text_labels.reshape(-1),
            ignore_index=-100,
        )

        total_loss = self.args.text_loss_weight * text_loss
        speech_loss = None

        # ========== Local transformer speech-token loss ==========
        if self.args.train_speech_loss:
            speech_loss = self.compute_speech_loss(
                model=model,
                hidden_states=hidden_states,
                labels=labels,
                text_labels=text_labels,
                attention_mask=attention_mask,
            )

            if speech_loss is not None:
                total_loss = total_loss + self.args.speech_loss_weight * speech_loss

        output_dict = {
            "loss": total_loss,
            "text_loss": text_loss.detach(),
        }

        if speech_loss is not None:
            output_dict["speech_loss"] = speech_loss.detach()

        return (total_loss, output_dict) if return_outputs else total_loss

    def compute_speech_loss(
        self,
        model: MiMoAudioForCausalLM,
        hidden_states: torch.Tensor,
        labels: torch.Tensor,
        text_labels: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        """
        用 teacher forcing 训练 local transformer。

        generation 时逻辑是：
        global hidden -> text token
        如果 text token == empty_idx，则 local transformer 生成 speech token group。

        训练时对应：
        hidden[t] 预测 labels[t+1] 的 speech group。
        """
        B, T_group, _ = hidden_states.shape
        group_size = model.config.group_size
        audio_channels = model.config.audio_channels
        empty_idx = model.args.empty_idx

        if T_group <= 1:
            return None

        # hidden[t] 预测 group[t+1]
        local_start_embeds = model.hidden_states_downcast(
            hidden_states[:, :-1, :]
        )  # [B, T-1, local_dim]

        next_text_labels = text_labels[:, 1:]  # [B, T-1]

        # speech labels: [B, C, T*group] -> [B, C, T, group] -> [B, T, C, group]
        speech_labels = labels[:, 1:, :]
        speech_labels = speech_labels.view(
            B,
            audio_channels,
            T_group,
            group_size,
        ).permute(0, 2, 1, 3).contiguous()

        target_speech = speech_labels[:, 1:, :, :]  # [B, T-1, C, group]

        N = B * (T_group - 1)

        local_embeds = local_start_embeds.reshape(
            N,
            1,
            model.local_config.hidden_size,
        )

        target_speech = target_speech.reshape(
            N,
            audio_channels,
            group_size,
        )

        next_text_labels = next_text_labels.reshape(N)

        # 只在 next text 是 empty_idx 时训练 speech token
        is_speech_group = next_text_labels == empty_idx  # [N]

        past_key_values = None
        losses = []

        delay_iters = group_size + max(model.delay_pattern)

        for t in range(delay_iters):
            local_outputs: BaseModelOutputWithPast = model.local_transformer(
                inputs_embeds=local_embeds,
                past_key_values=past_key_values,
                return_dict=True,
                use_cache=True,
            )

            local_hidden = local_outputs.last_hidden_state  # [N, 1, local_dim]
            past_key_values = local_outputs.past_key_values

            next_local_embeds = torch.zeros_like(local_embeds)

            for ch in range(audio_channels):
                cur_start = model.delay_pattern[ch]
                cur_end = cur_start + group_size

                if not (cur_start <= t < cur_end):
                    continue

                pos = t - cur_start
                cur_target = target_speech[:, ch, pos]  # [N]

                cur_empty = model.speech_empty_ids[ch]

                valid_mask = (
                    is_speech_group
                    & (cur_target != -100)
                    & (cur_target != cur_empty)
                )

                if valid_mask.any():
                    cur_logits = model.local_transformer_lm_heads[ch](
                        local_hidden[:, -1, :]
                    )  # [N, vocab_ch]

                    cur_loss = F.cross_entropy(
                        cur_logits[valid_mask],
                        cur_target[valid_mask].long(),
                        ignore_index=-100,
                    )
                    losses.append(cur_loss)

                # teacher forcing: 当前 step 的目标 token 作为下一 step 输入
                embed_target = cur_target.clone()
                embed_target[embed_target == -100] = cur_empty
                embed_target = embed_target.long()

                cur_input_embed = model.speech_embeddings[ch](
                    embed_target.unsqueeze(1)
                )  # [N, 1, input_local_dim]

                if model.speech_embeddings_to_local is not None:
                    cur_input_embed = model.speech_embeddings_to_local(cur_input_embed)

                # 非 speech group 不喂 speech token
                cur_input_embed = cur_input_embed * is_speech_group.view(N, 1, 1)
                next_local_embeds = next_local_embeds + cur_input_embed

            local_embeds = next_local_embeds

        if len(losses) == 0:
            return None

        return torch.stack(losses).mean()