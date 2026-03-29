"""
TrackletAggregator - BiLSTM + attention pooling over PARSeq crop logits.

Architecture (from docs/finetune_and_aggregation.md):
  Per crop  : flatten(3x11 logits) -> Linear(33->64) -> ReLU
  Sequence  : BiLSTM(64, hidden=128, bidirectional) -> (T, 256)
  Pooling   : learned attention -> context (256)
  Head      : [concat p_2digit if enabled] -> Linear(->128) -> ReLU -> Dropout(0.3) -> Linear(->100)

Toggle use_digit_classifier=False to run without the MoviNet 1-vs-2 digit signal.
"""

from typing import Optional

import torch
import torch.nn as nn


class TrackletAggregator(nn.Module):
    def __init__(self, use_digit_classifier: bool = True):
        super().__init__()
        self.use_digit_classifier = use_digit_classifier

        # Per-crop encoder: 3x11 logits flattened -> 64-dim embedding
        self.crop_encoder = nn.Sequential(
            nn.Linear(33, 64),
            nn.ReLU(),
        )

        # Temporal aggregation
        self.bilstm = nn.LSTM(
            input_size=64,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # Attention: score each hidden state -> scalar weight
        self.attention = nn.Linear(256, 1, bias=False)

        # Classification head
        context_dim = 256 + (1 if use_digit_classifier else 0)
        self.classifier = nn.Sequential(
            nn.Linear(context_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 100),
        )

    def forward(
        self,
        logits_seq: torch.Tensor,   # (B, T_max, 33) padded
        lengths: torch.Tensor,       # (B,) actual T per sample
        p_2digit: Optional[torch.Tensor] = None,  # (B, 1) or None
    ) -> torch.Tensor:               # (B, 100)
        B, T, _ = logits_seq.shape

        # Encode each crop independently
        crop_emb = self.crop_encoder(logits_seq)  # (B, T, 64)

        # BiLSTM with variable-length sequences
        packed = nn.utils.rnn.pack_padded_sequence(
            crop_emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        lstm_out, _ = self.bilstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
            lstm_out, batch_first=True, total_length=T
        )  # (B, T, 256)

        # Masked attention pooling - padding positions get -inf before softmax
        attn_scores = self.attention(lstm_out).squeeze(-1)  # (B, T)
        pad_mask = (
            torch.arange(T, device=logits_seq.device).unsqueeze(0)
            >= lengths.unsqueeze(1)
        )
        attn_scores = attn_scores.masked_fill(pad_mask, float("-inf"))
        attn_weights = torch.softmax(attn_scores, dim=1)  # (B, T)
        context = (lstm_out * attn_weights.unsqueeze(-1)).sum(dim=1)  # (B, 256)

        # Inject tracklet-level digit-count confidence (if available)
        if self.use_digit_classifier:
            if p_2digit is None:
                raise ValueError(
                    "use_digit_classifier=True but p_2digit was not provided. "
                    "Pass p_2digit or create the model with use_digit_classifier=False."
                )
            context = torch.cat([context, p_2digit], dim=-1)  # (B, 257)

        return self.classifier(context)  # (B, 100)

    # ------------------------------------------------------------------
    # Convenience: number of trainable parameters
    # ------------------------------------------------------------------
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
