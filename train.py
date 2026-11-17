#!/usr/bin/env python3

import argparse
import json
import math
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    from tokenizers import Tokenizer, decoders, models, trainers
    from tokenizers.pre_tokenizers import ByteLevel
    from transformers import PreTrainedTokenizerFast
except ImportError:
    Tokenizer = PreTrainedTokenizerFast = None

try:
    from huggingface_hub import hf_hub_download, list_repo_files
except ImportError:
    hf_hub_download = list_repo_files = None


                                                                                

@dataclass
class ZydraConfig:
    model_type: str = "zydra"
    vocab_size: int = 4096
    hidden_size: int = 384
    intermediate_size: int = 1024
    num_hidden_layers: int = 6
    num_attention_heads: int = 6
    num_key_value_heads: int = 2
    max_position_embeddings: int = 384
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    attention_dropout: float = 0.0
    residual_dropout: float = 0.0
    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token_id: int = 3

    def validate(self):
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden-size harus habis dibagi heads")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("heads harus habis dibagi kv-heads")
        if (self.hidden_size // self.num_attention_heads) % 2:
            raise ValueError("head dimension harus genap untuk RoPE")
        if self.num_hidden_layers < 1 or self.max_position_embeddings < 8:
            raise ValueError("layers harus positif dan block-size minimal 8")

    @classmethod
    def load(cls, path):
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(**{k: v for k, v in raw.items() if k in cls.__dataclass_fields__})

    def save(self, path):
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")


class ZydraRMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight).to(dtype)


def rotate_half(x):
    a, b = x.chunk(2, dim=-1)
    return torch.cat((-b, a), dim=-1)


class ZydraRoPE(nn.Module):
    def __init__(self, head_dim, theta):
        super().__init__()
        inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv, persistent=False)

    def forward(self, length, device, dtype):
        pos = torch.arange(length, device=device, dtype=self.inv_freq.dtype)
        freq = torch.outer(pos, self.inv_freq.to(device))
        emb = torch.cat((freq, freq), dim=-1)
        return emb.cos().to(dtype)[None, None], emb.sin().to(dtype)[None, None]


class ZydraAttention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.heads = c.num_attention_heads
        self.kv_heads = c.num_key_value_heads
        self.dim = c.hidden_size // c.num_attention_heads
        self.dropout = c.attention_dropout
        self.q = nn.Linear(c.hidden_size, self.heads * self.dim, bias=False)
        self.k = nn.Linear(c.hidden_size, self.kv_heads * self.dim, bias=False)
        self.v = nn.Linear(c.hidden_size, self.kv_heads * self.dim, bias=False)
        self.o = nn.Linear(c.hidden_size, c.hidden_size, bias=False)
        self.rope = ZydraRoPE(self.dim, c.rope_theta)

    def forward(self, x, attention_mask=None):
        b, t, _ = x.shape
        q = self.q(x).view(b, t, self.heads, self.dim).transpose(1, 2)
        k = self.k(x).view(b, t, self.kv_heads, self.dim).transpose(1, 2)
        v = self.v(x).view(b, t, self.kv_heads, self.dim).transpose(1, 2)
        cos, sin = self.rope(t, x.device, q.dtype)
        q, k = q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
        repeat = self.heads // self.kv_heads
        if repeat > 1:
            k, v = k.repeat_interleave(repeat, 1), v.repeat_interleave(repeat, 1)

        mask = attention_mask[:, None, None, :].bool() if attention_mask is not None else None
        p = self.dropout if self.training else 0.0
        if hasattr(F, "scaled_dot_product_attention"):
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                                dropout_p=p, is_causal=True)
        else:
            scores = q @ k.transpose(-2, -1) / math.sqrt(self.dim)
            causal = torch.tril(torch.ones(t, t, device=x.device, dtype=torch.bool))
            scores = scores.masked_fill(~causal[None, None], torch.finfo(scores.dtype).min)
            if mask is not None:
                scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
            probs = torch.softmax(scores.float(), dim=-1).to(scores.dtype)
            y = F.dropout(probs, p=p, training=self.training) @ v
        return self.o(y.transpose(1, 2).contiguous().view(b, t, -1))


class ZydraSwiGLU(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.gate = nn.Linear(c.hidden_size, c.intermediate_size, bias=False)
        self.up = nn.Linear(c.hidden_size, c.intermediate_size, bias=False)
        self.down = nn.Linear(c.intermediate_size, c.hidden_size, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class ZydraBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.norm1 = ZydraRMSNorm(c.hidden_size, c.rms_norm_eps)
        self.attn = ZydraAttention(c)
        self.norm2 = ZydraRMSNorm(c.hidden_size, c.rms_norm_eps)
        self.mlp = ZydraSwiGLU(c)
        self.dropout = nn.Dropout(c.residual_dropout)

    def forward(self, x, mask):
        x = x + self.dropout(self.attn(self.norm1(x), mask))
        return x + self.dropout(self.mlp(self.norm2(x)))


class ZydraForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        config.validate()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList([ZydraBlock(config) for _ in range(config.num_hidden_layers)])
        self.final_norm = ZydraRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, attention_mask=None, labels=None):
        x = self.token_embedding(input_ids)
        for block in self.blocks:
            x = block(x, attention_mask)
        logits = self.lm_head(self.final_norm(x))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits[:, :-1].contiguous().view(-1, logits.size(-1)),
                                   labels[:, 1:].contiguous().view(-1), ignore_index=-100)
        return {"loss": loss, "logits": logits}

    @torch.inference_mode()
    def generate(self, input_ids, max_new_tokens, eos_token_id):
        self.eval()
        for _ in range(max_new_tokens):
            ctx = input_ids[:, -self.config.max_position_embeddings:]
            nxt = self(ctx)["logits"][:, -1].argmax(-1, keepdim=True)
            input_ids = torch.cat((input_ids, nxt), dim=1)
            if torch.all(nxt == eos_token_id):
                break
        return input_ids

    def save_pretrained(self, folder):
        folder.mkdir(parents=True, exist_ok=True)
        self.config.save(folder / "config.json")
        torch.save(self.state_dict(), folder / "model.pt")

    @classmethod
    def from_pretrained(cls, folder, map_location="cpu"):
        c = ZydraConfig.load(folder / "config.json")
        model = cls(c)
        state = torch.load(folder / "model.pt", map_location=map_location, weights_only=True)
        model.load_state_dict(state)
        return model


                                                                              

def parse_txt(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return []

                                                                           
    lines = text.splitlines()
    pairs, pending_user, pending_assistant = [], None, []
    for raw in lines:
        line = raw.strip()
        m = re.match(r"^(user|human|pengguna)\s*:\s*(.*)$", line, re.I)
        if m:
            if pending_user and pending_assistant:
                pairs.append((pending_user, "\n".join(pending_assistant).strip()))
            pending_user, pending_assistant = m.group(2).strip(), []
            continue
        m = re.match(r"^(assistant|bot|zydra|asisten)\s*:\s*(.*)$", line, re.I)
        if m and pending_user is not None:
            pending_assistant.append(m.group(2).strip())
            continue
        if pending_assistant and line:
            pending_assistant.append(line)
    if pending_user and pending_assistant:
        pairs.append((pending_user, "\n".join(pending_assistant).strip()))
    if pairs:
        return [(q, a) for q, a in pairs if q and a]

                                                            
    out = []
    for block in re.split(r"\n\s*\n", text):
        lines = [x.strip() for x in block.splitlines() if x.strip()]
        if lines and re.match(r"^#{1,6}\s*(percakapan|contoh|conversation|sample)\b", lines[0], re.I):
            lines = lines[1:]
        if len(lines) < 2:
            continue
        q = re.sub(r"^(user|human|pengguna|prompt|pertanyaan)\s*:\s*", "", lines[0], flags=re.I)
        ans = [re.sub(r"^(assistant|bot|zydra|asisten|response|jawaban)\s*:\s*", "", x, flags=re.I)
               for x in lines[1:]]
        if q.strip() and " ".join(ans).strip():
            out.append((q.strip(), " ".join(ans).strip()))
    return out


def load_examples(path):
    if path.suffix.lower() != ".jsonl":
        return parse_txt(path.read_text(encoding="utf-8-sig", errors="replace"))
    out = []
    with path.open(encoding="utf-8-sig") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] {path.name}:{n} JSON tidak valid: {e}")
                continue
            if not isinstance(row, dict):
                continue
            q = row.get("instruction") or row.get("prompt") or row.get("question")
            a = row.get("output") or row.get("response") or row.get("answer")
            if not (isinstance(q, str) and isinstance(a, str)):
                raw = row.get("text")
                if isinstance(raw, str) and "\n" in raw:
                    q, a = raw.split("\n", 1)
                else:
                    continue
            q, a = q.strip(), a.strip()
            if q and a and "\ufffd" not in q + a:
                out.append((q, a))
    return out


def discover_files(folder):
    return sorted(p for p in folder.rglob("*")
                  if p.is_file() and p.suffix.lower() in {".txt", ".jsonl"}) if folder.is_dir() else []


def get_files(args):
    if args.data:
        files = args.data
    elif args.data_repo:
        if hf_hub_download is None:
            raise SystemExit("Pasang huggingface_hub: pip install huggingface_hub")
        names = list_repo_files(args.data_repo, revision=args.data_revision)
        prefix = args.data_subdir.strip("/")
        names = [n for n in names if (not prefix or n.startswith(prefix + "/"))
                 and Path(n).suffix.lower() in {".txt", ".jsonl"}]
        files = [Path(hf_hub_download(args.data_repo, filename=n, revision=args.data_revision))
                 for n in names]
    else:
        files = discover_files(args.data_dir)
    if not files:
        raise SystemExit("Tidak ada file dataset .txt/.jsonl ditemukan.")
    missing = [str(p) for p in files if not p.is_file()]
    if missing:
        raise SystemExit("File tidak ditemukan: " + ", ".join(missing))
    return files


def report_raw_size(files):
    total = 0
    print("\n========== UKURAN RAW DATASET ==========")
    for p in files:
        size = p.stat().st_size
        total += size
        print(f"{p.name}: {size / 1024**2:.2f} MB")
    print(f"TOTAL: {total / 1024**2:.2f} MB | {total / 1024**3:.4f} GB")
    print("========================================\n")


def prepare_data(files, output):
    rows, seen = [], set()
    for p in files:
        loaded = load_examples(p)
        count = 0
        for row in loaded:
            if row not in seen:
                seen.add(row)
                rows.append(row)
                count += 1
        print(f"[DATA] {p.name}: {len(loaded)} terbaca, {count} unik")
    if not rows:
        raise SystemExit("Tidak ada pasangan Q/A valid. Cek format file TXT.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for q, a in rows:
            f.write(json.dumps({"instruction": q, "output": a}, ensure_ascii=False) + "\n")
    print(f"[SAVED] {len(rows)} contoh -> {output}")


                                                                               

def train_tokenizer(rows, vocab_size, folder):
    if Tokenizer is None:
        raise SystemExit("Pasang tokenizers dan transformers terlebih dahulu.")
    tok = Tokenizer(models.BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    special = ["[UNK]", "[BOS]", "[EOS]", "[PAD]"]
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=2,
                                  special_tokens=special, initial_alphabet=ByteLevel.alphabet())
    tok.train_from_iterator((s for q, a in rows for s in (q, a)), trainer)
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]",
                                   bos_token="[BOS]", eos_token="[EOS]", pad_token="[PAD]")
    fast.save_pretrained(folder)
    return fast


class QADataset(Dataset):
    def __init__(self, rows, tokenizer, block):
        self.rows, self.skipped = [], 0
        for q, a in rows:
            qids = tokenizer.encode(q + "\n", add_special_tokens=False)[-(block // 2):]
            aids = tokenizer.encode(a, add_special_tokens=False)[:max(0, block - len(qids) - 2)]
            if not qids or not aids:
                self.skipped += 1
                continue
            ids = [tokenizer.bos_token_id] + qids + aids + [tokenizer.eos_token_id]
            labels = [-100] * (1 + len(qids)) + aids + [tokenizer.eos_token_id]
            self.rows.append((ids, labels))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def collator(tokenizer):
    def batchify(rows):
        n, width = len(rows), max(len(x[0]) for x in rows)
        ids = torch.full((n, width), tokenizer.pad_token_id, dtype=torch.long)
        labels = torch.full((n, width), -100, dtype=torch.long)
        mask = torch.zeros((n, width), dtype=torch.long)
        for i, (x, y) in enumerate(rows):
            ids[i, :len(x)] = torch.tensor(x)
            labels[i, :len(y)] = torch.tensor(y)
            mask[i, :len(x)] = 1
        return {"input_ids": ids, "labels": labels, "attention_mask": mask}
    return batchify


                                                                               

def evaluate(model, loader, device):
    model.eval()
    total, count = 0.0, 0
    with torch.inference_mode():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch)["loss"]
            if torch.isfinite(loss):
                total += loss.item()
                count += 1
    return total / max(1, count)


def save_checkpoint(model, tokenizer, optimizer, scheduler, folder, epoch, best):
    folder.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(folder)
    tokenizer.save_pretrained(folder)
    torch.save(optimizer.state_dict(), folder / "optimizer.pt")
    torch.save(scheduler.state_dict(), folder / "scheduler.pt")
    torch.save({"epoch": epoch, "best_val_loss": best,
                "python_rng": random.getstate(), "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
               folder / "train_state.pt")


def sample(model, tokenizer, prompts, device, count=48):
    model.eval()
    print("[SAMPLE]")
    for prompt in prompts:
        ids = [tokenizer.bos_token_id] + tokenizer.encode(prompt, add_special_tokens=False)
        ids = ids[-(model.config.max_position_embeddings - 1):]
        x = torch.tensor([ids], dtype=torch.long, device=device)
        y = model.generate(x, count, tokenizer.eos_token_id)
        answer = tokenizer.decode(y[0, len(ids):], skip_special_tokens=True)
        print(f"Q: {prompt[:90]}\nA: {answer[:180] or '(kosong)'}")


def main():
    p = argparse.ArgumentParser(description="Native Zydra Transformer training")
    p.add_argument("--data", type=Path, nargs="+")
    p.add_argument("--data-repo", default="fareld3v-hub/zydra-datasheet")
    p.add_argument("--data-subdir", default="datasheet")
    p.add_argument("--data-revision", default="main")
    p.add_argument("--data-dir", type=Path, default=Path("datasheet"))
    p.add_argument("--prepare-data-only", action="store_true")
    p.add_argument("--prepared-output", type=Path, default=Path("data/chat_train_plain.jsonl"))
    p.add_argument("--model-dir", type=Path, default=Path("output/zydra-v2-full"))
    p.add_argument("--resume", action="store_true")
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--export-gguf", action="store_true")
    p.add_argument("--vocab-size", type=int, default=4096)
    p.add_argument("--block-size", type=int, default=384)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--hidden-size", type=int, default=384)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--kv-heads", type=int, default=2)
    p.add_argument("--intermediate-size", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--gradient-accumulation-steps", type=int, default=2)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--validation-fraction", type=float, default=0.05)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--no-sample", action="store_true")
    args = p.parse_args()

    if args.export_gguf:
        raise SystemExit("GGUF belum didukung: arsitektur native Zydra memerlukan converter Zydra di llama.cpp.")
    if args.resume and args.fresh:
        p.error("--resume dan --fresh tidak boleh dipakai bersamaan.")
    if args.epochs < 1 or args.batch_size < 1 or args.gradient_accumulation_steps < 1:
        p.error("epochs, batch-size, dan gradient accumulation harus positif.")
    if args.block_size < 8 or args.vocab_size < 260:
        p.error("block-size minimal 8 dan vocab-size minimal 260.")
    if not 0 < args.validation_fraction < 1:
        p.error("validation-fraction harus di antara 0 dan 1.")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    files = args.data
    if files is None:
        if args.data_repo:
            if list_repo_files is None:
                raise SystemExit("Pasang huggingface_hub atau gunakan --data-repo '' untuk dataset lokal.")
            names = list_repo_files(args.data_repo, revision=args.data_revision)
            prefix = args.data_subdir.strip("/")
            names = [n for n in names if (not prefix or n.startswith(prefix + "/"))
                     and Path(n).suffix.lower() in {".txt", ".jsonl"}]
            if not names:
                raise SystemExit("Tidak ada file TXT/JSONL pada subfolder repo.")
            files = [Path(hf_hub_download(args.data_repo, filename=n, revision=args.data_revision))
                     for n in names]
        else:
            files = discover_files(args.data_dir)
    if not files:
        raise SystemExit("Dataset kosong. Gunakan --data atau --data-dir.")
    if any(not f.is_file() for f in files):
        raise SystemExit("Ada file dataset yang tidak ditemukan.")
    report_raw_size(files)

    if args.prepare_data_only:
        prepare_data(files, args.prepared_output)
        return

    if args.fresh and args.model_dir.exists() and any(args.model_dir.iterdir()):
        raise SystemExit("Folder model sudah berisi file. Gunakan --model-dir baru untuk --fresh.")

    rows, seen = [], set()
    for f in files:
        parsed = load_examples(f)
        before = len(rows)
        for pair in parsed:
            if pair not in seen:
                seen.add(pair)
                rows.append(pair)
        print(f"[DATA] {f.name}: {len(parsed)} dibaca, {len(rows)-before} unik")
    if len(rows) < 2:
        raise SystemExit("Kurang dari dua pasangan data valid; cek format dataset.")
    if args.max_samples:
        rows = random.Random(args.seed).sample(rows, min(args.max_samples, len(rows)))
    random.Random(args.seed).shuffle(rows)
    nv = max(1, int(len(rows) * args.validation_fraction))
    val_rows, train_rows = rows[:nv], rows[nv:]
    print(f"[SPLIT] train={len(train_rows)} validation={len(val_rows)}")

    args.model_dir.mkdir(parents=True, exist_ok=True)
    exists = (args.model_dir / "config.json").exists() and (args.model_dir / "model.pt").exists()
    resume = args.resume or (exists and not args.fresh)
    if args.resume and not exists:
        raise SystemExit("Checkpoint Zydra untuk resume tidak ditemukan.")
    if resume:
        from transformers import PreTrainedTokenizerFast as FastTokenizer
        tokenizer = FastTokenizer.from_pretrained(args.model_dir)
        model = ZydraForCausalLM.from_pretrained(args.model_dir)
        args.block_size = model.config.max_position_embeddings
        print("[MODEL] Memuat checkpoint native Zydra.")
    else:
        tokenizer = train_tokenizer(train_rows, args.vocab_size, args.model_dir)
        config = ZydraConfig(
            vocab_size=len(tokenizer), hidden_size=args.hidden_size,
            intermediate_size=args.intermediate_size, num_hidden_layers=args.layers,
            num_attention_heads=args.heads, num_key_value_heads=args.kv_heads,
            max_position_embeddings=args.block_size, bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
        )
        model = ZydraForCausalLM(config)
        print("[MODEL] Inisialisasi arsitektur Zydra baru.")

    print(f"[ARCH] Zydra | layers={model.config.num_hidden_layers} hidden={model.config.hidden_size} "
          f"heads={model.config.num_attention_heads} kv={model.config.num_key_value_heads} "
          f"vocab={model.config.vocab_size} context={model.config.max_position_embeddings}")
    print(f"[PARAMS] {sum(x.numel() for x in model.parameters()):,}")

    train_ds = QADataset(train_rows, tokenizer, args.block_size)
    val_ds = QADataset(val_rows, tokenizer, args.block_size)
    print(f"[TOKENIZED] train={len(train_ds)} skipped={train_ds.skipped} "
          f"val={len(val_ds)} skipped={val_ds.skipped}")
    if not len(train_ds) or not len(val_ds):
        raise SystemExit("Tidak ada contoh tersisa setelah tokenisasi. Periksa block-size/data.")

    collate = collator(tokenizer)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay, betas=(0.9, 0.95))
    updates = math.ceil(len(train_loader) / args.gradient_accumulation_steps) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, updates))
    start_epoch, best_loss = 0, float("inf")
    if resume and (args.model_dir / "train_state.pt").exists():
        state = torch.load(args.model_dir / "train_state.pt", map_location="cpu", weights_only=False)
        try:
            optimizer.load_state_dict(torch.load(args.model_dir / "optimizer.pt", map_location=device, weights_only=True))
            scheduler.load_state_dict(torch.load(args.model_dir / "scheduler.pt", map_location=device, weights_only=True))
        except Exception as e:
            print(f"[WARN] Optimizer/scheduler tidak dipulihkan: {e}")
        start_epoch = int(state.get("epoch", 0))
        best_loss = float(state.get("best_val_loss", float("inf")))
        random.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"])
        if torch.cuda.is_available() and state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        print(f"[RESUME] Mulai epoch {start_epoch + 1}")
    elif resume:
        print("[RESUME] Bobot dimuat; optimizer dimulai ulang.")

    prompts = [q for q, _ in train_rows[:3]]
    print(f"[TRAIN] device={device} batch={args.batch_size} "
          f"effective_batch={args.batch_size * args.gradient_accumulation_steps}")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total, n = 0.0, len(train_loader)
        for i, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch)["loss"]
            group_start = (i // args.gradient_accumulation_steps) * args.gradient_accumulation_steps
            group_n = min(args.gradient_accumulation_steps, n - group_start)
            (loss / group_n).backward()
            total += loss.item()
            update = ((i + 1) % args.gradient_accumulation_steps == 0 or i + 1 == n)
            if update:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if (i + 1) % 100 == 0 or i + 1 == n:
                print(f"Epoch {epoch+1}/{args.epochs} batch {i+1}/{n} "
                      f"loss={loss.item():.4f} lr={scheduler.get_last_lr()[0]:.2e}", flush=True)

        train_loss = total / max(1, n)
        val_loss = evaluate(model, val_loader, device)
        improved = val_loss < best_loss
        if improved:
            best_loss = val_loss
            best_dir = args.model_dir.parent / (args.model_dir.name + "-best")
            best_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            (best_dir / "metrics.json").write_text(
                json.dumps({"epoch": epoch + 1, "validation_loss": best_loss}, indent=2),
                encoding="utf-8")
        print(f"[EPOCH {epoch+1}] train_loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f} gap={val_loss-train_loss:+.4f}")
        if not args.no_sample:
            sample(model, tokenizer, prompts, device)
        save_checkpoint(model, tokenizer, optimizer, scheduler, args.model_dir,
                        epoch + 1, best_loss)
        print(f"[SAVE] latest={args.model_dir} best={args.model_dir.parent / (args.model_dir.name + '-best')}")

    print(f"Training selesai. Best validation loss={best_loss:.4f}")


if __name__ == "__main__":
    main()
