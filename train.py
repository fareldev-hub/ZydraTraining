import argparse
import hashlib
import json
import math
import random
import re
import shutil
import subprocess
import sys
import tempfile
import torch
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files
from transformers import LlamaConfig, LlamaForCausalLM

MODEL_DIR = Path("output/zydra-v2-full")
torch.set_num_threads(4)


class ZydraConfig(LlamaConfig):
    model_type = "zydra"


class ZydraForCausalLM(LlamaForCausalLM):
    config_class = ZydraConfig


try:
    from transformers import AutoConfig, AutoModelForCausalLM

    AutoConfig.register(ZydraConfig.model_type, ZydraConfig)
    AutoModelForCausalLM.register(ZydraConfig, ZydraForCausalLM)
except ValueError:
    pass


def export_gguf(model_dir, llama_cpp_dir, outfile, outtype):
    from transformers import AutoTokenizer

    check_text = (
        "\n \n\n \n\n\n \t \t\t \t\n  \n   \n    \n     \n🚀 (normal) 😶\u200d🌫️ "
        "(multiple emojis concatenated) ✅ 🦙🦙 3 33 333 3333 33333 333333 "
        "3333333 33333333 3.3 3..3 3...3 កាន់តែពិសេសអាច😁 ?我想在apple工作1314151天～ "
        "------======= нещо на Български ''''''```````\"\"\"\"......!!!!!!?????? "
        "I've been 'told he's there, 'RE you sure? 'M not sure I'll make it, 'D you like some tea? We'Ve a'lL"
    )
    model_dir = model_dir.resolve()
    llama_cpp_dir = llama_cpp_dir.resolve()
    converter_path = llama_cpp_dir / "convert_hf_to_gguf.py"
    if not model_dir.is_dir() or not (model_dir / "config.json").is_file() or not (model_dir / "model.safetensors").is_file():
        raise SystemExit(f"Checkpoint tidak lengkap: {model_dir}")
    if not converter_path.is_file():
        raise SystemExit(f"Converter llama.cpp tidak ditemukan: {converter_path}")
    outfile.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="zydra-gguf-") as temporary:
        root = Path(temporary)
        staged_model = root / "hf-model"
        converter_root = root / "converter"
        converter_root.mkdir()
        shutil.copytree(model_dir, staged_model)
        config_path = staged_model / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["model_type"] = "llama"
        config["architectures"] = ["LlamaForCausalLM"]
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        tokenizer = AutoTokenizer.from_pretrained(staged_model)
        token_hash = hashlib.sha256(str(tokenizer.encode(check_text)).encode()).hexdigest()
        shutil.copy2(converter_path, converter_root / converter_path.name)
        shutil.copytree(llama_cpp_dir / "conversion", converter_root / "conversion")
        shutil.copytree(llama_cpp_dir / "gguf-py", converter_root / "gguf-py")
        base_path = converter_root / "conversion" / "base.py"
        source = base_path.read_text(encoding="utf-8")
        insertion = '        if res is None:\n            logger.warning("\\n")'
        if insertion not in source:
            raise SystemExit("Layout converter llama.cpp berubah; tokenizer Zydra tidak dapat didaftarkan.")
        mapping = f'        if chkhsh == "{token_hash}":\n            res = "gpt-2"\n\n'
        base_path.write_text(source.replace(insertion, mapping + insertion, 1), encoding="utf-8")
        command = [
            sys.executable, str(converter_root / converter_path.name), str(staged_model),
            "--outfile", str(outfile.resolve()), "--outtype", outtype,
            "--model-name", "Zydra",
        ]
        subprocess.run(command, check=True)
    print(f"GGUF tersimpan di: {outfile}")


def parse_qa_blocks(text):
    rows = []
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) < 2:
            continue
        question = lines[0]
        answer = " ".join(lines[1:]).strip()
        if question and answer:
            rows.append((question, answer))
    return rows


def load_examples(path):
    if path.suffix.lower() == ".jsonl":
        examples = []
        with path.open(encoding="utf-8-sig") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"JSON tidak valid pada baris {line_number}: {error}") from error
                if not isinstance(record, dict):
                    continue
                instruction = record.get("instruction") or record.get("prompt") or record.get("question")
                answer = record.get("output") or record.get("response") or record.get("answer")
                if isinstance(instruction, str) and isinstance(answer, str):
                    prompt, completion = instruction.strip(), answer.strip()
                else:
                    text = record.get("text")
                    if not isinstance(text, str) or "\n" not in text:
                        continue
                    prompt, completion = (part.strip() for part in text.split("\n", 1))
                if "\ufffd" in completion:
                    continue
                if len(completion) > 180 and not re.search(r"[.!?](?:[\])\"']*)?\s*$", completion):
                    sentence_ends = list(re.finditer(r"[.!?](?:[\])\"']*)?(?=\s|$)", completion))
                    if not sentence_ends or sentence_ends[-1].end() < 60:
                        continue
                    completion = completion[:sentence_ends[-1].end()].strip()
                if prompt and completion:
                    examples.append((prompt, completion))
        if len(examples) < 2:
            raise ValueError(f"Minimal perlu 2 pasangan instruksi-jawaban yang valid: {path}")
        return examples

    return parse_qa_blocks(path.read_text(encoding="utf-8-sig"))


def discover_data_files(data_dir):
    if not data_dir.is_dir():
        return []
    return sorted(
        path for path in data_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".txt", ".jsonl"}
    )


def download_data_files(repo_id, subdir, revision):
    try:
        all_files = list_repo_files(repo_id, revision=revision)
    except Exception as error:
        raise SystemExit(f"Gagal mengakses repo {repo_id}: {error}") from error

    prefix = f"{subdir.strip('/')}/" if subdir else ""
    targets = [
        f for f in all_files
        if f.startswith(prefix) and Path(f).suffix.lower() in {".txt", ".jsonl"}
    ]
    if not targets:
        raise SystemExit(
            f"Tidak ada file .txt/.jsonl di {repo_id}/{subdir} (revision={revision})"
        )

    print(f"Mengunduh {len(targets)} file dari {repo_id}/{subdir} ...")
    local_paths = []
    for remote in targets:
        local = hf_hub_download(
            repo_id=repo_id,
            filename=remote,
            revision=revision,
        )
        local_paths.append(Path(local))
    return local_paths


def prepare_dataset_from_files(files, output):
    rows = []
    seen = set()
    for path in files:
        try:
            parsed = load_examples(path)
        except Exception as error:
            print(f"[SKIP] {path.name}: {error}")
            continue
        added = 0
        for prompt, answer in parsed:
            row = (prompt, answer)
            if row not in seen:
                seen.add(row)
                rows.append(row)
                added += 1
        print(f"[OK] {path.name}: {added} baris unik")
    if not rows:
        raise SystemExit("Tidak ada baris valid ditemukan.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as destination:
        for prompt, answer in rows:
            destination.write(json.dumps({"instruction": prompt, "output": answer}, ensure_ascii=False) + "\n")
    print(f"Total: {len(rows)} baris; output: {output}")


def build_tokenizer(texts, vocab_size, output_dir):
    from tokenizers import Tokenizer, decoders, models, trainers
    from tokenizers.pre_tokenizers import ByteLevel
    from transformers import PreTrainedTokenizerFast

    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    special_tokens = ["[UNK]", "[BOS]", "[EOS]", "[PAD]"]
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=special_tokens,
        initial_alphabet=ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        bos_token="[BOS]",
        eos_token="[EOS]",
        pad_token="[PAD]",
    )
    fast_tokenizer.save_pretrained(output_dir)
    return fast_tokenizer


def distill_teacher_data(data_paths, output_path, teacher_name, limit, max_new_tokens, seed):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if limit < 1 or max_new_tokens < 1:
        raise SystemExit("Batas distilasi dan token teacher harus positif")
    groups = []
    for path in data_paths:
        if path.resolve() == output_path.resolve():
            continue
        rows = load_examples(path)
        if rows:
            groups.append((path, rows))
    if not groups:
        raise SystemExit("Tidak ada contoh sumber valid untuk distilasi")

    rng = random.Random(seed)
    selected = []
    budget = max(1, math.ceil(limit / len(groups)))
    for path, rows in groups:
        sample = rng.sample(rows, min(len(rows), budget))
        selected.extend((path, prompt, answer) for prompt, answer in sample)
    selected = selected[:limit]

    existing_prompts = set()
    if output_path.is_file():
        with output_path.open(encoding="utf-8-sig") as source:
            for line in source:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict) and isinstance(record.get("instruction"), str):
                    existing_prompts.add(record["instruction"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cpu" else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(teacher_name)
    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_name,
        dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device).eval()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output_path.open("a", encoding="utf-8") as destination:
        for index, (source_path, prompt, reference) in enumerate(selected, start=1):
            distilled_prompt = prompt.strip()
            if not distilled_prompt or distilled_prompt in existing_prompts:
                continue
            messages = [
                {
                    "role": "system",
                    "content": (
                        "Kamu asisten berbahasa Indonesia. Tulis SATU jawaban akhir yang langsung menjawab permintaan. "
                        "Gunakan materi rujukan untuk menjaga fakta dan maksud; jangan menyebut rujukan, datasheet, atau nama file. "
                        "Jangan mengulang label, pertanyaan, atau menambahkan topik yang tidak diminta. Jangan tampilkan chain-of-thought. "
                        "Jangan keluarkan kode kecuali pengguna meminta kode."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Permintaan pengguna: {prompt}\nMateri: {reference}\nJawaban akhir:",
                },
            ]
            if source_path.name == "coding_multilanguage.jsonl":
                messages[0]["content"] += " Untuk permintaan pemrograman, keluarkan kode lengkap dalam code fence bahasa yang sesuai."
            encoded = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(device)
            with torch.inference_mode():
                generated = teacher.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )
            answer = tokenizer.decode(generated[0, input_ids.shape[1]:], skip_special_tokens=True).strip()
            answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()
            answer = re.sub(r"^(?:jawaban(?: akhir)?|hasil akhir)\s*:\s*", "", answer, flags=re.IGNORECASE).strip()
            answer = re.sub(r"\b[\w./\\-]+\.(?:txt|jsonl)\b", "sumber internal", answer, flags=re.IGNORECASE)
            if (
                not answer
                or "\ufffd" in answer
                or re.match(r"^(?:materi|permintaan pengguna)\s*:", answer, re.IGNORECASE)
                or (source_path.name != "coding_multilanguage.jsonl" and "```" in answer)
                or len(answer) > 500
            ):
                continue
            destination.write(json.dumps({
                "instruction": distilled_prompt,
                "output": answer,
                "category": "teacher_distilled",
                "source": "offline_teacher",
            }, ensure_ascii=False) + "\n")
            existing_prompts.add(distilled_prompt)
            written += 1
            if index % 16 == 0:
                print(f"Distilasi {index}/{len(selected)}; tersimpan {written}", flush=True)

    del teacher
    print(f"Data distilasi baru: {written}; file: {output_path}")
    return written


def sample_generation(model, tokenizer, prompts, device, max_new_tokens=64):
    import torch

    model.eval()
    results = []
    for prompt in prompts:
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        if tokenizer.bos_token_id is not None:
            ids = [tokenizer.bos_token_id] + ids
        ids = ids[-(model.config.max_position_embeddings - 1):]
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.inference_mode():
            out = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        text = tokenizer.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True).strip()
        results.append((prompt, text))
    model.train()
    return results


def main():
    parser = argparse.ArgumentParser(description="Siapkan dataset atau latih model Zydra.")
    parser.add_argument(
        "--data",
        type=Path,
        nargs="+",
        default=None,
        help="File dataset spesifik. Jika tidak diisi, file diambil otomatis dari repo HF.",
    )
    parser.add_argument("--resume", action="store_true", help="Lanjutkan checkpoint + optimizer state.")
    parser.add_argument("--export-gguf", action="store_true", help="Ekspor checkpoint, tidak menjalankan training.")
    parser.add_argument("--model", type=Path, default=Path("output/zydra-v2-full"))
    parser.add_argument("--llama-cpp", type=Path, default=Path("llama.cpp"))
    parser.add_argument("--outfile", type=Path, default=Path("model/zydra-v2-f16.gguf"))
    parser.add_argument("--outtype", choices=("f32", "f16", "bf16", "q8_0", "tq1_0", "tq2_0", "auto"), default="f16")
    parser.add_argument("--prepare-data-only", action="store_true", help="Gabungkan datasheet ke data/chat_train_plain.jsonl lalu keluar.")
    parser.add_argument("--data-repo", default="fareld3v-hub/zydra-datasheet", help="Repo HF datasheet; set '' untuk pakai --data-dir lokal.")
    parser.add_argument("--data-subdir", default="datasheet", help="Subfolder di dalam repo HF.")
    parser.add_argument("--data-revision", default="main", help="Branch/revision repo HF.")
    parser.add_argument("--data-dir", type=Path, default=Path("datasheet"))
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--block-size", type=int, default=384)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--hidden-size", type=int, default=384)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--intermediate-size", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--max-samples", type=int, default=0, help="Batas opsional untuk smoke test.")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--distill-teacher", action="store_true", help="Buat jawaban teacher offline lalu fine-tune Zydra yang sama.")
    parser.add_argument("--teacher-model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--teacher-limit", type=int, default=128)
    parser.add_argument("--teacher-max-new-tokens", type=int, default=128)
    parser.add_argument("--teacher-output", type=Path, default=Path("data/teacher_distilled.jsonl"))
    parser.add_argument("--no-sample", action="store_true", help="Jangan print sample generation tiap epoch.")
    args = parser.parse_args()

    if not args.export_gguf and not args.prepare_data_only and args.data is None:
        if args.data_repo:
            args.data = download_data_files(
                args.data_repo, args.data_subdir, args.data_revision
            )
        else:
            args.data = discover_data_files(args.data_dir)
        if not args.data:
            parser.error("Tidak ada file .txt atau .jsonl ditemukan")
        print(f"Auto-discover {len(args.data)} file:")
        for path in args.data:
            print(f"  - {path}")

    if args.export_gguf:
        export_gguf(args.model, args.llama_cpp, args.outfile, args.outtype)
        return
    if args.prepare_data_only:
        if args.data_repo:
            files = download_data_files(args.data_repo, args.data_subdir, args.data_revision)
        else:
            files = discover_data_files(args.data_dir)
        if not files:
            raise SystemExit("Tidak ada file .txt atau .jsonl ditemukan")
        prepare_dataset_from_files(files, Path("data/chat_train_plain.jsonl"))
        return
    if args.distill_teacher:
        distill_teacher_data(
            args.data,
            args.teacher_output,
            args.teacher_model,
            args.teacher_limit,
            args.teacher_max_new_tokens,
            args.seed,
        )
        args.data = [args.teacher_output]

    missing_data = [path for path in args.data if not path.is_file()]
    if missing_data:
        parser.error(f"Dataset tidak ditemukan: {', '.join(map(str, missing_data))}")

    checkpoint_files = ("config.json", "model.safetensors", "tokenizer.json")
    has_checkpoint = all((MODEL_DIR / name).is_file() for name in checkpoint_files)
    output_has_content = MODEL_DIR.exists() and any(MODEL_DIR.iterdir())
    if args.resume and not has_checkpoint:
        parser.error(f"Checkpoint lengkap untuk resume tidak ditemukan di {MODEL_DIR}")
    if output_has_content and not has_checkpoint:
        parser.error(f"Folder model kanonis berisi data tetapi checkpoint tidak lengkap: {MODEL_DIR}")
    args.resume = args.resume or has_checkpoint

    if args.block_size < 8 or args.vocab_size < 260:
        parser.error("--block-size minimal 8 dan --vocab-size minimal 260")
    if args.epochs < 1 or args.batch_size < 1 or args.gradient_accumulation_steps < 1:
        parser.error("epoch, batch size, dan gradient accumulation harus positif")
    if not 0 < args.validation_fraction < 1:
        parser.error("--validation-fraction harus di antara 0 dan 1")

    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
    except ImportError as error:
        raise SystemExit(f"Dependensi training belum tersedia: {error}") from error

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    examples = []
    for data_path in args.data:
        data_examples = load_examples(data_path)
        print(f"Dataset {data_path}: {len(data_examples)} contoh")
        examples.extend(data_examples)

    if args.max_samples:
        if args.max_samples < 2:
            parser.error("--max-samples harus minimal 2")
        examples = random.Random(args.seed).sample(examples, min(args.max_samples, len(examples)))
    examples = list(dict.fromkeys(examples))
    random.Random(args.seed).shuffle(examples)

    validation_count = max(1, int(len(examples) * args.validation_fraction))
    validation_examples = examples[:validation_count]
    training_examples = examples[validation_count:]
    if not training_examples:
        parser.error("Dataset terlalu kecil untuk membuat training dan validation split")

    total_params = (
        args.layers * (
            (args.hidden_size * args.hidden_size * 4)
            + (args.hidden_size * args.intermediate_size * 3)
        )
        + (args.vocab_size * args.hidden_size * 2)
    )
    print(f"\n[CONFIG] layers={args.layers} hidden={args.hidden_size} vocab={args.vocab_size} "
          f"block={args.block_size} params≈{total_params/1e6:.2f}M")
    print(f"[DATA] train={len(training_examples)} val={len(validation_examples)}")
    if total_params < 5_000_000:
        print("[WARN] Model < 5M parameter. Output hanya hafalan, bukan pemahaman.")
    if len(training_examples) < 1000:
        print("[WARN] Dataset < 1000 contoh. Model akan overfit parah.")
    if args.epochs < 5:
        print("[WARN] Epochs < 5 Model kecil, butuh >=10 epoch untuk konvergen.")
    print()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    optimizer_state_path = MODEL_DIR / "optimizer.pt"
    scheduler_state_path = MODEL_DIR / "scheduler.pt"

    if args.resume:
        from transformers import PreTrainedTokenizerFast

        tokenizer = PreTrainedTokenizerFast.from_pretrained(MODEL_DIR)
        model = ZydraForCausalLM.from_pretrained(MODEL_DIR)
        args.block_size = model.config.max_position_embeddings
        print(f"Melanjutkan checkpoint: {MODEL_DIR}")
    else:
        all_texts = (f"{prompt}\n{answer}" for prompt, answer in examples)
        tokenizer = build_tokenizer(all_texts, args.vocab_size, MODEL_DIR)
        config = ZydraConfig(
            vocab_size=len(tokenizer), max_position_embeddings=args.block_size,
            hidden_size=args.hidden_size, intermediate_size=args.intermediate_size,
            num_hidden_layers=args.layers, num_attention_heads=args.heads,
            num_key_value_heads=args.kv_heads, bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
        )
        model = ZydraForCausalLM(config)
        print(f"[ACTUAL] params = {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    class InstructionDataset(Dataset):
        def __init__(self, rows):
            self.rows = []
            self.skipped = 0
            for prompt, answer in rows:
                prompt_ids = tokenizer.encode(f"{prompt}\n", add_special_tokens=False)
                answer_ids = tokenizer.encode(answer, add_special_tokens=False)
                prompt_ids = prompt_ids[-max(1, args.block_size // 2):]
                answer_ids = answer_ids[:max(1, args.block_size - len(prompt_ids) - 2)]
                input_ids = [tokenizer.bos_token_id] + prompt_ids + answer_ids + [tokenizer.eos_token_id]
                prompt_length = 1 + len(prompt_ids)
                labels = [-100] * prompt_length + input_ids[prompt_length:]
                if any(label != -100 for label in labels):
                    self.rows.append((input_ids, labels))
                else:
                    self.skipped += 1

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            return self.rows[index]

    def collate_batch(rows):
        max_length = max(len(input_ids) for input_ids, _ in rows)
        input_ids = torch.full((len(rows), max_length), tokenizer.pad_token_id, dtype=torch.long)
        labels = torch.full((len(rows), max_length), -100, dtype=torch.long)
        attention_mask = torch.zeros((len(rows), max_length), dtype=torch.long)
        for row_index, (ids, row_labels) in enumerate(rows):
            length = len(ids)
            input_ids[row_index, :length] = torch.tensor(ids, dtype=torch.long)
            labels[row_index, :length] = torch.tensor(row_labels, dtype=torch.long)
            attention_mask[row_index, :length] = 1
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}

    training_data = InstructionDataset(training_examples)
    validation_data = InstructionDataset(validation_examples)
    print(f"[TOKENIZED] train={len(training_data)} (skip {training_data.skipped}) "
          f"val={len(validation_data)} (skip {validation_data.skipped})")
    if not training_data or not validation_data:
        parser.error("Tidak ada contoh valid setelah tokenisasi dan pemotongan")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    train_loader = DataLoader(training_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
    validation_loader = DataLoader(validation_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)
    updates_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    total_updates = updates_per_epoch * args.epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: max(0.05, 1.0 - step / max(1, total_updates))
    )

    start_epoch = 0
    best_val_loss = float("inf")
    if args.resume and optimizer_state_path.is_file() and scheduler_state_path.is_file():
        try:
            optimizer.load_state_dict(torch.load(optimizer_state_path, map_location=device))
            scheduler.load_state_dict(torch.load(scheduler_state_path, map_location=device))
            meta_path = MODEL_DIR / "train_meta.json"
            if meta_path.is_file():
                meta = json.loads(meta_path.read_text())
                start_epoch = meta.get("epoch", 0)
                best_val_loss = meta.get("best_val_loss", float("inf"))
            print(f"[RESUME] optimizer + scheduler dimuat. Lanjut dari epoch {start_epoch + 1}")
        except Exception as e:
            print(f"[WARN] Gagal resume optimizer state: {e}. Mulai optimizer fresh.")
    elif args.resume:
        print("[RESUME] Optimizer state tidak ada, mulai optimizer fresh (model weight tetap dilanjutkan).")

    prompts_for_sample = [p for p, _ in training_examples[:3]]
    print(f"\nMulai training: {len(training_data)} train, {len(validation_data)} validation, device={device}")
    print(f"Sample prompt untuk cek belajar: {prompts_for_sample}\n")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        accumulated = 0
        epoch_loss = 0.0
        loss_count = 0
        for batch_index, batch in enumerate(train_loader):
            batch = {key: value.to(device) for key, value in batch.items()}
            output = model(**batch)
            group_start = (batch_index // args.gradient_accumulation_steps) * args.gradient_accumulation_steps
            group_size = min(args.gradient_accumulation_steps, len(train_loader) - group_start)
            (output.loss / group_size).backward()
            accumulated += 1
            epoch_loss += output.loss.item()
            loss_count += 1
            if accumulated == group_size:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                accumulated = 0
            if (batch_index + 1) % 100 == 0:
                print(
                    f"Epoch {epoch + 1}/{args.epochs}: batch {batch_index + 1}/{len(train_loader)}; "
                    f"loss={output.loss.item():.4f}; lr={scheduler.get_last_lr()[0]:.2e}",
                    flush=True,
                )
        train_loss = epoch_loss / max(1, loss_count)

        model.eval()
        validation_loss = 0.0
        with torch.inference_mode():
            for batch in validation_loader:
                batch = {key: value.to(device) for key, value in batch.items()}
                validation_loss += model(**batch).loss.item()
        validation_loss /= max(1, len(validation_loader))

        gap = validation_loss - train_loss
        overfit_flag = " (OVERFIT)" if gap > 0.5 else ""
        print(f"\nEpoch {epoch + 1}/{args.epochs}: train_loss={train_loss:.4f} "
              f"val_loss={validation_loss:.4f} gap={gap:+.4f}{overfit_flag}")

        if not args.no_sample:
            samples = sample_generation(model, tokenizer, prompts_for_sample, device)
            print("[SAMPLE]")
            for prompt, text in samples:
                preview = text[:150].replace("\n", " ") if text else "(kosong)"
                print(f"  Q: {prompt[:60]}")
                print(f"  A: {preview}")
            print()

        model.config.use_cache = True
        model.save_pretrained(MODEL_DIR)
        tokenizer.save_pretrained(MODEL_DIR)
        torch.save(optimizer.state_dict(), optimizer_state_path)
        torch.save(scheduler.state_dict(), scheduler_state_path)
        meta = {"epoch": epoch + 1, "best_val_loss": min(best_val_loss, validation_loss)}
        if validation_loss < best_val_loss:
            best_val_loss = validation_loss
            meta["best_val_loss"] = best_val_loss
        (MODEL_DIR / "train_meta.json").write_text(json.dumps(meta, indent=2))
        print(f"Checkpoint epoch {epoch + 1} tersimpan di: {MODEL_DIR}\n", flush=True)

    print(f"\nTraining selesai. Model terbaik (val_loss={best_val_loss:.4f}) di {MODEL_DIR}")
    print("Test model dengan:")
    print(f"  python -c \"import torch; from transformers import AutoModelForCausalLM, AutoTokenizer; "
          f"tok = AutoTokenizer.from_pretrained('{MODEL_DIR}'); "
          f"m = AutoModelForCausalLM.from_pretrained('{MODEL_DIR}').eval(); "
          f"i = tok('Halo', return_tensors='pt'); "
          f"o = m.generate(**i, max_new_tokens=50, do_sample=False, pad_token_id=tok.eos_token_id); "
          f"print(repr(tok.decode(o[0], skip_special_tokens=True)))\"")


if __name__ == "__main__":
    main()