import argparse
import os
from pathlib import Path

import torch
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast


class ZydraConfig(LlamaConfig):
    model_type = "zydra"


class ZydraForCausalLM(LlamaForCausalLM):
    config_class = ZydraConfig


def main():
    parser = argparse.ArgumentParser(description="Chat terminal live dengan model Zydra.")
    parser.add_argument("--model", default="output/zydra-v2-full",
                        help="Path ke direktori model (relatif dari root).")
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[error] Model tidak ditemukan di: {model_path.resolve()}")
        print("Jalankan dari root repo (folder yang berisi 'output/').")
        return 1

    print(f"[info] Memuat model dari: {model_path}")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(model_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(min(2, os.cpu_count() or 1))

    try:
        model = ZydraForCausalLM.from_pretrained(str(model_path)).to(device)
    except Exception as e:
        print(f"[warn] Gagal load sebagai ZydraForCausalLM ({e}).")
        print("[warn] Coba fallback ke LlamaForCausalLM...")
        from transformers import LlamaForCausalLM as _Llama
        model = _Llama.from_pretrained(str(model_path)).to(device)
    model.eval()

    context_limit = getattr(model.config, "max_position_embeddings", 4096)
    print(f"[info] Model siap di {device} | context_limit={context_limit} | vocab={model.config.vocab_size}")

    print("\n=== Zydra Live Chat ===")
    print("Perintah: /reset (hapus riwayat), keluar (exit)")
    print("Ketik pesan Anda dan tekan Enter.\n")

    history = []
    while True:
        try:
            user_text = input("Kamu> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user_text.lower() in {"keluar", "/exit", "/quit"}:
            break
        if user_text == "/reset":
            history = []
            print("Riwayat dihapus.")
            continue
        if not user_text:
            continue

        history_text = "".join(
            f"{item['user']}\n{item['assistant']}\n" for item in history[-4:]
        )
        prompt = f"{history_text}{user_text}\n"
        input_tokens = tokenizer.encode(prompt, add_special_tokens=False)[-(context_limit - 1):]
        if tokenizer.bos_token_id is not None:
            input_tokens.insert(0, tokenizer.bos_token_id)
        input_ids = torch.tensor([input_tokens], dtype=torch.long, device=device)
        generation_limit = min(args.max_new_tokens, context_limit - input_ids.shape[1])
        if generation_limit < 1:
            print("Zydra> Konteks penuh. Ketik /reset untuk memulai percakapan baru.")
            continue

        generation_args = {
            "input_ids": input_ids,
            "max_new_tokens": generation_limit,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "do_sample": args.temperature > 0,
        }
        if args.temperature > 0:
            generation_args.update(temperature=args.temperature, top_p=args.top_p,
                                   repetition_penalty=1.3)

        with torch.inference_mode():
            output_ids = model.generate(**generation_args)
        raw = tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True).strip()

        if not raw:
            print("Zydra> (model tidak menghasilkan teks. Coba prompt lain.)")
            continue
        print(f"Zydra> {raw}")
        history.append({"user": user_text, "assistant": raw})


if __name__ == "__main__":
    raise SystemExit(main())