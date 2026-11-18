# Zydra Training

Zydra adalah *proof of concept* untuk melatih model bahasa kausal kecil dari bobot awal. Proyek ini membangun arsitektur, tokenizer BPE, dataset gabungan, dan checkpoint melalui `train.py`; ini **bukan proses fine-tuning** terhadap model bahasa yang sudah dilatih.

Implementasi saat ini mendukung:

- Inisialisasi model Zydra dari bobot acak
- Pelatihan *causal language modeling* pada pasangan instruksi/pertanyaan dan jawaban
- Dataset dari Hugging Face atau file lokal
- Melanjutkan training dari checkpoint native Zydra
- Chat interaktif melalui terminal

Zydra bukan model serbaguna dan tidak menjamin jawaban yang benar atau koheren.

## Cara Kerja

`train.py` melakukan beberapa tahap:

1. Mengambil daftar data `.txt` atau `.jsonl`.
2. Membaca pasangan pertanyaan dan jawaban.
3. Menghapus duplikat.
4. Membagi data menjadi train dan validation.
5. Melatih tokenizer BPE khusus checkpoint.
6. Menginisialisasi atau memuat model Zydra.
7. Melatih model dan menyimpan checkpoint serta metriks.

Secara default, dataset diambil dari Hugging Face:

```text
fareld3v-hub/zydra-datasheet
```

dengan subfolder `datasheet`. Untuk menggunakan dataset lokal, jalankan dengan `--data-repo ""`.

## Instalasi

Disarankan menggunakan Python 3.10 atau lebih baru.

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Linux/macOS:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

Training dapat memakai CPU atau GPU PyTorch yang terpasang. Untuk memasang build PyTorch CPU khusus Windows, instal sebelum dependensi proyek:

```powershell
.venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu torch
```

## Format Dataset

File `.jsonl` menerima salah satu kombinasi nama kolom berikut:

```json
{"instruction": "Pertanyaan", "output": "Jawaban"}
```

atau:

```json
{"prompt": "Pertanyaan", "response": "Jawaban"}
```

Kolom lain yang didukung adalah `question`/`answer` serta `text` dengan format `"pertanyaan\njawaban"`.

File `.txt` mendukung percakapan berlabel:

```text
user: Pertanyaan
assistant: Jawaban

human: Pertanyaan lain
bot: Jawaban lain
```

Format blok dua baris dengan label `User:`/`Assistant:` atau heading percakapan juga dapat dibaca.

## Menyiapkan Data

Untuk mengambil dan menggabungkan dataset Hugging Face tanpa training:

```powershell
.venv\Scripts\python.exe train.py --prepare-data-only
```

Hasil gabungan disimpan default ke:

```text
data/chat_train_plain.jsonl
```

Untuk memilih lokasi output:

```powershell
.venv\Scripts\python.exe train.py --prepare-data-only --prepared-output data/dataset.jsonl
```

Untuk memakai satu atau beberapa file lokal:

```powershell
.venv\Scripts\python.exe train.py --data-repo "" --data .\datasheet\data.jsonl
```

Atau:

```powershell
.venv\Scripts\python.exe train.py --data-repo "" --data-dir .\datasheet
```

## Training dari Awal

Training default memakai konfigurasi arsitektur Zydra berikut:

- Vocab: 4096 token
- Hidden size: 384
- 6 lapisan transformer
- 6 attention heads
- 2 KV heads
- Intermediate size: 1024
- Panjang konteks: 384 token

Jalankan:

```powershell
.venv\Scripts\python.exe train.py
```

Contoh smoke test dengan jumlah sampel terbatas:

```powershell
.venv\Scripts\python.exe train.py --max-samples 4096 --epochs 1
```

Konfigurasi penting dapat diubah melalui argument CLI:

```text
--data                 satu atau beberapa file .txt/.jsonl
--data-repo            repository Hugging Face dataset
--data-subdir          subfolder dataset di repository
--data-revision        revision/branch repository
--data-dir             folder dataset lokal
--model-dir            folder checkpoint, default output/zydra-v2-full
--fresh                wajib memakai folder model kosong
--resume               memuat checkpoint native Zydra
--epochs               jumlah target epoch
--batch-size           batch per proses
--gradient-accumulation-steps
                       mengakumulasi beberapa batch sebelum update
--learning-rate        learning rate AdamW
--weight-decay         weight decay AdamW
--validation-fraction  proporsi data validation, default 5%
--max-samples          membatasi jumlah pasangan yang dipakai
--block-size           panjang konteks
--vocab-size           ukuran vocab tokenizer baru
--hidden-size          ukuran hidden model baru
--layers               jumlah lapisan model baru
--heads                jumlah attention heads
--kv-heads             jumlah KV heads
--intermediate-size    ukuran lapisan SwiGLU
--no-sample            tidak mencetak contoh generasi tiap epoch
```

Selama training, hasil disimpan ke:

- `output/zydra-v2-full/`: checkpoint terbaru
- `output/zydra-v2-full-best/`: checkpoint terbaik berdasarkan validation loss
- `metrics.json`: epoch dan best validation loss pada checkpoint terbaik
- `train_state.pt`, `optimizer.pt`, dan `scheduler.pt`: state untuk resume

## Melanjutkan Training

Gunakan checkpoint dan tokenizer dari folder yang sama:

```powershell
.venv\Scripts\python.exe train.py --resume --epochs 16
```

`--epochs` di README ini berarti jumlah target epoch keseluruhan. Resume hanya berfungsi jika `config.json` dan `model.pt` ada pada folder checkpoint. Jika state optimizer tidak ada atau gagal dimuat, proses tetap dapat melanjutkan bobot dengan optimizer yang dimulai ulang.

Untuk membuat model baru pada folder kosong:

```powershell
.venv\Scripts\python.exe train.py --fresh --model-dir output\zydra-experiment
```

`--fresh` tidak dapat digabung dengan `--resume`.

## Arsitektur

Model native Zydra menggunakan Transformer decoder dengan:

- RMSNorm
- Rotary Positional Embedding (RoPE)
- Grouped-Query Attention (GQA)
- SwiGLU
- Causal self-attention

Training memakai cross entropy untuk memprediksi token jawaban berikutnya. Label token pertanyaan tidak digunakan sebagai target loss.

## Chat Terminal

Jalankan chat setelah checkpoint tersedia:

```powershell
.venv\Scripts\python.exe chat.py
```

Atau pilih checkpoint lain:

```powershell
.venv\Scripts\python.exe chat.py --model output\zydra-v2-full
```

Opsi generasi:

```powershell
.venv\Scripts\python.exe chat.py --max-new-tokens 192 --temperature 0.7 --top-p 0.95
```

Perintah chat:

- `keluar`, `/exit`, atau `/quit` untuk berhenti
- `/reset` untuk menghapus riwayat terminal

Chat menyalurkan percakapan sebagai teks dan tidak memiliki pencarian web atau retrieval datasheet. Jika `temperature` bernilai `0`, generasi bersifat greedy; nilai di atas `0` mengaktifkan sampling.

## Ekspor GGUF

Ekspor GGUF saat ini **belum didukung**. `train.py --export-gguf` sengaja berhenti dengan pesan bahwa arsitektur native Zydra memerlukan converter khusus untuk `llama.cpp`.

## Evaluasi

Training mencetak train loss, validation loss, dan `gap = validation loss - train loss` setiap epoch. Gunakan dataset yang tidak tercampur dengan train untuk evaluasi kemampuan yang lebih bermakna.

Loss yang rendah tidak otomatis membuktikan kualitas jawaban. Periksa contoh generasi dan evaluasi secara terpisah menggunakan prompt yang belum pernah dilatih.
