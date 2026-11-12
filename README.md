# Zydra

Zydra v2 adalah model bahasa kausal kecil yang dimulai dari bobot acak dan dilatih secara lokal pada subset dataset instruksi bahasa Indonesia. Ini proof of concept, bukan model serbaguna atau model penalaran yang mumpuni. Checkpoint saat ini hanya mendukung masukan teks; analisis gambar belum didukung.

## Susun data

`train.py` memakai `info.jsonl`, contoh coding, percakapan, identitas, riset coding, slang, normalisasi, dan sentimen emotikon. Stopword serta `halo.txt` yang berisi petunjuk format tidak dijadikan jawaban dialog. `.venv/bin/python train.py --prepare-data-only` membangun JSONL gabungan dengan format `instruction/output` menggunakan parser sumber yang sama.

## Satu Checkpoint

Model kanonis berada di `output/zydra-v2-full`. Checkpoint coding/eksperimen lama memakai tokenizer berbeda, jadi bobotnya tidak dapat digabung langsung; training sekarang memakai satu checkpoint dan tokenizer yang sama.

Menjalankan kode train.py
```bash
python train.py
```

[coding_multilanguage.jsonl](datasheet/coding_multilanguage.jsonl) berisi 75 contoh orisinal untuk 22 bahasa/format pemrograman. Cakupan ini starter, bukan semua bahasa/framework. Data coding dilatih bersama dataset percakapan dan sumber lainnya:

```bash
.venv/bin/python train.py \
  --resume \
  --epochs 1
```

Jika checkpoint sudah ada, `train.py` otomatis melanjutkannya di folder yang sama; `--resume` membuat maksud itu eksplisit. Jangan berikan folder output baru untuk training lanjutan. Dataset coding yang kecil belum menjadikan model ahli coding; evaluasi tetap diperlukan.

Fine-tuning coding terakhir memakai 72 contoh training dan 3 validasi selama 20 epoch (loss validasi 1,9821). Generasi HTML, Python, JavaScript, dan Rust berhasil pada prompt uji, tetapi ini belum membuktikan kemampuan coding umum.

## Latih Zydra

Training dimulai dari bobot acak. Siapkan environment CPU:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch>=2.2'
.venv/bin/python -m pip install -r requirements.txt
```

Untuk melatih atau melanjutkan model kanonis dengan seluruh data:

```bash
.venv/bin/python train.py \
  --resume \
  --epochs 1
```

Tambahkan `--max-samples 4096` hanya untuk smoke test. Training tanpa `--resume` hanya membuat checkpoint bila belum ada; jika checkpoint ada, ia tetap dideteksi dan dilanjutkan. Evaluasi loss validasi dan uji jawaban setelah setiap sesi.

Semua persiapan data dan training dijalankan melalui `train.py`; gunakan `--prepare-data-only` untuk membangun dataset perantara saja. Arsitektur Zydra memakai blok kausal Llama (RMSNorm, RoPE, grouped-query attention, dan SwiGLU).


## Chat interaktif di terminal

```bash
.venv/bin/python chat.py
```

# Testing Model
```bash
cd output/zydra-v2-full
```

```bash
sed -i 's/"zydra"/"llama"/g; s/"ZydraForCausalLM"/"LlamaForCausalLM"/g' config.json
grep -E "model_type|architectures" config.json
```

```bash
python -c "
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained('.')
m = AutoModelForCausalLM.from_pretrained('.').eval()
i = tok('Halo', return_tensors='pt')
with torch.no_grad(): o = m.generate(**i, max_new_tokens=50, do_sample=False, pad_token_id=tok.eos_token_id)
print(repr(tok.decode(o[0], skip_special_tokens=True)))
"
```

CLI tetap memuat checkpoint berarsitektur Zydra. Secara default, semua datasheet dipakai sebagai konteks retrieval: kecocokan pertanyaan yang persis dijawab langsung, sedangkan pertanyaan baru diberikan kepada generator Zydra bersama bukti terkait. `--use-retrieval` menampilkan hasil retrieval saja tanpa generasi. Karena Zydra kecil dan dilatih dari awal, jawaban generatif masih dapat keliru atau tidak koheren. Gunakan `keluar` untuk berhenti atau `/reset` untuk menghapus riwayat.

Untuk informasi terkini, CLI mencoba pencarian web otomatis pada pertanyaan bertanda waktu seperti “hari ini/terbaru”; pencarian eksplisit dapat memakai `/web <pertanyaan>`. Hasil web menjadi konteks bagi Zydra dan URL sumber ditampilkan. Pertanyaan dikirim ke layanan pencarian pihak ketiga. Gunakan `--no-web` untuk mematikan akses jaringan.

Soal aritmetika dan persamaan linear dasar dihitung dengan solver matematika terpisah, bukan ditebak oleh model. Kalkulus, pembuktian, dan soal cerita yang belum dapat diubah menjadi ekspresi didukung belum tersedia; solver akan menyatakan batas tersebut.

## Ekspor ke GGUF

Pastikan folder `llama.cpp/` tersedia di root proyek. Ekspor checkpoint Hugging Face Zydra v2 ke GGUF F16 dengan:

```bash
.venv/bin/python train.py --export-gguf \
  --model output/zydra-v2-full \
  --llama-cpp llama.cpp \
  --outfile model/zydra-v2-f16.gguf \
  --outtype f16
```

Mode ekspor di `train.py` menyiapkan salinan sementara dengan metadata arsitektur Llama dan mendaftarkan tokenizer ByteLevel Zydra. File checkpoint dan sumber `llama.cpp` asli tidak diubah.

Untuk mengecilkan file setelah ekspor, gunakan executable `llama-quantize` dari build lokal:

```bash
llama.cpp/build/bin/llama-quantize \
  model/zydra-v2-f16.gguf \
  model/zydra-v2-Q4_K_M.gguf \
  Q4_K_M
```

Uji GGUF menggunakan `llama-cli`:

```bash
llama.cpp/build/bin/llama-cli \
  -m model/zydra-v2-f16.gguf \
  -p "halo" \
  -n 64
```

GGUF hasil ekspor memakai arsitektur Llama yang kompatibel, bukan metadata runtime `zydra`; implementasi blok Zydra saat ini mewarisi Llama. Ekspor GGUF tidak meningkatkan kemampuan model, dan kualitas hasil tetap bergantung pada data serta pelatihannya.

Model dilatih dengan objective causal language modeling: memprediksi token berikutnya dari teks sebelumnya. Ini memungkinkan model meneruskan teks, tetapi bukan berarti model memiliki pemahaman atau kemampuan berpikir seperti manusia dan tidak menjamin jawabannya benar. Untuk mengajarkan pola jawaban, sertakan contoh pertanyaan, penjelasan yang benar dan jawaban akhir di korpus. Nilai ketepatan menggunakan data evaluasi terpisah. Kualitas generasi sangat bergantung pada data; korpus bawaan yang sintetis dan kecil hanya untuk uji alur.
