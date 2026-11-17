# Zydra v2

Zydra v2 adalah model bahasa kausal berukuran kecil yang dilatih secara lokal, dimulai dari bobot acak menggunakan subset dataset instruksi berbahasa Indonesia.

Proyek ini merupakan *proof of concept* (PoC), bukan model serbaguna atau model penalaran tingkat lanjut. Checkpoint saat ini hanya mendukung masukan teks; analisis gambar belum tersedia.

## Susunan Data

`train.py` mengolah beberapa sumber data, antara lain:

- `info.jsonl`
- Contoh pemrograman
- Percakapan
- Data identitas
- Riset pemrograman
- Slang
- Data normalisasi
- Sentimen emotikon

Stopword dan `halo.txt` yang berisi petunjuk format tidak digunakan sebagai jawaban dialog.

Untuk membangun dataset gabungan tanpa menjalankan training:

```bash
.venv/bin/python train.py --prepare-data-only
```

Perintah tersebut menghasilkan JSONL gabungan dengan format `instruction/output` menggunakan parser sumber data yang sama dengan proses training.

## Checkpoint Utama

Checkpoint kanonis Zydra v2 berada di:

```text
output/zydra-v2-full
```

Checkpoint coding dan eksperimen lama menggunakan tokenizer yang berbeda. Oleh karena itu, bobot dari checkpoint tersebut tidak dapat langsung digabungkan.

Training saat ini menggunakan satu checkpoint dan tokenizer yang sama agar proses pelatihan dan kelanjutannya tetap konsisten.

## Menjalankan Training

Jalankan training dengan:

```bash
python train.py
```

### Dataset Coding

File [`datasheet/coding_multilanguage.jsonl`](datasheet/coding_multilanguage.jsonl) berisi 75 contoh orisinal yang mencakup 22 bahasa atau format pemrograman.

Cakupan tersebut merupakan data awal (*starter dataset*), bukan representasi seluruh bahasa pemrograman dan framework.

Data coding dilatih bersama dataset percakapan dan sumber data lainnya.

Untuk melanjutkan training dengan dataset gabungan:

```bash
.venv/bin/python train.py \
  --resume \
  --epochs 1
```

Jika checkpoint sudah tersedia, `train.py` akan melanjutkan training dari checkpoint tersebut di folder yang sama. Opsi `--resume` digunakan untuk menyatakan bahwa training memang dimaksudkan untuk dilanjutkan.

Jangan menggunakan folder output baru ketika ingin melanjutkan training checkpoint yang sudah ada.

Fine-tuning coding terakhir menggunakan:
- 72 contoh training
- 3 contoh validasi
- 20 epoch
- Loss validasi sebesar `1.9821`

Generasi kode HTML, Python, JavaScript, dan Rust berhasil dilakukan pada prompt pengujian. Namun, hasil tersebut belum membuktikan bahwa Zydra memiliki kemampuan pemrograman umum atau setara dengan model coding khusus.

## Menyiapkan Environment

Training dapat dijalankan menggunakan CPU.

Buat virtual environment:

```bash
python3 -m venv .venv
```

Aktifkan pip dan perbarui versinya:

```bash
.venv/bin/python -m pip install --upgrade pip
```

Instal PyTorch versi CPU:

```bash
.venv/bin/python -m pip install \
  --index-url https://download.pytorch.org/whl/cpu \
  'torch>=2.2'
```

Instal dependensi proyek:

```bash
.venv/bin/python -m pip install -r requirements.txt
```

### Training atau Melanjutkan Training

Untuk melatih atau melanjutkan model kanonis menggunakan seluruh data:

```bash
.venv/bin/python train.py \
  --resume \
  --epochs 1
```

Opsi `--max-samples 4096` dapat ditambahkan untuk melakukan *smoke test* dengan jumlah sampel terbatas.

Tanpa `--resume`, `train.py` tetap mendeteksi checkpoint yang sudah tersedia dan melanjutkan training sesuai perilaku yang diterapkan dalam skrip.

Setelah setiap sesi training, evaluasi loss validasi dan uji kualitas jawaban model. Jangan hanya mengandalkan loss training untuk menilai kemampuan model.

Seluruh proses persiapan data dan training dijalankan melalui `train.py`. Gunakan `--prepare-data-only` jika hanya ingin membangun dataset perantara tanpa menjalankan training.

## Arsitektur Model

Zydra menggunakan implementasi blok Transformer kausal yang mengadopsi komponen-komponen yang juga digunakan dalam arsitektur Llama, yaitu:

- RMSNorm
- Rotary Positional Embedding (RoPE)
- Grouped-Query Attention (GQA)
- SwiGLU
- Causal self-attention

Model dilatih menggunakan *causal language modeling*, yaitu memprediksi token berikutnya berdasarkan token-token sebelumnya.

Dengan demikian, Zydra dapat mempelajari pola bahasa dan meneruskan teks berdasarkan data training. Namun, hal tersebut tidak menjamin bahwa model memiliki pemahaman atau kemampuan berpikir seperti manusia maupun bahwa semua jawabannya benar.

Arsitektur yang digunakan saat ini merupakan implementasi Zydra dengan komponen Transformer yang sudah dikenal, bukan arsitektur baru yang seluruh mekanismenya ditemukan secara independen.

## Chat Interaktif di Terminal

Jalankan:

```bash
.venv/bin/python chat.py
```

Secara default, CLI menggunakan seluruh datasheet sebagai konteks retrieval.

Mekanisme jawabannya:
1. Pertanyaan yang cocok secara persis dengan data dapat dijawab langsung menggunakan hasil pencocokan.
2. Pertanyaan baru diteruskan ke generator Zydra bersama bukti relevan dari proses retrieval.
3. Opsi `--use-retrieval` menampilkan hasil retrieval saja tanpa menjalankan generasi teks.

Karena Zydra berukuran kecil dan dilatih dari awal, jawaban generatif masih dapat keliru, tidak lengkap, atau tidak koheren.

Perintah CLI:
- `keluar` untuk menghentikan sesi.
- `/reset` untuk menghapus riwayat percakapan.

### Pencarian Web

Untuk pertanyaan yang membutuhkan informasi terkini, CLI dapat mencoba pencarian web otomatis, misalnya ketika pertanyaan mengandung kata seperti "hari ini" atau "terbaru".

Pencarian eksplisit dapat dilakukan menggunakan:

```text
/web <pertanyaan>
```

Hasil pencarian digunakan sebagai konteks tambahan bagi Zydra, dan URL sumber ditampilkan kepada pengguna.

**Catatan privasi:** Pertanyaan yang memicu pencarian dikirim ke layanan pencarian pihak ketiga. Jika tidak ingin menggunakan akses jaringan, jalankan:

```bash
.venv/bin/python chat.py --no-web
```

### Solver Matematika

Soal aritmetika dan persamaan linear dasar dihitung menggunakan solver matematika terpisah, bukan sekadar ditebak oleh model bahasa.

Kemampuan yang belum tersedia meliputi:
- Kalkulus
- Pembuktian matematika
- Soal cerita yang belum dapat diubah menjadi ekspresi matematika

Untuk jenis soal yang belum didukung, solver akan menyatakan keterbatasannya.

## Pengujian Model

Checkpoint utama berada di:

```text
output/zydra-v2-full
```

Untuk pengujian menggunakan `transformers`, konfigurasi model perlu menggunakan metadata arsitektur Llama yang kompatibel dengan pemuatan model tersebut.

**Penting:** Jangan mengubah `config.json` pada checkpoint kanonis hanya untuk pengujian kompatibilitas. Gunakan salinan checkpoint agar metadata asli Zydra tetap terjaga.

Pada salinan checkpoint, penyesuaian metadata dapat dilakukan dengan:

```bash
sed -i \
  's/"zydra"/"llama"/g; s/"ZydraForCausalLM"/"LlamaForCausalLM"/g' \
  config.json
```

Periksa hasilnya:

```bash
grep -E "model_type|architectures" config.json
```

Kemudian uji pemuatan model:

```bash
python -c "
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained('.')
model = AutoModelForCausalLM.from_pretrained('.').eval()

inputs = tok('Halo', return_tensors='pt')

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=False,
        pad_token_id=tok.eos_token_id
    )

print(repr(tok.decode(outputs[0], skip_special_tokens=True)))
"
```

Pengujian ini memeriksa apakah checkpoint dapat dimuat melalui jalur kompatibilitas `transformers`. Keberhasilan pemuatan tidak otomatis membuktikan kualitas generasi atau kompatibilitas seluruh fitur model.

CLI Zydra tetap menggunakan mekanisme pemuatan checkpoint yang diterapkan dalam proyek dan tidak perlu mengubah metadata checkpoint kanonis untuk pengujian di atas.

## Ekspor ke GGUF

Pastikan direktori `llama.cpp/` tersedia di root proyek.

Ekspor checkpoint Hugging Face Zydra v2 ke format GGUF F16 menggunakan:

```bash
.venv/bin/python train.py --export-gguf \
  --model output/zydra-v2-full \
  --llama-cpp llama.cpp \
  --outfile model/zydra-v2-f16.gguf \
  --outtype f16
```

Mode ekspor pada `train.py` menyiapkan salinan sementara dengan metadata arsitektur Llama dan mendaftarkan tokenizer ByteLevel Zydra.

Proses ini dirancang agar file checkpoint asli dan sumber `llama.cpp` tidak diubah.

### Kuantisasi GGUF

Untuk memperkecil ukuran file setelah ekspor, gunakan executable `llama-quantize` dari build lokal:

```bash
llama.cpp/build/bin/llama-quantize \
  model/zydra-v2-f16.gguf \
  model/zydra-v2-Q4_K_M.gguf \
  Q4_K_M
```

### Menguji File GGUF

Gunakan `llama-cli` untuk menjalankan pengujian:

```bash
llama.cpp/build/bin/llama-cli \
  -m model/zydra-v2-f16.gguf \
  -p "halo" \
  -n 64
```

GGUF hasil ekspor menggunakan metadata arsitektur Llama yang kompatibel, bukan metadata runtime `zydra`. Implementasi blok Zydra saat ini menggunakan komponen yang berasal dari keluarga arsitektur Llama.

Ekspor ke GGUF maupun kuantisasi tidak meningkatkan kemampuan model. Kualitas hasil tetap bergantung pada arsitektur, data training, proses pelatihan, dan evaluasi.

## Objective Training dan Keterbatasan

Zydra dilatih menggunakan objective *causal language modeling*: memprediksi token berikutnya berdasarkan teks sebelumnya.

Objective ini memungkinkan model mempelajari pola teks dan menghasilkan kelanjutan berdasarkan konteks. Namun, kemampuan tersebut tidak menjamin pemahaman seperti manusia, penalaran yang benar, atau ketepatan setiap jawaban.

Untuk mengajarkan pola jawaban tertentu, dataset perlu menyertakan contoh yang memuat:
- Pertanyaan atau instruksi
- Penjelasan yang benar jika diperlukan
- Jawaban akhir yang diharapkan

Ketepatan model harus dinilai menggunakan dataset evaluasi yang terpisah dari data training.

Kualitas generasi sangat bergantung pada kualitas, variasi, dan jumlah data. Korpus bawaan yang sintetis dan berukuran kecil hanya ditujukan untuk menguji alur persiapan data, training, dan generasi.

Zydra v2 masih merupakan proyek eksperimental. Hasil generasi perlu diperiksa dan dievaluasi sebelum digunakan untuk kebutuhan yang lebih luas.
