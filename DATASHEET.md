# Datasheet Model Zydra

## Identitas

- Nama model: Zydra
- Keluaran checkpoint kanonis: `output/zydra-v2-full/`
- Keluaran GGUF: belum dibuat; converter/runtime `llama.cpp` belum mendukung `general.architecture=zydra`
- Inisialisasi: bobot acak; tidak memakai bobot model pralatih
- Arsitektur: decoder-only dengan config `zydra`; implementasi blok menggunakan Llama (`RMSNorm`, RoPE, GQA, SwiGLU)
- Tokenizer: Byte-level BPE checkpoint kanonis; dipertahankan tetap sama saat resume
- Jumlah parameter: 4.000.000

## Konfigurasi awal

| Parameter | Bawaan | Keterangan |
| --- | ---: | --- |
| Kosakata | 2.048 | Kosakata BPE hasil pelatihan |
| Panjang konteks | 256 token | Panjang maksimum urutan pelatihan |
| Lapisan Transformer | 4 | Blok decoder |
| Ukuran tersembunyi | 256 | Lebar embedding dan model |
| Head attention | 4 | Harus membagi ukuran tersembunyi tanpa sisa |
| Head key/value | 2 | GQA |
| Ukuran intermediate | 704 | Lebar SwiGLU |
| Ukuran batch | 4 | Urutan pelatihan per pembaruan bobot |
| Epoch | 1 | Jumlah pengulangan atas blok pelatihan |
| Learning rate | 0.0003 | Learning rate AdamW |

Model ini hanya proof of concept kecil yang dilatih dari bobot acak, bukan model serbaguna. Generasi setelah training masih tidak koheren dan mengulang kata. Pelatihan dari awal memerlukan korpus jauh lebih besar, evaluasi yang disiplin, dan komputasi yang memadai untuk menghasilkan kemampuan bahasa yang berguna.

## Data dan penggunaan

- Bahasa yang dituju: Indonesia, dialog dasar, contoh coding lintas 22 bahasa/format, dan matematika melalui solver
- Pengguna dan tugas yang dituju: belum layak untuk penggunaan nyata
- Penggunaan yang tidak didukung: isi
- Sumber data pelatihan dan lisensi: catat setiap sumber di `data/README.md`
- Batasan dan hasil evaluasi: generasi bahasa umum tetap terbatas; uji kode sederhana HTML/Python/JavaScript/Rust berhasil setelah fine-tuning, tetapi belum cukup untuk penggunaan coding nyata

## Catatan pelatihan

- Versi / tanggal dataset: `datasheet/info.jsonl`, `coding_multilanguage.jsonl`, `percakapan.txt`, `identitas.txt`, `riset.txt`, `slangword.txt`, `normalization.txt`, `emoticon-sentiment.txt`
- Ukuran dataset: 66.262 pasangan sebelum split; dialog, identitas, riset, dan coding diberi pengulangan seimbang setelah split
- Perangkat keras dan durasi pelatihan: CPU
- Loss validasi checkpoint awal: 5.2618; fine-tuning coding: 1.9821; satu epoch campuran seluruh datasheet: 2.7536
- Dataset evaluasi dan hasilnya: exact match percakapan “Kau bisa apa saja?” dan permintaan HTML dijawab dari datasheet; generasi bebas tetap terbatas
- Tipe / kuantisasi GGUF: belum dibuat