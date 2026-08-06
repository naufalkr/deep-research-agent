# deep-research-agent — Project Overview

Dokumen desain internal. Berisi konsep, arsitektur, best practice, pilihan model,
tech stack, roadmap, dan jebakan yang perlu dihindari.

---

## 1. Apa yang Dibangun

**Deep research assistant** — kamu kasih satu pertanyaan besar dan terbuka
(misal: *"Bagaimana tren kredit macet perbankan digital Indonesia 3 tahun
terakhir, dan faktor apa yang paling berpengaruh?"*), lalu sistem:

1. Memecah jadi beberapa sub-pertanyaan
2. Menugaskan beberapa agent untuk meneliti sub-pertanyaan itu **secara paralel**
3. Tiap agent mencari dari sumbernya masing-masing (web, atau database)
4. Ada agent yang mengecek ulang hasilnya
5. Digabung jadi satu laporan panjang dengan sitasi yang bisa ditelusuri

### Pembeda

Kebanyakan deep research agent cuma bisa baca web. Ini bisa **web + data
terstruktur** — kalau agent butuh angka konkret, dia query database beneran
(lewat [nlquery-agent](https://github.com/naufalkr/NLQuery-agent) sebagai MCP
tool), bukan comot angka dari artikel yang belum tentu akurat.

Ini sekaligus jembatan alami ke proyek berikutnya (agent orchestration
platform): manggil agent lain sebagai tool adalah inti dari orkestrasi.

---

## 2. Arsitektur: Pola Orchestrator-Worker

Pola yang terbukti di produksi. Anthropic memakai persis ini untuk fitur
Research mereka: satu **lead agent** merencanakan, lalu men-spawn 3–5
**subagent** yang jalan paralel, dan hasilnya disintesis dengan **citation pass
terpisah**. Hasilnya mengungguli sistem single-agent sebesar **90,2%** pada
evaluasi riset internal mereka — dengan biaya sekitar **15× token** dibanding
percakapan chat biasa.

```
                    ┌─────────────────┐
   Pertanyaan ─────▶│  LEAD AGENT     │  (model kuat)
                    │  - rencana      │
                    │  - delegasi     │
                    │  - sintesis     │
                    └────────┬────────┘
                             │ spawn paralel
          ┌──────────────────┼──────────────────┐
          ▼                  ▼                  ▼
    ┌───────────┐      ┌───────────┐      ┌───────────┐
    │ SUBAGENT  │      │ SUBAGENT  │      │ SUBAGENT  │  (model murah)
    │  (web)    │      │  (web)    │      │   (DB)    │
    │ context   │      │ context   │      │ context   │
    │ sendiri   │      │ sendiri   │      │ sendiri   │
    └─────┬─────┘      └─────┬─────┘      └─────┬─────┘
          └──────────────────┼──────────────────┘
                             ▼
                    ┌─────────────────┐
                    │  CRITIC AGENT   │  cek: ada klaim tanpa
                    │  (verifikasi)   │  bukti? kontradiksi?
                    └────────┬────────┘
                             ▼
                    ┌─────────────────┐
                    │ CITATION AGENT  │  pasang sitasi ke tiap klaim
                    └────────┬────────┘
                             ▼
                        Laporan akhir
```

### Peran tiap agent

| Agent | Tugas | Kenapa terpisah |
|---|---|---|
| **Lead / Orchestrator** | Baca pertanyaan → susun rencana riset → tentukan berapa subagent & apa tugas masing-masing → gabung hasil jadi narasi | Perencanaan dan eksekusi dipisah supaya lebih terkontrol dan blast radius kegagalan lebih kecil |
| **Web Researcher** (bisa >1) | Cari & baca sumber web, ekstrak fakta + URL sumbernya | Tiap punya context window sendiri, jadi bisa gali dalam tanpa mengotori context agent lain |
| **Data Researcher** | Query database lewat nlquery-agent, ambil angka aktual | Pembeda utama — sumber yang bisa diverifikasi, bukan klaim artikel |
| **Critic / Verifier** | Cek tiap klaim: ada buktinya? ada kontradiksi antar-sumber? | Self-critique dari agent yang sama itu lemah — verifier dengan context bersih jauh lebih jujur |
| **Citation agent** | Tempelkan sitasi ke tiap klaim di laporan akhir | Dijalankan sebagai pass terpisah, bukan digabung ke sintesis — supaya sitasi tidak dikarang |

**Kenapa context terpisah itu kunci:** tiap subagent punya context window, tools,
dan jalur eksplorasi sendiri. Kalau semua riset dijejalkan ke satu context,
agent akan tenggelam — kualitas turun drastis di percakapan panjang.

---

## 3. Alur End-to-End

```
1. INTAKE
   User: "Bagaimana tren X dan apa penyebabnya?"
   ↓
2. PLANNING (lead agent)
   → Klasifikasi kompleksitas: simple / moderate / complex
   → Pecah jadi sub-pertanyaan:
      a. "Apa data historis X?"           → butuh DATABASE
      b. "Apa kata analis tentang X?"     → butuh WEB
      c. "Faktor makro apa yang terkait?" → butuh WEB
   → Tentukan jumlah subagent (2 untuk simple, 5+ untuk complex)
   ↓
3. DELEGATION
   Tiap subagent dapat: objective jelas + format output + tools yang boleh
   dipakai + batasan tugas
   ↓
4. PARALLEL EXECUTION
   Subagent jalan bersamaan (asyncio.gather / LangGraph parallel node)
   Tiap subagent: search → baca → ekstrak → catat sumber → return findings
   terstruktur
   ↓
5. GATHERING
   Lead mengumpulkan semua findings
   → Cukup? Lanjut. Ada gap? Spawn subagent ronde kedua.
   ↓
6. VERIFICATION (critic)
   Cek: klaim tanpa sumber, angka kontradiktif antar-sumber, over-generalisasi
   → Kalau ada masalah serius: balik ke step 4 untuk sub-pertanyaan tertentu
   ↓
7. SYNTHESIS
   Lead menyusun laporan naratif dari findings yang sudah terverifikasi
   ↓
8. CITATION PASS
   Agent terpisah memasang sitasi [1][2] ke tiap klaim
   ↓
9. OUTPUT
   Laporan + daftar sumber + trace (bisa dilihat prosesnya)
```

---

## 4. Best Practices

### 4.1 Prompt delegasi adalah pekerjaan utama, bukan sampingan

Pelajaran paling mahal dari Anthropic: mereka menghabiskan **berminggu-minggu**
hanya untuk menonton agent gagal di simulasi lalu menulis ulang prompt delegasi
untuk memperbaiki failure mode spesifik.

Tiap subagent **wajib** menerima 4 hal secara eksplisit:

1. **Objective** — apa persisnya yang harus dicari. Bukan "riset tentang X",
   tapi "temukan data NPL bank digital Indonesia 2023–2025 dari sumber resmi"
2. **Output format** — bentuk hasil yang diharapkan (JSON dengan field `claim`,
   `evidence`, `source_url`, `confidence`)
3. **Tool guidance** — tools mana yang dipakai dan kapan
4. **Task boundaries** — apa yang **bukan** tugasnya, biar tidak overlap dengan
   subagent lain

Prinsip yang dipublikasikan Anthropic: *"think like your agents"*, *"scale effort
to query complexity"*, dan *"teach the orchestrator how to delegate"*.

### 4.2 Skalakan usaha ke kompleksitas pertanyaan

Jangan spawn 5 subagent untuk pertanyaan sederhana. Lead agent harus
mengklasifikasi dulu:

| Kompleksitas | Jumlah subagent | Contoh |
|---|---|---|
| Simple | 1 (atau jawab langsung) | "Berapa NPL Bank X tahun 2024?" → cukup satu query DB |
| Moderate | 2–3 | "Bandingkan NPL 3 bank digital" |
| Complex | 4–6 | "Analisis tren + penyebab + proyeksi" |

Tanpa ini, biaya meledak untuk pertanyaan yang sebenarnya sepele. Ingat:
multi-agent itu ~15× token dibanding chat biasa.

### 4.3 Routing tool: kapan web, kapan database

Ini pembeda proyek, jadi harus dirancang eksplisit. Lead agent perlu aturan jelas:

```
Butuh angka spesifik/agregat dari data internal   → tool: query_database
Butuh konteks, opini, kejadian terbaru, definisi  → tool: web_search
Butuh keduanya (klaim web perlu divalidasi angka) → keduanya, lalu cross-check
```

**Pola paling kuat:** subagent web menemukan klaim → critic memicu subagent DB
untuk memverifikasi angkanya. Ini yang tidak bisa dilakukan deep research agent
biasa, dan persis kebutuhan di lingkungan finansial/perbankan.

### 4.4 Manajemen context

- Tiap subagent punya context sendiri — jangan share
- Subagent mengembalikan **findings terstruktur** (JSON ringkas), bukan seluruh
  isi halaman web yang dia baca
- Untuk sesi panjang: pakai summarization/compaction supaya context tidak
  membengkak
- Lead agent hanya melihat ringkasan, tidak melihat raw scraping

Kalau tidak begini, satu subagent yang membaca 20 halaman akan menghabiskan
context lead agent sendirian.

### 4.5 Model tiering — pengendali biaya utama

Jangan pakai model termahal untuk semua. Pola yang terbukti: **lead agent pakai
model kuat, subagent pakai model lebih murah.**

Detail pilihan model ada di bagian 5.

### 4.6 Sitasi sebagai pass terpisah — jangan digabung

Kalau satu agent diminta "tulis laporan sekaligus pasang sitasi", dia akan
mengarang sitasi. Pisahkan:

1. Sintesis menghasilkan narasi (tanpa sitasi)
2. Citation agent menerima narasi + daftar findings dengan sumbernya →
   memasangkan

Ini yang bikin hasilnya bisa dipercaya, dan sejalan dengan filosofi dua proyek
sebelumnya (SQL selalu ditampilkan, kode selalu di-cite `file:line`).

### 4.7 Sandboxing untuk executor

Agent executor harus jalan di lingkungan tersandbox. Untuk kasus ini penting
karena DB subagent mengeksekusi query — untungnya nlquery-agent **sudah** punya
guard read-only dua lapis. Pertahankan itu, dan tambahkan: rate limit untuk web
fetching, timeout per subagent, dan batas maksimal ronde riset.

### 4.8 Evaluation harness — yang membedakan "demo" dari "sistem"

Bagian yang paling sering dilewati orang, dan justru yang bikin proyek naik
kelas. Bangun:

- **Set pertanyaan uji** (20–30 pertanyaan dengan jawaban yang sudah diketahui)
- **LLM-as-judge** untuk menilai: akurasi faktual, kelengkapan, kualitas sitasi
- **Metrik biaya**: token & USD per pertanyaan
- **Metrik latensi**: berapa lama per pertanyaan

Tanpa ini tidak bisa tahu apakah perubahan prompt membuat sistem lebih baik atau
lebih buruk. Dengan ini, ada bahan konkret untuk portfolio: *"meningkatkan
akurasi dari X% ke Y% dengan menambahkan verifier agent"* — jauh lebih kuat
daripada "saya bikin research agent".

### 4.9 Observability & tracing

Extend `observability.py` dari nlquery-agent jadi tracing multi-agent:

- Tiap subagent: berapa token, berapa lama, berapa tool call, sumber apa yang
  dibaca
- Visualisasi pohon: pertanyaan → sub-pertanyaan → findings → laporan
- Total biaya per sesi riset

Dashboard ini sekaligus jadi showcase front-end.

### 4.10 Error handling: kesalahan agent itu berlipat

Beda dengan pipeline biasa, di sistem agentic satu kesalahan di langkah awal
merambat ke semua langkah berikutnya. Yang perlu:

- Subagent gagal → jangan gagalkan seluruh riset; lead lanjut dengan findings
  yang ada + catat gap-nya
- Retry dengan backoff (sudah ada di `llm.py` nlquery-agent)
- Batas maksimal ronde riset (misal 3) supaya tidak infinite loop
- Kalau critic terus menolak → berhenti dan laporkan ketidakpastiannya, jangan
  memaksakan jawaban

---

## 5. Pilihan Model

### Catatan penting: benchmark coding ≠ benchmark research

Model open-source yang sering disebut untuk agentic (Kimi K3, Qwen3-Coder,
DeepSeek V4, GLM-5.2) diranking berdasarkan **SWE-bench dan LiveBench Agentic
Coding** — kemampuan menulis dan memperbaiki kode. Deep research agent butuh
sumbu kemampuan yang **berbeda**:

| Coding agent butuh | Research agent butuh |
|---|---|
| Menulis kode yang benar | **Tool-calling yang reliable** (panggil search/DB berkali-kali tanpa salah format) |
| Memahami repo besar | **Sintesis long-context** (baca 20 sumber → satu narasi koheren) |
| Debugging & iterasi | **Instruction following** (patuh pada batasan tugas delegasi) |

Benchmark yang relevan: **BFCL v3** (Berkeley Function Calling Leaderboard),
**τ-bench**, dan **MCP-Bench** — yang terakhir paling dekat dengan workload
produksi karena menguji koordinasi multi-hop lewat 250 tools di 28 MCP server
nyata.

### Posisi open-weight (data 2026)

Di BFCL v3: **GLM-4.5 memimpin di 76,7%**, disusul **Qwen3 32B di 75,7%** —
dua-duanya open-weight.

Temuan penting: **long-context retrieval & sintesis** sekarang dimenangkan model
open dengan selisih besar dari sisi price-to-performance. **Tapi** model
proprietary frontier masih unggul di *long-horizon agentic workflow*,
*reliabilitas tool use dalam skala besar*, dan *judgment pada kasus edge* — dan
itu persis pekerjaan orchestrator dan critic.

### Rekomendasi per peran — hybrid, bukan salah satu

| Agent | Model | Alasan |
|---|---|---|
| **Lead / Orchestrator** | Proprietary frontier (`claude-opus-5` atau `claude-sonnet-5`) | Perencanaan, delegasi, sintesis akhir = long-horizon agentic. ~10% dari total call tapi 80% penentu kualitas |
| **Critic / Verifier** | Sama dengan lead | Verifikasi butuh judgment; salah di sini = laporan salah tapi terlihat meyakinkan |
| **Web subagent** | Open-weight (GLM-4.5 / Qwen3 32B via provider) | Ekstraksi + ringkasan = tugas yang open model sudah sangat kompetitif, dan volumenya paling banyak |
| **DB subagent** | nlquery-agent apa adanya | Sudah punya tiering fast/smart sendiri; Groq llama sudah cukup untuk SQL |
| **Citation agent** | Model termurah (`claude-haiku-4-5` atau open kecil) | Tugas mekanis: cocokkan klaim ke sumber |

### Lokal vs hosted — reality check

Model-model besar itu **tidak realistis dijalankan di laptop**. GLM-5.2 itu 753B
parameter (40B aktif per token), Qwen3-Coder 480B. Bahkan Hermes 4 35B-A3B yang
paling ramah butuh RTX 4090 dengan ~22GB VRAM.

Yang realistis:

- **Hosted open-weight** — pakai open model lewat provider murah: **Groq**,
  **OpenRouter**, **Together**, atau **DeepInfra**. Dapat model open-source tanpa
  punya GPU. Ini jalur yang benar.
- **Lokal** — hanya masuk akal untuk model kecil (7B–14B) via Ollama, dan itu
  terlalu lemah untuk orchestrator. Bisa dipakai sebagai *mode demo offline*
  saja, seperti mode `mock` di nlquery-agent.

### Konfigurasi per-peran

`llm.py` dari nlquery-agent sudah provider-agnostic (`groq` / `anthropic` /
`mock`). Yang perlu ditambah:

1. Provider baru (OpenRouter — API-nya OpenAI-compatible, gampang)
2. Konfigurasi model **per-peran**, bukan cuma fast/smart global:
   `LEAD_MODEL`, `SUBAGENT_MODEL`, `CRITIC_MODEL`, `CITATION_MODEL` di `.env`

Dengan begitu bisa A/B test: jalankan set pertanyaan uji dengan konfigurasi
all-Claude vs hybrid vs all-open, lalu bandingkan akurasi dan biayanya di
evaluation harness. **Itu sendiri jadi bahan portfolio yang kuat** — bukan cuma
bikin agent, tapi punya data soal trade-off model.

---

## 6. Tech Stack

| Komponen | Pilihan | Alasan |
|---|---|---|
| **Orkestrasi** | LangGraph | Kontrol eksplisit atas state antar-agent, mendukung node paralel, ada checkpointing. Lebih cocok daripada CrewAI untuk sistem yang butuh kontrol presisi |
| **LLM client** | Angkat `llm.py` dari nlquery-agent | Sudah provider-agnostic, sudah ada retry & observability |
| **Web search** | Tavily / Brave Search API | Tavily didesain untuk agent; hasilnya sudah bersih |
| **Web fetch** | Jina Reader / trafilatura | Konversi HTML → teks bersih |
| **Database** | nlquery-agent, dipanggil sebagai MCP tool | Reuse proyek lama = pembeda utama |
| **Memory** | Mem0 (opsional, fase lanjut) | Agent ingat preferensi riset user lintas sesi |
| **Backend** | FastAPI + SSE streaming | Riset butuh waktu lama — user harus lihat progres real-time |
| **Frontend** | React + Vite + Tailwind | Konsisten dengan dua proyek sebelumnya |
| **Storage** | SQLite untuk sesi riset & trace | Bisa buka lagi riset lama |

---

## 7. Roadmap Bertahap

Bangun berlapis, tiap fase harus jalan sendiri sebelum lanjut.

| Fase | Tag | Isi |
|---|---|---|
| **1. Baseline** | `v0.1-baseline` | Satu agent, satu loop, 2 tools: `web_search` + `query_database`. Belum multi-agent. Jadi **pembanding** untuk membuktikan multi-agent memang lebih baik |
| **2. Multi-agent** | `v0.2-multiagent` | Lead agent yang memecah pertanyaan dan spawn subagent paralel. Lompatan arsitektur terbesar |
| **3. Verifikasi** | `v0.3-verified` | Critic + citation pass terpisah. Kualitas output naik drastis |
| **4. Evaluasi** | `v0.4-evaluated` | Set pertanyaan uji + LLM judge + metrik. Sekarang bisa diukur, bukan ditebak |
| **5. Web UI** | `v0.5-ui` | Streaming + tracing dashboard, visualisasi proses riset real-time |
| **6. Polish** | `v1.0` | Persistent memory, ekspor laporan, Docker, dokumentasi |

Fase 1–3 sudah menghasilkan sesuatu yang layak didemokan. Fase 4 yang bikin ini
"serius".

**Kenapa pakai git tag per fase:** siapa pun yang lihat repo bisa membandingkan
`v0.1-baseline` vs `v0.2-multiagent` dan melihat bukti bahwa arsitektur
multi-agent meningkatkan hasil — jauh lebih meyakinkan daripada klaim di README.

---

## 8. Jebakan yang Bikin Proyek Ini Gagal

1. **Langsung bikin multi-agent tanpa baseline** — tidak akan bisa membuktikan
   multi-agent-nya berguna. Selalu mulai dari single agent.
2. **Prompt delegasi asal-asalan** — subagent akan kerja tumpang tindih atau
   meleset dari tugasnya. Penyebab kegagalan #1.
3. **Tidak memisahkan citation pass** — sitasi akan dikarang, dan itu fatal untuk
   research tool.
4. **Tidak ada batas biaya** — 15× token itu nyata; satu pertanyaan kompleks bisa
   menghabiskan biaya besar tanpa disadari. Pasang budget cap dari awal.
5. **Tidak ada evaluation** — akan iterasi berdasarkan perasaan, bukan data.
6. **Menganggap semua pertanyaan sama kompleksnya** — spawn 5 agent untuk
   pertanyaan sepele = pemborosan.
7. **Self-critique dengan agent yang sama** — verifier harus punya context bersih,
   kalau tidak dia akan membenarkan dirinya sendiri.
8. **Menyimpan seluruh raw content di context lead** — context meledak, kualitas
   anjlok.
9. **Bangun dulu, commit belakangan** — pelajaran dari nlquery-agent yang dibangun
   10 jam penuh lalu tidak pernah di-commit. Commit tiap modul selesai.

---

## 9. Referensi

- [How Anthropic Built a Multi-Agent Research System — ByteByteGo](https://blog.bytebytego.com/p/how-anthropic-built-a-multi-agent)
- [Anthropic's Multi-Agent Research Architecture Explained — The AI Engineer](https://theaiengineer.substack.com/p/how-anthropic-built-multi-agent-deep)
- [Building a Multi-Agent Research System for Complex Information Tasks — ZenML LLMOps Database](https://www.zenml.io/llmops-database/building-a-multi-agent-research-system-for-complex-information-tasks)
- [Planner–Executor–Critic Loops in Deep Agents](https://medium.com/@thisissiddharthhudda/planner-executor-critic-loops-in-deep-agents-4c4774d88632)
- [Deep Research Agent: Enterprise Guide & Architecture 2026 — Ampcome](https://www.ampcome.com/post/deep-research-agent)
- [Multi-Agent System Architecture Guide for 2026 — ClickIT](https://www.clickittech.com/ai/multi-agent-system-architecture/)
- [Deep Research: A Systematic Survey (arXiv)](https://arxiv.org/pdf/2512.02038)
- [Function Calling Benchmarks Leaderboard 2026 — Awesome Agents](https://awesomeagents.ai/leaderboards/function-calling-benchmarks-leaderboard/)
- [MCP-Bench: Benchmarking Tool-Using LLM Agents (arXiv)](https://arxiv.org/pdf/2508.20453)
- [The State of Open-Weight Models in 2026 — jamesm.blog](https://www.jamesm.blog/ai/state-of-open-weight-models-2026/)
