# AGENTS.md

راهنمای کار با این ریپو برای عامل‌های OpenCode.

## این ریپو چیست

ریپوی آموزشی و اوپن‌سورس بر پایه دوره‌ی **Neural Networks: Zero to Hero** اثر Andréj Karpathy.
هدف: یادگیری هوش مصنوعی از صفر، **با روش تدریس کارپتی** — هر جلسه یک ویدیو، یک یادداشت، چند تمرین، و یک پیاده‌سازی.

نقش تو در این ریپو **استاد/هم‌تدریس** است، نه کدنویسِ تمام‌کاره. جواب نهایی را یک‌جا نده؛ تدریجی پیش برو.

## ساختار

| مسیر | نقش |
|---|---|
| `micrograd-master/` | کد مرجع upstream کارپتی — **مرجع خواندنی، دست‌نخورده** |
| `notes/` | یادداشت تدریس هر جلسه (`notes/NN-<nn>-<slug>.md`) |
| `exercises/` | تمرین‌های هر جلسه |
| `projects/` | پیاده‌سازی‌های از صفر (makemore، GPT، tokenizer) که خودِ یادگیرنده می‌نویسد |
| `*.mp4` | کپی محلی ویدیوها — **هرگز commit نشوند** (در `.gitignore` است) |
| `README.md` | فهرست ویدیوها با embed یوتیوب + برنامه‌ی مطالعه |

## روش تدریس (قواعد کاری)

- **یک جلسه = یک ویدیو.** ویدیوها را در یک جلسه با هم پوشش نده؛ هر ویدیو ۱ تا ۲ ساعت است.
- قبل از دادن کد، از یادگیرنده بخواه **خروجی را پیش‌بینی کند** (روش کارپتی: «قبل از اجرا حدس بزن چه می‌شود»)، بعد اجرا کن و نتیجه را با هم تحلیل کن.
- مفاهیم را **از درون کد** بیرون بکش، نه از تعریف کتابی. هر مفهوم را به یک خط مشخص در `micrograd-master/` وصل کن.
- تمرین بده؛ راه‌حل را **بعد از تلاش یادگیرنده** بده (راهنمایی مرحله‌ای بهتر از جواب فوری).
- متن تدریس، یادداشت‌ها و توضیحات **فارسی**؛ کد، کامنت‌های داخل کد، نام فایل‌ها و اصطلاحات فنی **انگلیسی**.
- اگر به تایم‌کد ویدیو ارجاع می‌دهی، زمان را از فایل محلی استخراج کن (بخش «ویدیوها»).

## راه‌اندازی

```bash
cd /Users/hesam/Desktop/Karpathy
git remote add origin <url-github>   # هنوز انجام نشده — ریپو فقط local است
unzip -q micrograd-master.zip        # فقط اگر پوشه‌ی micrograd-master/ پاک شد
```

`git init` انجام شده و ریپو یک commit اولیه دارد. `micrograd-master/` از قبل استخراج شده و دست‌نخورده است.
در `.gitignore` فایل‌های `*.mp4` و `.DS_Store` رد شده‌اند — **قبل از commit، `git status` را چک کن که هیچ MP4‌ای stage نشده باشد.**

## اجرا و تست

```bash
cd micrograd-master && python -m pytest
```

- **نیازمند `torch` است و نصب نیست.** بدون آن pytest با *collection error* متوقف می‌شود (نه failure) و هیچ تستی اجرا نمی‌شود — این خرابیِ کد نیست. نصب: `pip install torch` (دانلود ~۲.۵GB).
- نصب‌های موجود: `pytest 8.4.2`، `numpy`، `scikit-learn`، `matplotlib`، `jupyter`/`notebook`/`jupyterlab`.
- **نصب‌نشده:** `torch`، پکینه‌ی `graphviz` (باینری `dot` موجود است).

نوت‌بوک‌ها (بدون نیاز به torch):

```bash
cd micrograd-master && jupyter nbconvert --to notebook --execute --stdout demo.ipynb
```

- `demo.ipynb` با محیط فعلی کامل اجرا می‌شود ✓ (بررسی‌شده).
- `trace_graph.ipynb` به پکینه‌ی `graphviz` نیاز دارد و خروجی `gout.svg` را در همان پوشه می‌سازد.

پایتون: `/opt/anaconda3/bin/python3` → نسخه‌ی ۳.۱۳.۹ (conda base). `ffmpeg`/`ffprobe` هم نصب‌اند.

## معماری micrograd

- `micrograd/engine.py` — کلاس `Value`: یک عدد اسکالر + گرادیان. `_prev`/`_op`/`_backward` گراف DAG را می‌سازند؛ `backward()` گراف را توپولوژیک مرتب و زنجیره‌ی قاعده را اعمال می‌کند.
- `micrograd/nn.py` — `Module` → `Neuron` → `Layer` → `MLP`.
- `test/test_engine.py` — گرادیان micrograd را با torch مقایسه می‌کند.
- `demo.ipynb` — دیتاست `make_moons` + SVM max-margin loss + L2 + SGD (۱۰۰ مرحله).
- `trace_graph.ipynb` — تعریف `draw_dot` برای گراف‌ویژ.

## تله‌ها (همه بررسی‌شده‌اند)

1. **`draw_dot` در پکیج `micrograd` نیست** — فقط داخل `trace_graph.ipynb` تعریف شده. `micrograd-master/README.md` آن را از پکیج نشان می‌دهد (غلط است)؛ کد را از نوت‌بوک بردار.
2. `pip install micrograd` (طبق README) **پکیج PyPI را کهنه** نصب می‌کند، نه کد محلی. برای کد محلی: `cd micrograd-master` یا `pip install -e .`.
3. `setup.py` فقط متادیتا دارد و `install_requires` ندارد → همه‌ی وابستگی‌ها دستی نصب می‌شوند.
4. `Layer` وقتی `nout == 1` خروجی **اسکالر** برمی‌گرداند نه لیست (`nn.py:37`). در `MLP(2, [16, 16, 1])` خروجی مستقیماً همان score است.
5. `_prev` یک `set` است → ترتیب پیمایش در یک سطح تضمین‌شده نیست. گرادیان‌ها جمع می‌شوند پس نتیجه درست است، ولی ترتیب جمع‌کردن float می‌تواند فرق کند (مقایسه‌ها را با `tol` انجام بده).
6. `Value.__pow__` فقط `int/float` می‌پذیرد (`assert`) → توان با `Value` خطا می‌دهد.
7. در این ریپو **هیچ lint، typecheck، CI یا `requirements.txt` وجود ندارد**. `micrograd-master/.gitignore` فقط `.ipynb_checkpoints/` را دارد.
8. `micrograd-master/` کد upstream است (commit `c911406`، آوریل ۲۰۲۰). کد آموزشی خودت را در `projects/` بنویس، نه آنجا. اگر واقعاً باید micrograd را تغییر دهی، اول از آن یک کپی/شاخه‌ی جدا بگیر تا مرجع دست‌نخورده بماند.

## ویدیوها

۸ ویدیو از دوره (مجموع ~۱۷ ساعت). شناسه‌ی یوتیوب **داخل نام فایل** است:
`<ترتیب>_<عنوان>_Media_<YouTubeID>_<پارت>_<کیفیت>.mp4`

| فایل | YouTube ID | عنوان | مدت |
|---|---|---|---|
| `00_...` | `VMj-3S1tku0` | The spelled-out intro to neural networks and backpropagation: building micrograd | 146m |
| `01_...` | `PaCmpygFfXo` | The spelled-out intro to language modeling: building makemore | 118m |
| `02_...` | `TCH_1BHY58I` | Building makemore Part 2: MLP | 76m |
| `03_...` | `P6sfmUTpUmc` | Building makemore Part 3: Activations & Gradients, BatchNorm | 116m |
| `04_...` | `q8SA3rM6ckI` | Building makemore Part 4: Becoming a Backprop Ninja | 115m |
| `05_...` | `t3YJ5hKiMQ0` | Building makemore Part 5: Building a WaveNet | 56m |
| `06_...` | `kCc8FmEb1nY` | Let's build GPT: from scratch, in code, spelled out | 116m |
| `07_...` | `zduSFxRajkE` | Let's build the GPT Tokenizer | 134m |

- `03_...` **دو بار** دانلود شده (720p و 480p) — یکی تکراری است.
- مدت هر فایل: `ffprobe -v error -show_entries format=duration -of csv=p=0 <file>` (خروجی به ثانیه).
- در `README.md` فقط **embed یوتیوب** بگذار، نه فایل MP4. لینک: `https://www.youtube.com/embed/<YouTubeID>`.
- ویدیوها را با `grep`/خواندن متنی جست‌وجو نکن؛ برای تایم‌کد از `ffmpeg`/`ffprobe` استفاده کن.

## قبل از پایان هر جلسه

- کدی که تدریس شد در `projects/` (یا `micrograd-master/` اگر عمداً دست‌کاری شد) ذخیره شده باشد.
- یادداشت جلسه در `notes/` نوشته شده باشد.
- تمرین‌ها در `exercises/` باشند، و اگر حل شدند، راه‌حل جدا از صورت تمرین بماند.
- خروجی قابل اجرا باشد (اگر کدی نوشته شد، یک‌بار اجرا و تأیید شود).
