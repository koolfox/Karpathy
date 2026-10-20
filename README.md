# Neural Networks: Zero to Hero — سفر یادگیری

> «What I cannot create, I do not understand.» — Richard Feynman

ریپوی آموزشی و اوپن‌سورس برای یادگیری هوش مصنوعی از صفر، بر پایهٔ دورهٔ
**Neural Networks: Zero to Hero** اثر [Andrej Karpathy](https://github.com/karpathy).

اینجا کد را **می‌خوانیم، می‌فهمیم و از صفر می‌نویسیم** — نه اینکه فقط کپی کنیم.

📌 **همین دوره در وب‌سایت:** [koolfox.github.io/karpathy](https://koolfox.github.io/karpathy/)

---

## ویدیوها

هر ویدیو یک جلسهٔ کامل است. نسخهٔ محلی (MP4) در همین پوشه نگه‌داری می‌شود ولی به‌دلیل حجم بالا (۱.۷ گیگابایت) در ریپو commit نمی‌شود؛
پایین هر ویدیو یک **تصویر بندانگشتی قابل کلیک** هست که ویدیو را در یوتیوب باز می‌کند.

> گیت‌هاب `<iframe>` و `<script>` را حذف می‌کند، به همین دلیل embed در README کار نمی‌کند و از تصویر بندانگشتی استفاده می‌کنیم.

### ۰۱ — مقدمهٔ کامل شبکه‌های عصبی و بک‌پراپ: ساخت micrograd

[![ویدیوی ۰۱ — The spelled-out intro to neural networks and backpropagation: building micrograd](https://img.youtube.com/vi/VMj-3S1tku0/0.jpg)](https://www.youtube.com/watch?v=VMj-3S1tku0)

**۱۴۶ دقیقه** · `VMj-3S1tku0` · فایل محلی: `00_The-spelled-out-intro-to-neural-networks_Media_VMj-3S1tku0_003_480p.mp4`

**موضوع:** از صفر یک موتور autodiff اسکالر-مقدار می‌سازیم: گراف پویا، مرتب‌سازی توپولوژیک، قاعدهٔ زنجیره، و `backward()`.
**خروجی جلسه:** `projects/micrograd/` — پیاده‌سازی خودمان از `Value` و یک MLP دو لایه.

### ۰۲ — مقدمهٔ کامل مدل‌سازی زبان: ساخت makemore

[![ویدیوی ۰۲ — The spelled-out intro to language modeling: building makemore](https://img.youtube.com/vi/PaCmpygFfXo/0.jpg)](https://www.youtube.com/watch?v=PaCmpygFfXo)

**۱۱۸ دقیقه** · `PaCmpygFfXo` · فایل محلی: `01_The-spelled-out-intro-to-language-modeli_Media_PaCmpygFfXo_002_720p.mp4`

**موضوع:** bigram و سپس bigram با کانولوشن چندسطحی، آمار هم‌بستگی، نمونه‌گیری و temperature.
**خروجی جلسه:** `projects/makemore/` — تولیدکنندهٔ متن.

### ۰۳ — makemore بخش ۲: MLP

[![ویدیوی ۰۳ — Building makemore Part 2: MLP](https://img.youtube.com/vi/TCH_1BHY58I/0.jpg)](https://www.youtube.com/watch?v=TCH_1BHY58I)

**۷۶ دقیقه** · `TCH_1BHY58I` · فایل محلی: `02_Building-makemore-Part-2-MLP_Media_TCH_1BHY58I_002_720p.mp4`

**موضوع:** چرا MLP بهتر از bigram است — با همان ابزار micrograd که خودمان ساختیم.
**خروجی جلسه:** آموزش MLP روی دیتاست‌های toy.

### ۰۴ — makemore بخش ۳: اکتیویشن‌ها و جریان گرادیان

[![ویدیوی ۰۴ — Building makemore Part 3: Activations & Gradients, BatchNorm](https://img.youtube.com/vi/P6sfmUTpUmc/0.jpg)](https://www.youtube.com/watch?v=P6sfmUTpUmc)

**۱۱۶ دقیقه** · `P6sfmUTpUmc` · فایل محلی: `03_Building-makemore-Part-3-Activations-_-G_Media_P6sfmUTpUmc_002_720p.mp4`

**موضوع:** مشکل گرادیان ناپایدار، tanh در برابر ReLU، BatchNorm، و Residual connections.
**خروجی جلسه:** مقایسهٔ آزمایشی توپولوژی‌های شبکه.

### ۰۵ — makemore بخش ۴: جادوی بک‌پراپ

[![ویدیوی ۰۵ — Building makemore Part 4: Becoming a Backprop Ninja](https://img.youtube.com/vi/q8SA3rM6ckI/0.jpg)](https://www.youtube.com/watch?v=q8SA3rM6ckI)

**۱۱۵ دقیقه** · `q8SA3rM6ckI` · فایل محلی: `04_Building-makemore-Part-4-Becoming-a-Back_Media_q8SA3rM6ckI_002_720p.mp4`

**موضوع:** درک شهودی گرادیان، چندجمله‌ای‌سازی، روش‌های عددی برای گرادیان، و کار با `micrograd`.
**خروجی جلسه:** ابزارهای تحلیل گرادیان.

### ۰۶ — makemore بخش ۵: ساخت WaveNet

[![ویدیوی ۰۶ — Building makemore Part 5: Building a WaveNet](https://img.youtube.com/vi/t3YJ5hKiMQ0/0.jpg)](https://www.youtube.com/watch?v=t3YJ5hKiMQ0)

**۵۶ دقیقه** · `t3YJ5hKiMQ0` · فایل محلی: `05_Building-makemore-Part-5-Building-a-Wave_Media_t3YJ5hKiMQ0_002_720p.mp4`

**موضوع:** معماری WaveNet با causal convolutions و residual stack.
**خروجی جلسه:** پیاده‌سازی WaveNet.

### ۰۷ — ساخت GPT از صفر

[![ویدیوی ۰۷ — Let's build GPT: from scratch, in code, spelled out](https://img.youtube.com/vi/kCc8FmEb1nY/0.jpg)](https://www.youtube.com/watch?v=kCc8FmEb1nY)

**۱۱۶ دقیقه** · `kCc8FmEb1nY` · فایل محلی: `06_Let-s-build-GPT-from-scratch-in-code-spe_Media_kCc8FmEb1nY_002_720p.mp4`

**موضوع:** attention از صفر، `torch.nn.functional.scaled_dot_product_attention`، BPE tokenizer، AdamW، weight decay، cosine schedule، distributed training.
**خروجی جلسه:** `projects/gpt/` — یک GPT کوچک که واقعاً آموزش می‌بیند.

### ۰۸ — ساخت tokenizer برای GPT

[![ویدیوی ۰۸ — Let's build the GPT Tokenizer](https://img.youtube.com/vi/zduSFxRajkE/0.jpg)](https://www.youtube.com/watch?v=zduSFxRajkE)

**۱۳۴ دقیقه** · `zduSFxRajkE` · فایل محلی: `07_Let-s-build-the-GPT-Tokenizer_Media_zduSFxRajkE_002_720p.mp4`

**موضوع:** BPE از صفر — بایت‌به‌بایت، regex، جفت‌های مرجوع، و مقایسه با `tiktoken`.
**خروجی جلسه:** `projects/tokenizer/` — پیاده‌سازی BPE.

---

## برنامهٔ مطالعه

هر جلسه یک ویدیو، و هر ویدیو چهار گام دارد:

| گام | کار |
|---|---|
| ۱. تماشا | ویدیو را ببین؛ سؤال‌هایت را یادداشت کن (نه کد را) |
| ۲. بازسازی | کد را **از صفر** خودت بنویس — بدون نگاه کردن به راه‌حل |
| ۳. تمرین | تمرین‌های `exercises/` همان جلسه را حل کن |
| ۴. یادداشت | چیزی که فهمیدی را در `notes/` بنویس، به زبان خودت |

اولویت‌بندی موضوعات برای پژوهش (اگر وقت اضافه داری):

- **پایه:** گرادیان descent روی تابع‌های غیرفیت‌سازی، مشتق‌های جزئی، قاعدهٔ زنجیره، پدیدهٔ کوه تند
- **میانی:** SGD در برابر Adam، initialization، normalization، overfitting و regularization
- **پیشرفته:** FlashAttention، KV-cache، mixed-precision، scaling laws، RLHF

---

## ساختار ریپو

```
.
├── README.md              # همین فایل — فهرست ویدیوها و برنامهٔ مطالعه
├── AGENTS.md              # راهنمای کار عامل‌ها با این ریپو
├── micrograd-master/      # کد مرجع upstream کارپتی (دست‌نخورده)
├── notes/                 # یادداشت تدریس هر جلسه
├── exercises/             # تمرین‌های هر جلسه
└── projects/              # پیاده‌سازی‌های از صفر
```

## اجرا

```bash
# تست‌های micrograd (نیازمند torch)
cd micrograd-master && python -m pytest

# دموی آموزش MLP
cd micrograd-master && jupyter nbconvert --to notebook --execute --stdout demo.ipynb
```

## رفرنس‌ها

- [کد ویدیوها — micrograd](https://github.com/karpathy/micrograd) · [makemore](https://github.com/karpathy/makemore) · [nanoGPT](https://github.com/karpathy/nanoGPT) · [build-nanogpt](https://github.com/karpathy/build-nanogpt)
- [متن کامل ویدیوها — llm.c](https://github.com/karpathy/llm.c) (متن کارپتی‌شدهٔ همین دوره)
- [دورهٔ LLMs from Scratch — Sebastian Raschka](https://github.com/Vaibhavs10/Books-LLM)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) · [Layer Normalization](https://arxiv.org/abs/1607.06450) · [AdamW](https://arxiv.org/abs/1711.05101) · [tiktoken](https://github.com/openai/tiktoken)

## لایسنس

MIT — بر پایهٔ [micrograd](https://github.com/karpathy/micrograd) اثر Andréj Karpathy.
