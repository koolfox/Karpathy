# Neural Networks: Zero to Hero — سفر یادگیری

> «What I cannot create, I do not understand.» — Richard Feynman

ریپوی آموزشی و اوپن‌سورس برای یادگیری هوش مصنوعی از صفر، بر پایه‌ی دوره‌ی
**Neural Networks: Zero to Hero** اثر [Andrej Karpathy](https://github.com/karpathy).

اینجا کد را **می‌خوانیم، می‌فهمیم و از صفر می‌نویسیم** — نه اینکه فقط کپی کنیم.

---

## ویدیوها

هر ویدیو یک جلسه‌ی کامل است. نسخه‌ی محلی (MP4) در همین پوشه نگه‌داری می‌شود ولی به‌دلیل حجم بالا (۱.۷ گیگابایت) در ریپو commit نمی‌شود؛ لینک‌های زیر embed یوتیوب هستند.

### ۰۱ — مقدمه‌ی کامل شبکه‌های عصبی و بک‌پراپ: ساخت micrograd

`VMj-3S1tku0` · ۱۴۶ دقیقه · `00_The-spelled-out-intro-to-neural-networks_Media_VMj-3S1tku0_003_480p.mp4`

<iframe width="100%" height="380" src="https://www.youtube.com/embed/VMj-3S1tku0" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen title="The spelled-out intro to neural networks and backpropagation: building micrograd"></iframe>

**موضوع:** از صفر یک موتور autodiff اسکالر-مقدار می‌سازیم: گراف پویا، مرتب‌سازی توپولوژیک، قاعده‌ی زنجیره، و `backward()`.
**خروجی جلسه:** `projects/micrograd/` — پیاده‌سازی خودمان از `Value` و یک MLP دو لایه.

### ۰۲ — مقدمه‌ی کامل مدل‌سازی زبان: ساخت makemore

`PaCmpygFfXo` · ۱۱۸ دقیقه · `01_The-spelled-out-intro-to-language-modeli_Media_PaCmpygFfXo_002_720p.mp4`

<iframe width="100%" height="380" src="https://www.youtube.com/embed/PaCmpygFfXo" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen title="The spelled-out intro to language modeling: building makemore"></iframe>

**موضوع:** bigram و سپس bigram با کانولوشن چندسطحی، آمار هم‌بستگی، نمونه‌گیری و temperature.
**خروجی جلسه:** `projects/makemore/` — تولیدکننده‌ی متن.

### ۰۳ — makemore بخش ۲: MLP

`TCH_1BHY58I` · ۷۶ دقیقه · `02_Building-makemore-Part-2-MLP_Media_TCH_1BHY58I_002_720p.mp4`

<iframe width="100%" height="380" src="https://www.youtube.com/embed/TCH_1BHY58I" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen title="Building makemore Part 2: MLP"></iframe>

**موضوع:** چرا MLP بهتر از bigram است — با همان ابزار micrograd که خودمان ساختیم.
**خروجی جلسه:** آموزش MLP روی دیتاست‌های toy.

### ۰۴ — makemore بخش ۳: اکتیویشن‌ها و جریان گرادیان

`P6sfmUTpUmc` · ۱۱۶ دقیقه · `03_Building-makemore-Part-3-Activations-_-G_Media_P6sfmUTpUmc_002_720p.mp4`

<iframe width="100%" height="380" src="https://www.youtube.com/embed/P6sfmUTpUmc" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen title="Building makemore Part 3: Activations & Gradients, BatchNorm"></iframe>

**موضوع:** مشکل گرادیان ناپایدار، tanh در برابر ReLU، BatchNorm، و Residual connections.
**خروجی جلسه:** مقایسه‌ی آزمایشی توپولوژی‌های شبکه.

### ۰۵ — makemore بخش ۴: جادوی بک‌پراپ

`q8SA3rM6ckI` · ۱۱۵ دقیقه · `04_Building-makemore-Part-4-Becoming-a-Back_Media_q8SA3rM6ckI_002_720p.mp4`

<iframe width="100%" height="380" src="https://www.youtube.com/embed/q8SA3rM6ckI" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen title="Building makemore Part 4: Becoming a Backprop Ninja"></iframe>

**موضوع:** درک شهودی گرادیان، چندجمله‌ای‌سازی، روش‌های عددی برای گرادیان، و کار با `micrograd`.
**خروجی جلسه:** ابزارهای تحلیل گرادیان.

### ۰۶ — makemore بخش ۵: ساخت WaveNet

`t3YJ5hKiMQ0` · ۵۶ دقیقه · `05_Building-makemore-Part-5-Building-a-Wave_Media_t3YJ5hKiMQ0_002_720p.mp4`

<iframe width="100%" height="380" src="https://www.youtube.com/embed/t3YJ5hKiMQ0" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen title="Building makemore Part 5: Building a WaveNet"></iframe>

**موضوع:** معماری WaveNet با causal convolutions و residual stack.
**خروجی جلسه:** پیاده‌سازی WaveNet.

### ۰۷ — ساخت GPT از صفر

`kCc8FmEb1nY` · ۱۱۶ دقیقه · `06_Let-s-build-GPT-from-scratch-in-code-spe_Media_kCc8FmEb1nY_002_720p.mp4`

<iframe width="100%" height="380" src="https://www.youtube.com/embed/kCc8FmEb1nY" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen title="Let's build GPT: from scratch, in code, spelled out."></iframe>

**موضوع:** attention از صفر، `torch.nn.functional.scaled_dot_product_attention`، BPE tokenizer، AdamW، weight decay، cosine schedule، distributed training.
**خروجی جلسه:** `projects/gpt/` — یک GPT کوچک که واقعاً آموزش می‌بیند.

### ۰۸ — ساخت tokenizer برای GPT

`zduSFxRajkE` · ۱۳۴ دقیقه · `07_Let-s-build-the-GPT-Tokenizer_Media_zduSFxRajkE_002_720p.mp4`

<iframe width="100%" height="380" src="https://www.youtube.com/embed/zduSFxRajkE" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen title="Let's build the GPT Tokenizer"></iframe>

**موضوع:** BPE از صفر — بایت‌به‌بایت، regex، جفت‌های مرجوع، و مقایسه با `tiktoken`.
**خروجی جلسه:** `projects/tokenizer/` — پیاده‌سازی BPE.

---

## برنامه‌ی مطالعه

هر جلسه یک ویدیو، و هر ویدیو چهار گام دارد:

| گام | کار |
|---|---|
| ۱. تماشا | ویدیو را ببین؛ سؤال‌هایت را یادداشت کن (نه کد را) |
| ۲. بازسازی | کد را **از صفر** خودت بنویس — بدون نگاه کردن به راه‌حل |
| ۳. تمرین | تمرین‌های `exercises/` همان جلسه را حل کن |
| ۴. یادداشت | چیزی که فهمیدی را در `notes/` بنویس، به زبان خودت |

اولویت‌بندی موضوعات برای پژوهش (اگر وقت اضافه داری):

- **پایه:** گرادیان descent روی تابع‌های غیرفیت‌سازی، مشتق‌های جزئی، قاعده‌ی زنجیره، پدیده‌ی کوه تند
- **میانی:** SGD در برابر Adam، initialization،/normalization، overfitting و regularization
- **پیشرفته:** FlashAttention، KV-cache، mixed-precision، scaling laws، RLHF

---

## ساختار ریپو

```
.
├── README.md              # همین فایل — فهرست ویدیوها و برنامه‌ی مطالعه
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
- [متن کامل ویدیوها — llm.c](https://github.com/karpathy/llm.c) (متن کارپتی‌شده‌ی همین دوره)
- [دوره‌ی LLMs from Scratch — Sebastian Raschka](https://github.com/Vaibhavs10/Books-LLM)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) · [Layer Normalization](https://arxiv.org/abs/1607.06450) · [AdamW](https://arxiv.org/abs/1711.05101) · [tiktoken](https://github.com/openai/tiktoken)

## لایسنس

MIT — بر پایه‌ی [micrograd](https://github.com/karpathy/micrograd) اثر Andréj Karpathy.
