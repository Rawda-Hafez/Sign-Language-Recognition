# 🤟 ASL Sign Language Recognition System

نظام ذكي للتعرف على حروف لغة الإشارة الأمريكية (American Sign Language - ASL) في الوقت الفعلي باستخدام شبكات التعلم العميق (Deep CNN) وواجهة تفاعلية حديثة مبنية بـ **Streamlit**.

---

## 📌 مميزات المشروع (Key Features)

1. **بث مباشر من الكاميرا (Live Webcam with ROI Bounding Box):**
   - تم حل مشكلة تشويش الخلفية بوضع مربع تركيز أخضر (`Place Hand Here`) ليتم اقتصاص كف اليد وتحليله فقط.
   - تشغيل وإيقاف سلس للكاميرا عبر أزرار التحكم بدون تجميد المتصفح.
2. **التقاط فوري عبر المتصفح (Camera Snapshot):**
   - استخدام ميزة `st.camera_input` لالتقاط صورة اليد مباشرة من أي متصفح.
3. **رفع الصور وتحليلها (Image Upload):**
   - إمكانية رفع أي صورة بصيغ (JPG, JPEG, PNG) ومعاينتها وتصنيفها فوراً.
4. **حساب نسبة الثقة وعرض أفضل 3 احتمالات (Top-3 Predictions):**
   - عرض الحرف المتوقع بنسبة الثقة المئوية (`Confidence Score`).
   - إمكانية ضبط حد القبول الأدنى (`Confidence Threshold`) من الشريط الجانبي.
   - عرض شريط تقدم لأعلى 3 حروف محتملة (Top-3 Probabilities).
5. **معاينة دخل النموذج (28x28 Input Preview):**
   - إظهار الصورة بعد المعالجة المسبقة للمستخدم لمساعدته في ضبط الإضاءة ووضعية اليد.

---

## ℹ️ الحروف المدعومة (Supported Characters)

يدعم النموذج **24 حرفاً** من الأبجدية الإنجليزية:
> **A, B, C, D, E, F, G, H, I, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y**

⚠️ **ملاحظة هامة:**
تم استبعاد حرفي **(J و Z)** في مجموعة بيانات **Sign Language MNIST** لأن تمثيلهما في لغة الإشارة الأمريكية يعتمد على حركة اليد (Dynamic Gestures)، بينما النموذج مبني على الإشارات الثابتة (Static Gestures).

---

## 📂 هيكلية المشروع (Project Structure)

```text
sign_language/
│
├── dataset/                     # مجلد بيانات الصور المصنفة لكل حرف (a-y)
├── Sign_Language.ipynb          # دفتر تدريب واختبار النموذج والـ Fine-Tuning
├── app.py                       # تطبيق واجهة المستخدم (Streamlit App)
├── label_map.json               # خريطة أرقام التصنيفات إلى الحروف
├── sign_mnist_finetuned.keras   # نموذج التعلم العميق النهائي المحفوظ
├── sign_mnist_finetuned.h5      # نسخة احتياطية من أوزان النموذج (H5)
├── sign_mnist_train.csv         # بيانات التدريب Sign MNIST
├── sign_mnist_test.csv          # بيانات الاختبار Sign MNIST
├── requirements.txt             # المكتبات والاعتماديات المطلوبة
├── .gitignore                   # تجاهل الملفات المؤقتة والبيانات الضخمة
└── README.md                    # ملف التوثيق ودليل الاستخدام
```

---

## 🚀 طريقة التثبيت والتشغيل (How to Run)

### 1. تجهيز بيئة العمل (Virtual Environment)
يفضل إنشاء بيئة افتراضية جديدة:
```bash
python -m venv venv
# تفعيل البيئة على ويندوز:
venv\Scripts\activate
```

### 2. تثبيت المكتبات المطلوبة (Install Dependencies)
```bash
pip install -r requirements.txt
```

### 3. تشغيل التطبيق (Run Streamlit App)
```bash
streamlit run app.py
```
سيفتح التطبيق تلقائياً في المتصفح على العنوان: `http://localhost:8501`.

---

## 🧠 تدريب وبناء النموذج (Model Architecture & Training)

- **البيانات الأولية:** Sign Language MNIST (28x28 Grayscale images).
- **بنية الشبكة (CNN Architecture):**
  - طبقات تلافيفية متعددة (`Conv2D` + `BatchNormalization` + `ReLU`).
  - طبقات تجميع أقصى (`MaxPooling2D`).
  - طبقات لمنع الإفراط في التخصيص (`Dropout`).
  - طبقات كثيفة كاملة الاتصال (`Dense` + `Softmax` لـ 24 فئة).
- **الضبط الدقيق (Fine-Tuning):**
  - تم استخدام توليد وتكبير البيانات (`ImageDataGenerator`) لزيادة قدرة النموذج على التعميم (Rotation, Shifting, Zooming, Brightness).
  - تم تحقيق دقة تتجاوز **98%** على مجموعة التحقق (`Validation Accuracy`).
