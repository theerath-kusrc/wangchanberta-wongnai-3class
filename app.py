import streamlit as st
import pandas as pd
from transformers import pipeline
import torch

# 1. การตั้งค่าหน้าเว็บ
st.set_page_config(
    page_title="Wongnai Sentiment Analysis (3-Class)",
    page_icon="🇹🇭",
    layout="wide"
)

from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

model_name = "Kanyasiri/wangchanberta-wongnai-3class"

# 1. โหลด Config มาก่อน
config = AutoConfig.from_pretrained(model_name)

# 2. บังคับใส่ model_type เป็น camembert (เพราะ WangchanBERTa ใช้โครงสร้างนี้)
if not hasattr(config, "model_type") or config.model_type is None:
    config.model_type = "camembert"

# 3. โหลด Tokenizer และ Model โดยใช้ Config ที่เราแก้ไขแล้ว
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)

@st.cache_resource
def load_model():
    try:
        # โหลด pipeline สำหรับ Text Classification
        return pipeline('text-classification', model=MODEL_ID, top_k=None)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

classifier = load_model()

# 3. ส่วนหัวของเว็บไซต์
st.title("🍽️ Thai Sentiment Analysis: Wongnai Reviews")
st.markdown("วิเคราะห์ความรู้สึกจากรีวิวด้วยโมเดล **WangchanBERTa** (รองรับ Positive / Neutral / Negative)")

# 4. ระบบจัดการประวัติ (Sidebar)
if 'history' not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.header("🕒 History")
    if st.session_state.history:
        for i, item in enumerate(st.session_state.history):
            st.info(f"**{i+1}.** {item['text'][:30]}...\n**Result:** {item['label']}")
    else:
        st.write("ยังไม่มีประวัติ")
    if st.button("ล้างประวัติ"):
        st.session_state.history = []
        st.rerun()

# 5. ส่วนการทำงานหลัก (Tabs)
tab1, tab2 = st.tabs(["🔍 วิเคราะห์ข้อความเดี่ยว", "📂 อัปโหลดไฟล์ CSV"])

with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        input_text = st.text_area("พิมพ์รีวิวร้านอาหารที่นี่:", placeholder="ตัวอย่าง: อาหารอร่อยมาก แต่พนักงานบริการช้าหน่อย...", height=150)
        predict_btn = st.button("วิเคราะห์ผล", type="primary", use_container_width=True)
        
        st.markdown("---")
        st.markdown("### 💡 ตัวอย่างรีวิว")
        examples = [
            "อร่อยมากครับ แนะนำเลยร้านนี้",
            "รสชาติกลางๆ พอใช้ได้ครับ",
            "แย่มาก อาหารไม่สดเลย เสียความรู้สึก"
        ]
        for ex in examples:
            if st.button(ex):
                st.info(f"คัดลอกข้อความนี้ไปวางด้านบน: {ex}")

    with col2:
        if predict_btn and input_text:
            if classifier:
                with st.spinner('กำลังประมวลผล...'):
                    results = classifier(input_text)[0]
                    
                    # 🚨 จุดสำคัญ: Mapping Label (0, 1, 2) ให้ตรงกับที่เพื่อนเทรน
                    # ปกติ 3-class คือ 0: Positive, 1: Neutral, 2: Negative
                    label_map = {
                        'LABEL_0': 'Positive 😊', 
                        'LABEL_1': 'Neutral 😐', 
                        'LABEL_2': 'Negative 😠'
                    }
                    
                    scores_dict = {label_map.get(r['label'], r['label']): r['score'] for r in results}
                    best_label = max(scores_dict, key=scores_dict.get)
                    
                    st.subheader("ผลการวิเคราะห์:")
                    for label, score in scores_dict.items():
                        st.write(f"**{label}**")
                        st.progress(score)
                        st.caption(f"Confidence: {score:.2%}")
                    
                    # บันทึกประวัติ
                    st.session_state.history.insert(0, {"text": input_text, "label": best_label})
            else:
                st.warning("ระบบยังโหลดโมเดลไม่สำเร็จ")

with tab2:
    st.subheader("วิเคราะห์หลายรายการ (Batch Processing)")
    csv_file = st.file_uploader("เลือกไฟล์ CSV (ต้องมีคอลัมน์ชื่อ 'review')", type=["csv"])
    
    if csv_file and classifier:
        df = pd.read_csv(csv_file)
        if st.button("เริ่มวิเคราะห์ไฟล์ CSV"):
            with st.spinner('กำลังคำนวณ...'):
                col_name = 'review' if 'review' in df.columns else df.columns[0]
                
                def get_sentiment(text):
                    res = classifier(str(text))[0]
                    top = max(res, key=lambda x: x['score'])
                    # ใช้ Mapping เดียวกัน
                    m = {'LABEL_0': 'Positive', 'LABEL_1': 'Neutral', 'LABEL_2': 'Negative'}
                    return m.get(top['label'], top['label']), f"{top['score']:.2%}"

                df[['Result', 'Confidence']] = df[col_name].apply(lambda x: pd.Series(get_sentiment(x)))
                st.success("สำเร็จ!")
                st.dataframe(df)
                
                csv_data = df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 ดาวน์โหลดผลลัพธ์", csv_data, "result.csv", "text/csv")



