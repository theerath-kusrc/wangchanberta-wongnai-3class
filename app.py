import streamlit as st
import pandas as pd
from transformers import pipeline
import torch

# 1. ⚙️ การตั้งค่าหน้าเว็บ (เก็บคะแนน UI/UX)
st.set_page_config(
    page_title="Thai Sentiment Analysis - Wongnai",
    page_icon="🍽️",
    layout="wide"
)

# 2. 📍 เชื่อมต่อโมเดล (ปรับ MODEL_ID ให้ตรงกับที่เพื่อนเทรน 3-Class)
# หากเพื่อนยังไม่ได้เทรน 3-class ให้ใช้ชื่อเดิม แต่ต้องแก้ num_labels ใน Colab เป็น 3
MODEL_ID = 'Kanyasiri/wangchanberta-wongnai-sentiment'

# ฟังก์ชันโหลดโมเดลแบบ Cache เพื่อไม่ให้โหลดใหม่ทุกครั้งที่กดปุ่ม
@st.cache_resource
def load_classifier():
    try:
        return pipeline('text-classification', model=MODEL_ID, top_k=None)
    except:
        return None

classifier = load_classifier()

# 3. 📝 สไตล์และส่วนหัว
st.title("🇹🇭 Thai Sentiment Analysis: Wongnai Reviews")
st.markdown("""
แอปพลิเคชันวิเคราะห์ความรู้สึกจากรีวิวร้านอาหาร โดยใช้โมเดล **WangchanBERTa** รองรับการจำแนก 3 ระดับ: **Positive (บวก)**, **Neutral (กลาง)**, และ **Negative (ลบ)**
""")

# 4. 🕒 ระบบประวัติการใช้งาน (Sidebar)
if 'history' not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.header("🕒 Recent History")
    if st.session_state.history:
        for idx, item in enumerate(st.session_state.history):
            with st.expander(f"{idx+1}. {item['text'][:20]}..."):
                st.write(f"**รีวิว:** {item['text']}")
                st.write(f"**ผลลัพธ์:** {item['label']}")
    else:
        st.caption("ยังไม่มีประวัติการวิเคราะห์")
    
    if st.button("ล้างประวัติ"):
        st.session_state.history = []
        st.rerun()

# 5. 🛠️ ส่วนการทำงานหลัก (Tabs)
tab1, tab2, tab3 = st.tabs(["🔍 วิเคราะห์ข้อความ", "📂 อัปโหลด CSV", "💡 ตัวอย่าง (Demo)"])

with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        input_text = st.text_area("กรอกรีวิวภาษาไทยที่นี่:", placeholder="เช่น อาหารรสชาติปานกลาง แต่บริการค่อนข้างช้า...", height=150)
        predict_btn = st.button("วิเคราะห์ความรู้สึก", type="primary", use_container_width=True)

    with col2:
        if predict_btn and input_text:
            if classifier:
                with st.spinner('กำลังประมวลผล...'):
                    raw_results = classifier(input_text)[0]
                    
                    # 🚨 Mapping 3 คลาส (เช็กกับเพื่อนว่า 0, 1, 2 เรียงอย่างไร)
                    # มาตรฐาน: 0=Pos, 1=Neu, 2=Neg
                    label_map = {
                        'LABEL_0': 'Positive 😊', 
                        'LABEL_1': 'Neutral 😐', 
                        'LABEL_2': 'Negative 😠'
                    }
                    
                    results_dict = {label_map.get(r['label'], r['label']): r['score'] for r in raw_results}
                    best_label = max(results_dict, key=results_dict.get)
                    
                    # แสดงผล
                    st.subheader("ผลลัพธ์:")
                    for label, score in results_dict.items():
                        st.write(f"**{label}**")
                        st.progress(score)
                        st.caption(f"Confidence: {score:.2%}")
                    
                    # บันทึกลง Session History
                    st.session_state.history.insert(0, {"text": input_text, "label": best_label})
            else:
                st.error("ไม่สามารถเชื่อมต่อโมเดลได้ กรุณาตรวจสอบ MODEL_ID")

with tab2:
    st.subheader("Batch Processing via CSV")
    uploaded_file = st.file_uploader("เลือกไฟล์ CSV (ต้องมีคอลัมน์ 'review')", type=["csv"])
    
    if uploaded_file and classifier:
        df = pd.read_csv(uploaded_file)
        if st.button("เริ่มวิเคราะห์ทั้งไฟล์"):
            with st.spinner('กำลังวิเคราะห์...'):
                review_col = 'review' if 'review' in df.columns else df.columns[0]
                
                def get_sentiment(text):
                    res = classifier(str(text))[0]
                    top = max(res, key=lambda x: x['score'])
                    # ใช้ Mapping เดียวกับข้างบน
                    label_map_batch = {'LABEL_0': 'Positive', 'LABEL_1': 'Neutral', 'LABEL_2': 'Negative'}
                    return label_map_batch.get(top['label'], top['label']), top['score']

                # ประมวลผล
                results = df[review_col].apply(get_sentiment)
                df['Sentiment'] = [r[0] for r in results]
                df['Confidence'] = [f"{r[1]:.2%}" for r in results]
                
                st.success("วิเคราะห์สำเร็จ!")
                st.dataframe(df, use_container_width=True)
                
                # ปุ่มดาวน์โหลด
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 ดาวน์โหลดผลลัพธ์", csv, "sentiment_results.csv", "text/csv")

with tab3:
    st.info("กดที่ข้อความเพื่อนำไปใส่ในช่องวิเคราะห์")
    demo_samples = [
        "อาหารอร่อยมาก พนักงานบริการดีเยี่ยม บรรยากาศในร้านดีสุดๆ",
        "รสชาติอาหารพอใช้ได้ แต่รอนานเกินไปหน่อย ราคาแอบสูง",
        "แย่มากครับ แมลงสาบวิ่งบนโต๊ะ อาหารไม่สด ไม่แนะนำอย่างยิ่ง"
    ]
    for sample in demo_samples:
        if st.button(sample):
            st.info(f"ก๊อปปี้ข้อความนี้ไปวางในแท็บแรกได้เลย: \n\n {sample}")
