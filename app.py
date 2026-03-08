import gradio as gr
import pandas as pd
from transformers import pipeline

# 1. 📍 ใส่ชื่อโมเดลของเพื่อนตรงนี้ (รอเพื่อนคนที่ 2 เทรนเสร็จ แล้วเอาลิงก์มาเปลี่ยนตรงนี้นะครับ)
MODEL_ID = 'your-username/wangchanberta-wongnai-sentiment'

# 2. เตรียม 5 คำถามสำหรับ Demo Mode ตามโจทย์กำหนด [cite: 43]
DEMOS = [
    ["อาหารอร่อยมาก บรรยากาศดี บริการเยี่ยม กลับมาอีกแน่นอน"],
    ["รอนานมาก อาหารเย็นแล้ว แถมราคาแพงเกินจริง"],
    ["ร้านธรรมดา อาหารกินได้ ราคาโอเค ไม่มีอะไรพิเศษ"],
    ["โคตรอร่อย ราคาถูก คุ้มมากกกก แนะนำเลย"],
    ["ห้องน้ำสกปรก พนักงานไม่สุภาพ อาหารรสชาติแย่มาก ไม่แนะนำเลย"]
]

# 3. โหลดโมเดลผ่าน Pipeline (ใช้ try-except ไว้ก่อน เผื่อเพื่อนยังเทรนไม่เสร็จจะได้รันเว็บได้)
try:
    classifier = pipeline('text-classification', model=MODEL_ID, top_k=None)
except Exception as e:
    print("ระบบจำลอง (Mockup): เนื่องจากยังไม่ได้เชื่อมโมเดลจริง")
    def classifier(text):
        return [[{'label': 'LABEL_0', 'score': 0.8}, {'label': 'LABEL_1', 'score': 0.15}, {'label': 'LABEL_2', 'score': 0.05}]]

def process_prediction(text):
    if not text.strip():
        return {"กรุณาพิมพ์ข้อความ": 1.0}, "-"
        
    results = classifier(text)[0]
    
    # แปลงผลลัพธ์ให้เป็น Positive, Neutral, Negative
    label_map = {'LABEL_0': 'Positive', 'LABEL_1': 'Neutral', 'LABEL_2': 'Negative'}
    
    output_dict = {}
    for res in results:
        label_name = label_map.get(res['label'], res['label'])
        output_dict[label_name] = res['score']
    
    best_label = max(output_dict, key=output_dict.get)
    return output_dict, best_label

# 4. ฟังก์ชันวิเคราะห์ข้อความเดียว + เก็บประวัติ [cite: 36-38]
def predict_single(text, history):
    scores_dict, best_label = process_prediction(text)
    confidence = scores_dict.get(best_label, 0)
    
    # อัปเดตตารางประวัติ
    new_entry = [text, best_label, f"{confidence:.2%}"]
    updated_history = [new_entry] + history
    
    return scores_dict, updated_history

# 5. ฟังก์ชันวิเคราะห์แบบ Batch ด้วยไฟล์ CSV 
def predict_batch(file):
    if file is None:
        return None
    
    df = pd.read_csv(file.name)
    review_col = 'review' if 'review' in df.columns else df.columns[0]
    
    sentiments = []
    confidences = []
    
    for text in df[review_col].astype(str):
        scores_dict, best_label = process_prediction(text)
        sentiments.append(best_label)
        confidences.append(f"{scores_dict.get(best_label, 0):.2%}")
        
    df['Sentiment'] = sentiments
    df['Confidence %'] = confidences
    return df

# 6. สร้างหน้าตา Web App (UI) แบบมี 2 แท็บ [cite: 34]
with gr.Blocks(theme=gr.themes.Soft(), title='Thai Sentiment Analysis') as demo:
    gr.Markdown("# 🇹🇭 Thai Sentiment Analysis: Wongnai Reviews")
    gr.Markdown("วิเคราะห์ความรู้สึกจากรีวิวร้านอาหารด้วยโมเดล WangchanBERTa (Positive / Neutral / Negative)")
    
    with gr.Tabs():
        # --- แท็บที่ 1: วิเคราะห์ข้อความ ---
        with gr.Tab("วิเคราะห์ข้อความ (Single)"):
            with gr.Row():
                with gr.Column():
                    inp = gr.Textbox(label="พิมพ์รีวิวร้านอาหารภาษาไทย", lines=3, placeholder="พิมพ์รีวิวที่นี่...")
                    btn = gr.Button("🔍 วิเคราะห์ Sentiment", variant='primary')
                    
                    gr.Markdown("### 💡 ตัวอย่างรีวิว (Demo Mode)")
                    gr.Examples(examples=DEMOS, inputs=inp)
                    
                with gr.Column():
                    # Sentiment Bar แสดงผล 3 คลาสแบบ Real-time 
                    out_label = gr.Label(label='Sentiment Confidence Score (Sentiment Bar)', num_top_classes=3)
            
            # ตารางประวัติ Query History 
            gr.Markdown("### 🕒 ประวัติการทำนายในเซสชันนี้ (Query History)")
            history_state = gr.State([])
            history_table = gr.Dataframe(headers=["ข้อความรีวิว", "ผลลัพธ์", "ความมั่นใจ %"], interactive=False)
            
            btn.click(fn=predict_single, inputs=[inp, history_state], outputs=[out_label, history_state])
            history_state.change(fn=lambda h: h, inputs=history_state, outputs=history_table)

        # --- แท็บที่ 2: วิเคราะห์จากไฟล์ CSV ---
        with gr.Tab("วิเคราะห์หลายรายการ (Batch CSV Upload)"):
            gr.Markdown("อัปโหลดไฟล์ CSV ที่มีคอลัมน์ชื่อ `review` เพื่อวิเคราะห์ทีละหลายรายการพร้อมกัน")
            with gr.Row():
                with gr.Column():
                    csv_inp = gr.File(label="อัปโหลดไฟล์ CSV ตรงนี้", file_types=['.csv'])
                    csv_btn = gr.Button("📂 วิเคราะห์ไฟล์ CSV", variant='primary')
                with gr.Column():
                    csv_out = gr.Dataframe(label="ผลลัพธ์การวิเคราะห์แบบกลุ่ม")
            
            csv_btn.click(fn=predict_batch, inputs=csv_inp, outputs=csv_out)

if __name__ == "__main__":
    demo.launch()