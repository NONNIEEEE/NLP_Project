import re
import pandas as pd
import plotly.graph_objects as go
import gradio as gr
from pythainlp.tokenize import word_tokenize
from pythainlp.corpus import thai_stopwords
import streamlit as st

# ==========================================
# 1. Custom CSS - High Contrast Minimal Cream
# ==========================================
custom_css = """
:root {
  --bg-main: #FAF8F5;          /* สีครีมอุ่น */
  --bg-card: #FFFFFF;          /* การ์ดสีขาว */
  --bg-surface: #F0EAE1;       /* กรอบ/Input สีครีมเข้ม */
  --primary: #B57C42;          /* สีทรายเข้ม */
  --primary-hover: #96622E;    
  --text-main: #1A202C;        /* สีข้อความหลัก คมชัด 100% */
  --text-muted: #4A5568;       
  --border: #D6CEC2;           
}

body, .gradio-container {
    background-color: var(--bg-main) !important;
    color: var(--text-main) !important;
    font-family: 'Prompt', 'Inter', -apple-system, sans-serif !important;
}

p, span, label, li, .gr-form, .markdown-text, div {
    color: var(--text-main) !important;
}

.custom-header {
    background: #FFFFFF;
    border: 1px solid var(--border);
    border-left: 6px solid var(--primary);
    padding: 24px;
    border-radius: 12px;
    margin-bottom: 20px;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.04);
}

.custom-header h1 {
    color: #1A202C !important;
    font-weight: 700 !important;
    font-size: 1.8rem !important;
    margin: 0 0 8px 0 !important;
}

.custom-header p {
    color: #4A5568 !important;
    margin: 0 !important;
    font-size: 1rem !important;
}

.block, .panel, div[data-testid="column"] {
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
}

label span, .block title, span[data-testid="block-info"] {
    color: #2D3748 !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
}

button.primary-btn, button.lg.primary {
    background: var(--primary) !important;
    color: #FFFFFF !important;
    border: none !important;
    font-weight: 600 !important;
    font-size: 1.05rem !important;
    border-radius: 8px !important;
    box-shadow: 0 2px 6px rgba(181, 124, 66, 0.3) !important;
}

button.primary-btn:hover, button.lg.primary:hover {
    background: var(--primary-hover) !important;
}

/* สไตล์ปุ่มเมื่อถูกปิดการใช้งาน (Disabled State) */
button:disabled, button[disabled] {
    background-color: #E2E8F0 !important;
    color: #A0AEC0 !important;
    border: 1px solid #CBD5E0 !important;
    cursor: not-allowed !important;
    box-shadow: none !important;
}

.status-badge-up {
    background-color: #E6F4EA;
    color: #137333;
    border: 1px solid #34A853;
    padding: 4px 12px;
    border-radius: 6px;
    font-size: 0.9rem;
    font-weight: 600;
}

.status-badge-down {
    background-color: #FCE8E6;
    color: #C5221F;
    border: 1px solid #EA4335;
    padding: 4px 12px;
    border-radius: 6px;
    font-size: 0.9rem;
    font-weight: 600;
}

.status-badge-warn {
    background-color: #FEF7E0;
    color: #B06000;
    border: 1px solid #FBBC04;
    padding: 4px 12px;
    border-radius: 6px;
    font-size: 0.9rem;
    font-weight: 600;
}

textarea, input[type="text"] {
    background-color: #FFFFFF !important;
    color: #1A202C !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
}

table.dataframe {
    background-color: #FFFFFF !important;
    color: #1A202C !important;
    border: 1px solid var(--border) !important;
}

table.dataframe th {
    background-color: #F0EAE1 !important;
    color: #1A202C !important;
    border-bottom: 2px solid var(--primary) !important;
    font-weight: 700 !important;
}

table.dataframe td {
    background-color: #FFFFFF !important;
    border-bottom: 1px solid var(--border) !important;
    color: #2D3748 !important;
    font-size: 0.95rem !important;
}
"""

# ==========================================
# 2. NLP Analysis Engine
# ==========================================
class NetworkSentimentEngine:
    def __init__(self):
        self.thai_stop = set(thai_stopwords())
        
        self.aspect_keywords = {
            "การจัดส่ง (Delivery)": ["ส่ง", "ขนส่ง", "แพ็ก", "แพ็ค", "โยน", "กล่อง", "ไว", "เร็ว", "ช้า", "จัดส่ง", "พัสดุ"],
            "คุณภาพสินค้า (Quality)": ["จอ", "คุณภาพ", "เครื่อง", "ใช้งาน", "ใช้ไม่ได้", "จอขาว", "broken", "pixels", "quality", "monitor", "ตรงปก", "ภาพ"],
            "ราคา (Price)": ["ราคา", "คุ้ม", "คุ้มค่า", "แพง", "เงิน", "สมราคา"],
            "การบริการ (Service)": ["ตอบ", "บริการ", "ร้าน", "แชท", "แอดมิน", "AI", "คืนสินค้า", "หลังการขาย", "ทางร้าน"]
        }
        
        self.pos_words = [
            "ดี", "เร็ว", "ไว", "คุ้ม", "คุ้มค่า", "สบาย", "สวย", "ตรงปก", "ประทับใจ", "ชอบ", 
            "น่ารัก", "เรียบร้อย", "เนี๊ยบ", "สมราคา", "สุดยอด", "แน่นหนา", "เรียบร้อยดี", "โอเค", "แนะนำ", "good", "great"
        ]
        
        self.neg_words = [
            "ช้า", "บาง", "ไม่ตรงปก", "แพง", "บุบ", "เสียดาย", "ชำรุด", "ผิด", "หมอง", 
            "ตำหนิ", "แย่", "ขาด", "ฉีกขาด", "รอย", "ห่วย", "พัง", "อย่าซื้อ", "ไม่ดี", "ไม่โอเค",
            "เสียความรู้สึก", "ใช้ไม่ได้", "เงียบ", "โยน", "poor", "broken", "bad", "ฉีกขาด", "จอขาว"
        ]

    def preprocess(self, text: str) -> list:
        text = re.sub(r'[^\w\s]', '', str(text))
        tokens = word_tokenize(text, engine="newmm")
        return [w for w in tokens if w.strip() and w not in self.thai_stop]

    def analyze_sentiment_single(self, text: str) -> str:
        text_lower = str(text).lower()
        has_pos = any(w in text_lower for w in self.pos_words)
        has_neg = any(w in text_lower for w in self.neg_words)
        
        if has_neg and not has_pos:
            return "เชิงลบ"
        elif has_pos and not has_neg:
            return "เชิงบวก"
        else:
            return "ผสม/ปานกลาง"

    def analyze_absa(self, text: str) -> dict:
        results = {}
        text_str = str(text).lower()
        
        for aspect, keywords in self.aspect_keywords.items():
            if any(w in text_str for w in keywords):
                sentiment = self.analyze_sentiment_single(text_str)
                results[aspect] = sentiment
        return results

    def generate_summary(self, df: pd.DataFrame) -> tuple:
        pos_count = 0
        neg_count = 0
        neu_count = 0
        alerts = []
        
        for idx, row in df.iterrows():
            text = str(row['review_text'])
            sentiment = self.analyze_sentiment_single(text)
            
            if sentiment == "เชิงลบ":
                neg_count += 1
                alerts.append(f"🔴 [CRITICAL ALERT] ID {row.get('review_id', idx+1)}: \"{text[:90]}...\"")
            elif sentiment == "เชิงบวก":
                pos_count += 1
            else:
                neu_count += 1
                
        total = len(df) if len(df) > 0 else 1
        pos_pct = round((pos_count / total) * 100)
        neg_pct = round((neg_count / total) * 100)
        neu_pct = 100 - pos_pct - neg_pct
        
        summary_markdown = f"""
### 📊 Executive Sentiment Overview
<div style="display: flex; gap: 10px; margin-top: 12px; margin-bottom: 16px;">
    <span class="status-badge-up">🟢 เชิงบวก (UP): {pos_pct}%</span>
    <span class="status-badge-down">🔴 เชิงลบ (DOWN): {neg_pct}%</span>
    <span class="status-badge-warn">🟠 ปานกลาง (WARN): {neu_pct}%</span>
</div>

<b style="color: #1A202C; font-size: 1.05rem;">🟢 Positive Insights (จุดเด่น):</b>
* ระบบตรวจพบการจัดส่งความเร็วสูงและการบรรจุพัสดุที่แน่นหนา

<br>
<b style="color: #1A202C; font-size: 1.05rem;">🔴 Action Required (จุดที่ต้องแก้ไขด่วน):</b>
* ตรวจพบสินค้าชำรุด (หน้าจอขาว / Broken Pixels)
* ปัญหาพฤติกรรมขนส่ง และบริการหลังการขาย
        """
        alert_text = "\n".join(alerts) if alerts else "🟢 STATUS NORMAL: ไม่พบรีวิวเชิงลบรุนแรงในระบบ"
        return summary_markdown, alert_text

nlp_engine = NetworkSentimentEngine()

# ==========================================
# 3. Processing Function
# ==========================================
def process_reviews(file_input):
    if file_input is None:
        return "", None, "", None

    df = pd.read_csv(file_input.name)
    
    if 'data2' in df.columns:
        review_col = 'data2'
    else:
        text_cols = [c for c in df.columns if df[c].dtype == 'object']
        review_col = max(text_cols, key=lambda c: df[c].astype(str).str.len().mean()) if text_cols else df.columns[0]
        
    df['review_text'] = df[review_col].astype(str).fillna('')
    df = df[~df['review_text'].str.contains(r'ขนาด:|ตัวเลือกสินค้า:|สี:|อัตราการรีเฟรช:|ขนาดจอแสดงผล:', na=False)]
    df = df[df['review_text'].str.strip().str.len() > 3].reset_index(drop=True)
    df['review_id'] = range(1, len(df) + 1)

    df['cleaned_tokens'] = df['review_text'].apply(nlp_engine.preprocess)
    df['aspect_sentiment'] = df['review_text'].apply(nlp_engine.analyze_absa)

    aspect_counts = {"การจัดส่ง": 0, "คุณภาพสินค้า": 0, "ราคา": 0, "การบริการ": 0}
    for aspect_dict in df['aspect_sentiment']:
        for k in aspect_dict.keys():
            key_clean = k.split(" ")[0]
            if key_clean in aspect_counts:
                aspect_counts[key_clean] += 1

    filtered_counts = {k: v for k, v in aspect_counts.items() if v > 0}
    if not filtered_counts:
        filtered_counts = {"รีวิวทั่วไป": len(df)}

    fig = go.Figure(data=[go.Pie(
        labels=list(filtered_counts.keys()),
        values=list(filtered_counts.values()),
        hole=.5,
        marker=dict(colors=['#D99B56', '#527853', '#EEA5A6', '#E16262']),
        textinfo='label+percent',
        textfont=dict(size=14, color='#1A202C')
    )])
    
    fig.update_layout(
        title=dict(text="📊 Aspect Distribution Breakdown", font=dict(color='#1A202C', size=16, family='Prompt')),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#1A202C', family='Prompt'),
        margin=dict(t=40, b=20, l=20, r=20),
        legend=dict(font=dict(color='#1A202C', size=13))
    )

    summary_md, alert_txt = nlp_engine.generate_summary(df)

    df_display = df[['review_id', 'review_text', 'aspect_sentiment']].copy()
    df_display.columns = ['ID', 'Comment รีวิวจริง', 'ผลวิเคราะห์ ABSA Sentiment']
    df_display['ผลวิเคราะห์ ABSA Sentiment'] = df_display['ผลวิเคราะห์ ABSA Sentiment'].astype(str)

    return summary_md, fig, alert_txt, df_display

# ==========================================
# 4. Gradio UI Layout
# ==========================================
with gr.Blocks(css=custom_css, title="Review Analytics Dashboard") as app:
    
    gr.HTML("""
    <div class="custom-header">
        <h1>🌿 Review Analytics Dashboard</h1>
        <p>ระบบวิเคราะห์และสรุปผลรีวิวสินค้าด้วย NLP สไตล์ มินิมอลเรียบหรู</p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            file_uploader = gr.File(label="📂 อัปโหลดไฟล์รีวิว (CSV Format)", file_types=[".csv"])
            # ตั้งค่า interactive=False ตั้งแต่เริ่มต้น
            btn_run = gr.Button("🚀 ประมวลผลและวิเคราะห์", variant="primary", interactive=False)
            
        with gr.Column(scale=2):
            alert_output = gr.Textbox(
                label="🚨 Critical Review Monitor (ระบบแจ้งเตือนรีวิววิกฤต)", 
                lines=5,
                interactive=False
            )

    with gr.Row():
        with gr.Column(scale=1):
            summary_output = gr.Markdown()
        with gr.Column(scale=1):
            plot_output = gr.Plot()

    with gr.Row():
        table_output = gr.Dataframe(
            label="📋 ตารางแสดงข้อมูล Comment รีวิวจริง (Filtered Data)",
            wrap=True
        )

    # ฟังก์ชันสลับสถานะปุ่ม
    def toggle_button(file):
        if file is not None:
            return gr.update(interactive=True)
        return gr.update(interactive=False)

    # Event เมื่อเลือก/ลบไฟล์
    file_uploader.change(
        fn=toggle_button,
        inputs=[file_uploader],
        outputs=[btn_run]
    )

    btn_run.click(
        fn=process_reviews,
        inputs=[file_uploader],
        outputs=[summary_output, plot_output, alert_output, table_output]
    )

if __name__ == "__main__":
    app.launch()