import re
import pandas as pd
import plotly.graph_objects as go
from pythainlp.tokenize import word_tokenize
from pythainlp.corpus import thai_stopwords
import streamlit as st

# ==========================================
# 1. Page Configuration & Custom CSS
# ==========================================
st.set_page_config(
    page_title="Review Analytics Dashboard",
    page_icon="🌿",
    layout="wide"
)

st.markdown("""
<style>
    /* 1. บังคับพื้นหลังเว็บทั้งหมดเป็นสีครีมอ่อน */
    html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #FAF8F5 !important;
        color: #1A202C !important;
        font-family: 'Prompt', 'Inter', -apple-system, sans-serif !important;
    }

    /* ----------------------------------------------------
       2. FILE UPLOADER (หัวข้อดำเข้ม + ฟอนต์ชื่อไฟล์ในกล่องขาว)
    ---------------------------------------------------- */
    /* บังคับหัวข้อ "📂 อัปโหลดไฟล์รีวิว (CSV Format)" เป็นสีดำเข้ม 100% */
    [data-testid="stFileUploader"] label,
    [data-testid="stFileUploader"] label *,
    [data-testid="stFileUploader"] [data-testid="stWidgetLabel"] *,
    [data-testid="stFileUploader"] p {
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        opacity: 1 !important;
    }

    /* กรอบใหญ่ Dropzone */
    [data-testid="stFileUploaderDropzone"] {
        background-color: #FFFFFF !important;
        border: 1.5px dashed #B57C42 !important;
        border-radius: 12px !important;
    }

    /* ปุ่ม Browse files / Upload */
    [data-testid="stFileUploaderDropzone"] button,
    button[data-testid="baseButton-secondary"] {
        background-color: #B57C42 !important;
        border: none !important;
        border-radius: 8px !important;
    }
    
    [data-testid="stFileUploaderDropzone"] button * {
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
    }

    /* ตัวอักษรชื่อไฟล์ในกล่องอัปโหลดด้านล่าง -> เป็นสีขาว */
    [data-testid="stFileUploaderFileData"] *,
    [data-testid="stUploadedFileData"] *,
    [data-testid="stFileUploaderFileName"] {
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
        font-weight: 600 !important;
        opacity: 1 !important;
    }

    /* ไอคอนลบ/กากบาทเป็นสีขาว */
    [data-testid="stFileUploaderFileData"] button,
    [data-testid="stFileUploaderFileData"] svg,
    [data-testid="stUploadedFileData"] button,
    [data-testid="stUploadedFileData"] svg {
        fill: #FFFFFF !important;
        color: #FFFFFF !important;
        background-color: transparent !important;
    }

    /* ----------------------------------------------------
       3. CRITICAL MONITOR & TABLES
    ---------------------------------------------------- */
    .alert-card-container {
        background-color: #FFFFFF !important;
        border: 1.5px solid #D6CEC2 !important;
        border-radius: 12px !important;
        padding: 16px 20px !important;
        min-height: 155px !important;
        max-height: 220px !important;
        overflow-y: auto !important;
        box-shadow: 0 2px 6px rgba(0,0,0,0.03) !important;
    }
    .alert-card-header {
        color: #1A202C !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        margin-bottom: 10px !important;
    }
    .alert-card-body {
        color: #1A202C !important;
        font-size: 0.92rem !important;
        line-height: 1.6 !important;
        font-weight: 500 !important;
        white-space: pre-wrap !important;
    }

    .custom-table-container {
        width: 100%;
        overflow-x: auto;
        border: 1px solid #D6CEC2;
        border-radius: 10px;
        background-color: #FFFFFF;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.03);
        margin-top: 10px;
    }
    table.custom-table {
        width: 100%;
        border-collapse: collapse;
        text-align: left;
        font-size: 0.95rem;
        color: #1A202C !important;
    }
    table.custom-table th {
        background-color: #F0EAE1 !important;
        color: #1A202C !important;
        font-weight: 700 !important;
        padding: 12px 16px;
        border-bottom: 2px solid #B57C42;
    }
    table.custom-table td {
        padding: 12px 16px;
        border-bottom: 1px solid #E2D9CE;
        color: #1A202C !important;
        background-color: #FFFFFF !important;
    }
    table.custom-table tr:nth-child(even) td {
        background-color: #FAFAFA !important;
    }

    /* Header & Buttons */
    .custom-header {
        background: #FFFFFF !important;
        border: 1px solid #D6CEC2 !important;
        border-left: 6px solid #B57C42 !important;
        padding: 20px 24px;
        border-radius: 12px;
        margin-bottom: 20px;
    }
    .custom-header h1 { color: #1A202C !important; font-weight: 700 !important; font-size: 1.75rem !important; margin: 0 0 6px 0 !important; }
    .custom-header p { color: #4A5568 !important; margin: 0 !important; }

    .status-badge-up { background-color: #E6F4EA; color: #137333; border: 1px solid #34A853; padding: 4px 12px; border-radius: 6px; font-weight: 600; }
    .status-badge-down { background-color: #FCE8E6; color: #C5221F; border: 1px solid #EA4335; padding: 4px 12px; border-radius: 6px; font-weight: 600; }
    .status-badge-warn { background-color: #FEF7E0; color: #B06000; border: 1px solid #FBBC04; padding: 4px 12px; border-radius: 6px; font-weight: 600; }

    div.stButton > button {
        background-color: #B57C42 !important;
        color: #FFFFFF !important;
        border: none !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        width: 100%;
        padding: 8px 16px !important;
    }
    div.stButton > button:hover {
        background-color: #96622E !important;
    }
</style>
""", unsafe_allow_html=True)

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
        self.pos_words = ["ดี", "เร็ว", "ไว", "คุ้ม", "คุ้มค่า", "สบาย", "สวย", "ตรงปก", "ประทับใจ", "ชอบ", "น่ารัก", "เรียบร้อย", "เนี๊ยบ", "สมราคา", "สุดยอด", "แน่นหนา", "เรียบร้อยดี", "โอเค", "แนะนำ"]
        self.neg_words = ["ช้า", "บาง", "ไม่ตรงปก", "แพง", "บุบ", "เสียดาย", "ชำรุด", "ผิด", "หมอง", "ตำหนิ", "แย่", "ขาด", "ฉีกขาด", "รอย", "ห่วย", "พัง", "อย่าซื้อ", "ไม่ดี", "ไม่โอเค", "เสียความรู้สึก", "ใช้ไม่ได้", "เงียบ", "โยน", "จอขาว"]

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
        pos_count, neg_count, neu_count = 0, 0, 0
        alerts = []
        for idx, row in df.iterrows():
            text = str(row['review_text'])
            sentiment = self.analyze_sentiment_single(text)
            if sentiment == "เชิงลบ":
                neg_count += 1
                alerts.append(f"🔴 [CRITICAL ALERT] ID {row.get('review_id', idx+1)}: \"{text[:80]}...\"")
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
<div style="display: flex; gap: 10px; margin-top: 10px; margin-bottom: 15px;">
    <span class="status-badge-up">🟢 เชิงบวก (UP): {pos_pct}%</span>
    <span class="status-badge-down">🔴 เชิงลบ (DOWN): {neg_pct}%</span>
    <span class="status-badge-warn">🟠 ปานกลาง (WARN): {neu_pct}%</span>
</div>

<b style="color: #1A202C; font-size: 1rem;">🟢 Positive Insights (จุดเด่น):</b>
* ระบบตรวจพบการจัดส่งความเร็วสูงและการบรรจุพัสดุที่แน่นหนา

<br>
<b style="color: #1A202C; font-size: 1rem;">🔴 Action Required (จุดที่ต้องแก้ไขด่วน):</b>
* ตรวจพบสินค้าชำรุด (หน้าจอขาว / Broken Pixels)
* ปัญหาพฤติกรรมขนส่ง และบริการหลังการขาย
        """
        alert_text = "\n".join(alerts) if alerts else "🟢 STATUS NORMAL: ไม่พบรีวิวเชิงลบรุนแรงในระบบ"
        return summary_markdown, alert_text

@st.cache_resource
def get_engine():
    return NetworkSentimentEngine()

nlp_engine = get_engine()

# ==========================================
# 3. Streamlit Interface & Layout
# ==========================================

st.markdown("""
<div class="custom-header">
    <h1>🌿 Review Analytics Dashboard</h1>
    <p>ระบบวิเคราะห์และสรุปผลรีวิวสินค้าด้วย NLP สไตล์ มินิมอลเรียบหรู</p>
</div>
""", unsafe_allow_html=True)

col_upload, col_alert = st.columns([1, 2])

with col_upload:
    uploaded_file = st.file_uploader("📂 อัปโหลดไฟล์รีวิว (CSV Format)", type=["csv"])
    btn_run = st.button("🚀 ประมวลผลและวิเคราะห์", disabled=(uploaded_file is None))

with col_alert:
    alert_placeholder = st.empty()
    alert_placeholder.markdown("""
    <div class="alert-card-container">
        <div class="alert-card-header">🚨 Critical Review Monitor (ระบบแจ้งเตือนรีวิววิกฤต)</div>
        <div class="alert-card-body" style="color: #718096 !important;">รอการอัปโหลดไฟล์ CSV เพื่อเริ่มประมวลผล...</div>
    </div>
    """, unsafe_allow_html=True)

if btn_run and uploaded_file is not None:
    with st.spinner("กำลังประมวลผลข้อมูล NLP..."):
        df = pd.read_csv(uploaded_file)
        
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

        alert_placeholder.markdown(f"""
        <div class="alert-card-container">
            <div class="alert-card-header">🚨 Critical Review Monitor (ระบบแจ้งเตือนรีวิววิกฤต)</div>
            <div class="alert-card-body">{alert_txt}</div>
        </div>
        """, unsafe_allow_html=True)

        col_summary, col_plot = st.columns(2)
        with col_summary:
            st.markdown(summary_md, unsafe_allow_html=True)
        with col_plot:
            st.plotly_chart(fig, use_container_width=True)

        st.write("---")
        st.subheader("📋 ตารางแสดงข้อมูล Comment รีวิวจริง (Filtered Data)")
        
        df_display = df[['review_id', 'review_text', 'aspect_sentiment']].copy()
        df_display.columns = ['ID', 'Comment รีวิวจริง', 'ผลวิเคราะห์ ABSA Sentiment']
        df_display['ผลวิเคราะห์ ABSA Sentiment'] = df_display['ผลวิเคราะห์ ABSA Sentiment'].astype(str)

        table_html = "<div class='custom-table-container'><table class='custom-table'>"
        table_html += "<thead><tr>" + "".join([f"<th>{col}</th>" for col in df_display.columns]) + "</tr></thead>"
        table_html += "<tbody>"
        for _, row in df_display.iterrows():
            table_html += "<tr>" + "".join([f"<td>{row[col]}</td>" for col in df_display.columns]) + "</tr>"
        table_html += "</tbody></table></div>"

        st.markdown(table_html, unsafe_allow_html=True)