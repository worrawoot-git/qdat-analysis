import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import re
import matplotlib as mpl
from wordcloud import WordCloud
from io import BytesIO
from docx import Document # ต้องติดตั้ง python-docx

# --- 1. ตั้งค่าการแสดงผลภาษาไทย ---
font_path = "Kanit-Regular.ttf" 

def setup_font():
    try:
        mpl.font_manager.fontManager.addfont(font_path)
        prop = mpl.font_manager.FontProperties(fname=font_path)
        mpl.rc('font', family=prop.get_name())
        mpl.rcParams['axes.unicode_minus'] = False 
        return True
    except:
        return False

# --- 2. ฟังก์ชันวิเคราะห์อารมณ์แบบผสมผสาน ---
def analyze_sentiment_thai(text):
    pos_words = ['ดี', 'เห็นด้วย', 'ภูมิใจ', 'สำเร็จ', 'ความสุข', 'พัฒนา', 'ประโยชน์', 'ยั่งยืน', 'พอเพียง', 'ประหยัด']
    neg_words = ['ไม่ดี', 'ปัญหา', 'แย่', 'ยากลำบาก', 'ขาดแคลน', 'อุปสรรค', 'หนี้สิน', 'เดือดร้อน', 'เสียดาย']
    pos_score = sum(1 for w in pos_words if w in text)
    neg_score = sum(1 for w in neg_words if w in text)
    if pos_score > neg_score: return "ค่อนไปทางบวก 😊"
    elif neg_score > pos_score: return "ค่อนไปทางลบ 😟"
    else: return "ปกติ / เป็นกลาง 😐"

# --- 3. ฟังก์ชันสร้างไฟล์ MS Word ---
def create_word_report(filename, sentiment, summary, keywords_df):
    doc = Document()
    doc.add_heading(f'รายงานการวิเคราะห์: {filename}', 0)
    
    doc.add_heading('ผลวิเคราะห์อารมณ์', level=1)
    doc.add_paragraph(sentiment)
    
    doc.add_heading('สรุปเนื้อหาสำคัญ', level=1)
    for s in summary:
        doc.add_paragraph(s, style='List Bullet')
        
    doc.add_heading('สถิติคำสำคัญ (Top Keywords)', level=1)
    table = doc.add_table(rows=1, cols=2)
    hdr_cells = table.rows[0].cells
    hdr_cells[0].text = 'คำสำคัญ'
    hdr_cells[1].text = 'จำนวนครั้ง'
    for index, row in keywords_df.iterrows():
        row_cells = table.add_row().cells
        row_cells[0].text = str(row['คำ'])
        row_cells[1].text = str(row['จำนวนครั้ง'])
        
    bio = BytesIO()
    doc.save(bio)
    return bio.getvalue()

# --- 4. การนำเข้า Library ภาษาไทย ---
try:
    from pythainlp.summarize import summarize
    from pythainlp.tokenize import word_tokenize
    from pythainlp.corpus import thai_stopwords
    THAI_READY = True
except ImportError:
    THAI_READY = False

st.set_page_config(layout="wide", page_title="Research Tool with Download")
st.title("📂 ระบบวิเคราะห์งานวิจัย (Export Edition)")

if not THAI_READY:
    st.error("❌ พบข้อผิดพลาดในการโหลด Library")
    st.stop()

setup_font()
uploaded_files = st.file_uploader("อัปโหลดไฟล์บทสัมภาษณ์ (.txt)", type=['txt'], accept_multiple_files=True)

if uploaded_files:
    for file in uploaded_files:
        text = file.read().decode("utf-8")
        tokens = word_tokenize(text, keep_whitespace=False)
        stop_words = list(thai_stopwords())
        
        # กรองคำ: ยาว >= 5 และไม่ใช่สัญลักษณ์
        filtered_by_length = [t.strip() for t in tokens if t.strip() and t not in stop_words and len(t.strip()) >= 5 and not re.match(r'^[0-9\W]+$', t)]
        word_counts = Counter(filtered_by_length)
        filtered_final = [word for word in filtered_by_length if word_counts[word] >= 3]
        
        with st.expander(f"📊 ผลการวิเคราะห์: {file.name}", expanded=True):
            col1, col2 = st.columns(2)
            
            s_label = analyze_sentiment_thai(text)
            try:
                brief = summarize(text, n=2)
            except:
                brief = ["ไม่สามารถสรุปได้"]

            with col1:
                st.subheader("🔍 ผลการวิเคราะห์")
                st.write(f"**อารมณ์:** {s_label}")
                
                # --- ส่วน Word Cloud และปุ่มดาวน์โหลดรูป ---
                if filtered_final:
                    wc = WordCloud(width=800, height=400, background_color="white", regexp=r"[\u0e00-\u0e7f]+", font_path=font_path).generate(" ".join(filtered_final))
                    fig, ax = plt.subplots()
                    ax.imshow(wc, interpolation='bilinear')
                    ax.axis("off")
                    st.pyplot(fig)
                    
                    # ปุ่มดาวน์โหลด PNG
                    buf = BytesIO()
                    fig.savefig(buf, format="png")
                    st.download_button(label="💾 ดาวน์โหลด Word Cloud (PNG)", data=buf.getvalue(), file_name=f"wordcloud_{file.name}.png", mime="image/png")
                
            with col2:
                st.subheader("📈 สถิติคำ")
                final_counts = Counter(filtered_final).most_common(12)
                df_counts = pd.DataFrame(final_counts, columns=['คำ', 'จำนวนครั้ง'])
                st.table(df_counts)
                
                # --- ปุ่มดาวน์โหลดรายงาน Word ---
                word_data = create_word_report(file.name, s_label, brief, df_counts)
                st.download_button(label="📄 ดาวน์โหลดรายงาน (MS Word)", data=word_data, file_name=f"report_{file.name}.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")

else:
    st.info("กรุณาอัปโหลดไฟล์")
