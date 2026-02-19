import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import re
import matplotlib as mpl
from wordcloud import WordCloud
from io import BytesIO
from docx import Document

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
    if pos_score > neg_score: return "บวก (Positive) 😊"
    elif neg_score > pos_score: return "ลบ (Negative) 😟"
    else: return "ปกติ / เป็นกลาง 😐"

# --- 3. ฟังก์ชันสร้างรายงาน MS Word ---
def create_word_report(filename, sentiment, summary, keywords_df, original_text):
    doc = Document()
    doc.add_heading(f'รายงานวิจัย: {filename}', 0)
    doc.add_heading('1. ผลวิเคราะห์อารมณ์', level=1)
    doc.add_paragraph(sentiment)
    doc.add_heading('2. สรุปเนื้อหาสำคัญ', level=1)
    for s in summary:
        doc.add_paragraph(s, style='List Bullet')
    doc.add_heading('3. คำสำคัญที่พบ (Top Keywords)', level=1)
    table = doc.add_table(rows=1, cols=2)
    hdr_cells = table.rows[0].cells
    hdr_cells[0].text = 'คำสำคัญ'
    hdr_cells[1].text = 'จำนวนครั้ง'
    for index, row in keywords_df.iterrows():
        row_cells = table.add_row().cells
        row_cells[0].text = str(row['คำ'])
        row_cells[1].text = str(row['จำนวนครั้ง'])
    doc.add_heading('4. ตัวอย่างข้อความต้นฉบับ', level=1)
    doc.add_paragraph(original_text[:1000] + "..." if len(original_text) > 1000 else original_text)
    bio = BytesIO()
    doc.save(bio)
    return bio.getvalue()

# --- 4. ฟังก์ชันสร้างไฟล์ Excel สำหรับตารางสรุป ---
def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='openpyxl')
    df.to_excel(writer, index=False, sheet_name='Sentiment_Summary')
    writer.close()
    processed_data = output.getvalue()
    return processed_data

# --- 5. การนำเข้า Library ---
try:
    from pythainlp.summarize import summarize
    from pythainlp.tokenize import word_tokenize
    from pythainlp.corpus import thai_stopwords
    THAI_READY = True
except ImportError:
    THAI_READY = False

st.set_page_config(layout="wide", page_title="Full Research Analysis Tool")
st.title("📂 ระบบวิเคราะห์งานวิจัยฉบับสมบูรณ์ (All-in-One)")

if not THAI_READY:
    st.error("❌ พบข้อผิดพลาดในการโหลด Library")
    st.stop()

setup_font()
uploaded_files = st.file_uploader("อัปโหลดไฟล์บทสัมภาษณ์ (.txt)", type=['txt'], accept_multiple_files=True)

if uploaded_files:
    comparison_list = [] 
    
    for file in uploaded_files:
        text = file.read().decode("utf-8")
        tokens = word_tokenize(text, keep_whitespace=False)
        stop_words = list(thai_stopwords())
        extra_stop = ['เนาะ', 'นะ', 'ครับ', 'ค่ะ', 'อืม', 'เอ่อ', 'คือ', 'แบบ']
        stop_words.extend(extra_stop)
        
        # กรองคำ: ยาว >= 5 และซ้ำ >= 3
        filtered_by_length = [t.strip() for t in tokens if t.strip() and t not in stop_words and len(t.strip()) >= 5 and not re.match(r'^[0-9\W]+$', t)]
        word_counts = Counter(filtered_by_length)
        filtered_final = [word for word in filtered_by_length if word_counts[word] >= 3]
        
        s_label = analyze_sentiment_thai(text)
        try:
            brief = summarize(text, n=2)
        except:
            brief = ["ไม่สามารถสรุปได้"]

        comparison_list.append({
            "ชื่อไฟล์": file.name,
            "โทนความรู้สึก": s_label,
            "คำสำคัญที่เด่นที่สุด": Counter(filtered_final).most_common(1)[0][0] if filtered_final else "ไม่พบ"
        })

        with st.expander(f"📑 ผลการวิเคราะห์ละเอียด: {file.name}", expanded=True):
            tab1, tab2 = st.tabs(["📊 สถิติและ Word Cloud", "📄 ข้อความต้นฉบับ"])
            
            with tab1:
                c1, c2 = st.columns(2)
                with c1:
                    st.subheader("💡 การวิเคราะห์เนื้อหา")
                    st.write(f"**ความรู้สึก:** {s_label}")
                    st.write("**สรุปประเด็น:**")
                    for b in brief: st.write(f"📌 {b}")
                    
                    if filtered_final:
                        wc = WordCloud(width=800, height=400, background_color="white", regexp=r"[\u0e00-\u0e7f]+", font_path=font_path).generate(" ".join(filtered_final))
                        fig, ax = plt.subplots()
                        ax.imshow(wc, interpolation='bilinear')
                        ax.axis("off")
                        st.pyplot(fig)
                        
                        buf = BytesIO()
                        fig.savefig(buf, format="png")
                        st.download_button(label="💾 โหลดรูป Word Cloud (PNG)", data=buf.getvalue(), file_name=f"cloud_{file.name}.png", mime="image/png")
                
                with c2:
                    st.subheader("📈 สถิติคำสำคัญ")
                    final_counts = Counter(filtered_final).most_common(12)
                    df_counts = pd.DataFrame(final_counts, columns=['คำ', 'จำนวนครั้ง'])
                    st.table(df_counts)
                    
                    word_report = create_word_report(file.name, s_label, brief, df_counts, text)
                    st.download_button(label="📄 โหลดรายงานสรุป (MS Word)", data=word_report, file_name=f"report_{file.name}.docx")

            with tab2:
                st.text_area(label="ข้อความต้นฉบับ", value=text, height=300)

    # --- ส่วนสุดท้าย: ตารางเปรียบเทียบรวมพร้อมปุ่มโหลด Excel ---
    st.divider()
    st.subheader("📋 ตารางสรุปเปรียบเทียบทุกเคส (Sentiment Analysis Summary)")
    df_compare = pd.DataFrame(comparison_list)
    st.table(df_compare)
    
    # ปุ่มดาวน์โหลด Excel
    excel_data = to_excel(df_compare)
    st.download_button(
        label="🟢 ดาวน์โหลดตารางสรุปเปรียบเทียบ (Excel)",
        data=excel_data,
        file_name="sentiment_summary_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

else:
    st.info("👋 กรุณาอัปโหลดไฟล์บทสัมภาษณ์เพื่อเริ่มการวิเคราะห์")
