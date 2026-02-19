import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import re
import matplotlib as mpl
from wordcloud import WordCloud

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
    try:
        from pythainlp.sentiment import sentiment
        res = sentiment(text)
        if res == "pos": return "บวก (Positive) 😊"
        if res == "neg": return "ลบ (Negative) 😟"
    except:
        pass

    pos_words = ['ดี', 'เห็นด้วย', 'ภูมิใจ', 'สำเร็จ', 'ความสุข', 'พัฒนา', 'ประโยชน์', 'ยั่งยืน', 'พอเพียง', 'ประหยัด']
    neg_words = ['ไม่ดี', 'ปัญหา', 'แย่', 'ยากลำบาก', 'ขาดแคลน', 'อุปสรรค', 'หนี้สิน', 'เดือดร้อน', 'เสียดาย']
    
    pos_score = sum(1 for w in pos_words if w in text)
    neg_score = sum(1 for w in neg_words if w in text)
    
    if pos_score > neg_score: return "ค่อนไปทางบวก 😊"
    elif neg_score > pos_score: return "ค่อนไปทางลบ 😟"
    else: return "ปกติ / เป็นกลาง 😐"

# --- 3. การนำเข้า Library ---
try:
    from pythainlp.summarize import summarize
    from pythainlp.tokenize import word_tokenize
    from pythainlp.corpus import thai_stopwords
    THAI_READY = True
except ImportError:
    THAI_READY = False

st.set_page_config(layout="wide", page_title="Professional Thai Research Tool")
st.title("📂 ระบบวิเคราะห์บทสัมภาษณ์งานวิจัย (Custom Filter Edition)")

if not THAI_READY:
    st.error("❌ พบข้อผิดพลาดในการโหลด Library ภาษาไทย")
    st.stop()

setup_font()

uploaded_files = st.file_uploader("อัปโหลดไฟล์บทสัมภาษณ์ (.txt)", type=['txt'], accept_multiple_files=True)

if uploaded_files:
    comparison_data = []
    
    for file in uploaded_files:
        text = file.read().decode("utf-8")
        
        # ตัดคำ
        tokens = word_tokenize(text, keep_whitespace=False)
        stop_words = list(thai_stopwords())
        extra_stop = ['เนาะ', 'นะ', 'ครับ', 'ค่ะ', 'อืม', 'เอ่อ']
        stop_words.extend(extra_stop)
        
        # --- เงื่อนไขใหม่: เลือกคำที่ยาวตั้งแต่ 5 ตัวอักษรขึ้นไป และไม่ใช่ตัวเลข/สัญลักษณ์ ---
        filtered_by_length = [
            t.strip() for t in tokens 
            if t.strip() and t not in stop_words and len(t.strip()) >= 5 and not re.match(r'^[0-9\W]+$', t)
        ]
        
        # --- เงื่อนไขใหม่: นับความถี่และเลือกคำที่ซ้ำตั้งแต่ 3 ครั้งขึ้นไป ---
        word_counts_full = Counter(filtered_by_length)
        filtered_final = [word for word in filtered_by_length if word_counts_full[word] >= 3]
        
        with st.expander(f"📊 ผลการวิเคราะห์: {file.name}", expanded=True):
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("🔍 สรุปข้อมูล")
                st.write(f"**โทนความรู้สึกรวม:** {analyze_sentiment_thai(text)}")
                
                try:
                    brief = summarize(text, n=2)
                    st.write("**สรุปเนื้อหา:**")
                    for b in brief: st.write(f"📌 {b}")
                except:
                    st.write("**สรุปเนื้อหา:** ไม่สามารถสรุปได้")

                st.write("**Word Cloud (ยาว >= 5 ตัวอักษร และ ซ้ำ >= 3 ครั้ง):**")
                if filtered_final:
                    try:
                        wc = WordCloud(
                            width=800, height=400, 
                            background_color="white", 
                            regexp=r"[\u0e00-\u0e7f]+",
                            font_path=font_path
                        ).generate(" ".join(filtered_final))
                        
                        fig_wc, ax_wc = plt.subplots()
                        ax_wc.imshow(wc, interpolation='bilinear')
                        ax_wc.axis("off")
                        st.pyplot(fig_wc)
                    except:
                        st.write("⚠️ ไม่สามารถสร้าง Word Cloud ได้")
                else:
                    st.warning("⚠️ ไม่พบคำที่ตรงตามเงื่อนไข (ยาว >= 5 และ ซ้ำ >= 3)")

            with col2:
                st.subheader("📈 สถิติคำสำคัญ")
                # แสดงผลคำที่ซ้ำสูงสุด 12 อันดับแรกจากรายการที่กรองแล้ว
                final_counts = Counter(filtered_final).most_common(12)
                if final_counts:
                    df_counts = pd.DataFrame(final_counts, columns=['คำ', 'จำนวนครั้ง'])
                    st.bar_chart(df_counts.set_index('คำ'))
                    st.table(df_counts)
                else:
                    st.write("ไม่พบข้อมูลที่ตรงตามเงื่อนไข")

            comparison_data.append({"ไฟล์": file.name, "ความรู้สึก": analyze_sentiment_thai(text)})

    st.divider()
    st.subheader("📋 ตารางสรุปเปรียบเทียบระหว่างเคส")
    st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)
else:
    st.info("กรุณาอัปโหลดไฟล์เพื่อเริ่มการวิเคราะห์")
