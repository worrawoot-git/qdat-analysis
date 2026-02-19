import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import re
import matplotlib as mpl
from wordcloud import WordCloud

# --- 1. ตั้งค่าการแสดงผลภาษาไทย (Font Configuration) ---
# ชื่อไฟล์ต้องตรงกับที่อัปโหลดขึ้น GitHub (Kanit-Regular.ttf)
font_path = "Kanit-Regular.ttf" 

try:
    # เพิ่มฟอนต์เข้าไปใน Matplotlib เพื่อให้กราฟอ่านภาษาไทยออก
    mpl.font_manager.fontManager.addfont(font_path)
    prop = mpl.font_manager.FontProperties(fname=font_path)
    mpl.rc('font', family=prop.get_name())
    # ป้องกันปัญหาสัญลักษณ์เครื่องหมายลบแสดงผลผิดพลาด
    mpl.rcParams['axes.unicode_minus'] = False 
except Exception as e:
    st.warning(f"⚠️ คำเตือน: ระบบหาไฟล์ฟอนต์ {font_path} ไม่พบ กราฟอาจแสดงผลภาษาไทยไม่ได้")

# --- 2. ฟังก์ชันหลักสำหรับวิเคราะห์ภาษาไทย ---
try:
    from pythainlp.summarize import summarize
    from pythainlp.tokenize import word_tokenize
    from pythainlp.corpus import thai_stopwords
    from pythainlp.sentiment import sentiment
    THAI_READY = True
except ImportError:
    THAI_READY = False

# --- 3. ส่วนการแสดงผลบนหน้าเว็บ (UI) ---
st.set_page_config(layout="wide", page_title="Advanced Thai Research Tool")

st.title("📂 ระบบวิเคราะห์บทสัมภาษณ์งานวิจัย (Full Version)")
st.write("ฟีเจอร์: ภาษาไทยสมบูรณ์ | Word Cloud | วิเคราะห์อารมณ์ | หลายไฟล์")

if not THAI_READY:
    st.error("❌ พบข้อผิดพลาด: ไม่สามารถติดตั้ง Library ภาษาไทยได้ กรุณาเช็กไฟล์ requirements.txt")
    st.stop()

# ส่วนอัปโหลดไฟล์
uploaded_files = st.file_uploader("อัปโหลดไฟล์บทสัมภาษณ์ (.txt)", type=['txt'], accept_multiple_files=True)

if uploaded_files:
    comparison_list = []
    
    for file in uploaded_files:
        # อ่านข้อความจากไฟล์
        text = file.read().decode("utf-8")
        
        # เตรียมข้อมูลเบื้องต้น
        tokens = word_tokenize(text, keep_whitespace=False)
        stop_words = list(thai_stopwords())
        # กรองคำที่ไม่จำเป็นและตัวเลขออก
        filtered = [t for t in tokens if t not in stop_words and len(t) > 1 and not re.match(r'[0-9]+', t)]
        
        with st.expander(f"📊 วิเคราะห์ไฟล์: {file.name}", expanded=True):
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("💡 สรุปและอารมณ์")
                
                # วิเคราะห์อารมณ์ (Sentiment)
                try:
                    s_val = sentiment(text)
                    s_label = "บวก (Positive) 😊" if s_val == "pos" else "ลบ (Negative) 😟" if s_val == "neg" else "ปกติ (Neutral) 😐"
                except:
                    s_label = "ไม่รองรับการวิเคราะห์อารมณ์"
                st.write(f"**โทนความรู้สึกรวม:** {s_label}")
                
                # สรุปใจความสำคัญ (Summary)
                st.write("**สรุปเนื้อหา:**")
                try:
                    brief = summarize(text, n=2)
                    for b in brief: st.write(f"📌 {b}")
                except:
                    st.write("- ไม่สามารถสรุปเนื้อหาได้อัตโนมัติ")

                # สร้าง Word Cloud ภาษาไทย
                st.write("**Word Cloud:**")
                try:
                    wc = WordCloud(
                        width=800, 
                        height=400, 
                        background_color="white", 
                        regexp=r"[\u0e00-\u0e7f]+", # ดึงเฉพาะตัวอักษรไทย
                        font_path=font_path
                    ).generate(" ".join(filtered))
                    
                    fig_wc, ax_wc = plt.subplots()
                    ax_wc.imshow(wc, interpolation='bilinear')
                    ax_wc.axis("off")
                    st.pyplot(fig_wc)
                except Exception as e:
                    st.write(f"⚠️ ไม่สามารถสร้าง Word Cloud ได้: {e}")

            with col2:
                st.subheader("📈 สถิติคำสำคัญ")
                # นับคำสำคัญ 12 อันดับแรก
                counts = Counter(filtered).most_common(12)
                df_counts = pd.DataFrame(counts, columns=['คำ', 'จำนวนครั้ง'])
                
                # แสดงกราฟแท่ง (จะแสดงภาษาไทยได้เพราะตั้งค่า mpl.rc ไว้ด้านบน)
                st.bar_chart(df_counts.set_index('คำ'))
                
                # แสดงตารางข้อมูล
                st.table(df_counts)

            comparison_list.append({"ชื่อไฟล์": file.name, "อารมณ์": s_label, "คำสำคัญหลัก": df_counts['คำ'].iloc[0] if not df_counts.empty else "-"})

    # ส่วนตารางเปรียบเทียบระหว่างเคส
    st.divider()
    st.subheader("📋 ตารางสรุปเปรียบเทียบ (Cross-Case Comparison)")
    st.dataframe(pd.DataFrame(comparison_list), use_container_width=True)

else:
    st.info("👋 ยินดีต้อนรับ! กรุณาอัปโหลดไฟล์ .txt เพื่อเริ่มการวิเคราะห์")
