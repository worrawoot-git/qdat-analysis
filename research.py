import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import re

# นำเข้า Library ภาษาไทยแบบปลอดภัย
try:
    from pythainlp.summarize import summarize
    from pythainlp.tokenize import word_tokenize
    from pythainlp.corpus import thai_stopwords
    from pythainlp import sentiment
    THAI_READY = True
except ImportError:
    THAI_READY = False

from wordcloud import WordCloud

st.set_page_config(layout="wide", page_title="Thai Research Tool")

st.title("📂 ระบบวิเคราะห์บทสัมภาษณ์ (Stable Version)")
st.markdown("---")

if not THAI_READY:
    st.error("❌ พบข้อผิดพลาดในการติดตั้ง Library ภาษาไทย กรุณาเช็กไฟล์ requirements.txt")
    st.stop()

uploaded_files = st.file_uploader("อัปโหลดไฟล์บทสัมภาษณ์ (.txt)", type=['txt'], accept_multiple_files=True)

if uploaded_files:
    summary_list = []
    for file in uploaded_files:
        text = file.read().decode("utf-8")
        
        with st.expander(f"📑 ไฟล์: {file.name}", expanded=True):
            col1, col2 = st.columns(2)
            
            # การตัดคำและจัดการ Stopwords
            tokens = word_tokenize(text, keep_whitespace=False)
            stop_words = list(thai_stopwords())
            filtered = [t for t in tokens if t not in stop_words and len(t) > 1 and not re.match(r'[0-9]+', t)]
            
            with col1:
                # วิเคราะห์อารมณ์แบบป้องกันการล่ม
                st.subheader("💡 การวิเคราะห์ใจความ")
                try:
                    s_val = sentiment(text)
                    s_label = "บวก 😊" if s_val == "pos" else "ลบ 😟" if s_val == "neg" else "ปกติ 😐"
                except:
                    s_label = "ไม่สามารถวิเคราะห์ได้"
                
                st.write(f"**ความรู้สึกรวม:** {s_label}")
                
                st.write("**สรุปเนื้อหา:**")
                try:
                    brief = summarize(text, n=2)
                    for b in brief: st.write(f"📌 {b}")
                except: st.write("- เนื้อหาสั้นเกินไปสำหรับการสรุป")

                st.write("**Word Cloud:**")
                try:
                    wc = WordCloud(width=800, height=400, background_color="white", regexp=r"[\u0e00-\u0e7f]+").generate(" ".join(filtered))
                    fig, ax = plt.subplots()
                    ax.imshow(wc)
                    ax.axis("off")
                    st.pyplot(fig)
                except: st.write("⚠️ ไม่สามารถสร้าง Word Cloud ได้")

            with col2:
                st.subheader("📊 สถิติคำ")
                counts = Counter(filtered).most_common(12)
                df = pd.DataFrame(counts, columns=['คำ', 'จำนวน'])
                st.bar_chart(df.set_index('คำ'))
                st.table(df)

            summary_list.append({"ไฟล์": file.name, "ความรู้สึก": s_label})

    st.divider()
    st.subheader("📋 ตารางเปรียบเทียบเคส")
    st.dataframe(pd.DataFrame(summary_list), use_container_width=True)
else:
    st.info("กรุณาอัปโหลดไฟล์ที่แถบด้านบน")
