import streamlit as st
from pythainlp.summarize import summarize
from pythainlp.tokenize import word_tokenize
from pythainlp.corpus import thai_stopwords
from pythainlp.sentiment import sentiment
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter

st.set_page_config(layout="wide", page_title="Advanced Thai Research Tool")

st.title("📂 ระบบวิเคราะห์งานวิจัยภาคสนาม (Advanced Version)")
st.write("ฟีเจอร์: Word Cloud | Sentiment | Multi-File Analysis")

# 1. Multiple File Support: รองรับหลายไฟล์พร้อมกัน
uploaded_files = st.file_uploader("อัปโหลดไฟล์บทสัมภาษณ์ (.txt)", type=['txt'], accept_multiple_files=True)

if uploaded_files:
    # สร้างลิสต์เพื่อเก็บข้อมูลสำหรับเปรียบเทียบ
    all_data = []

    for file in uploaded_files:
        text = file.read().decode("utf-8")
        tokens = word_tokenize(text, keep_whitespace=False)
        stopwords = list(thai_stopwords())
        filtered_words = [t for t in tokens if t not in stopwords and len(t) > 1]
        
        # 2. Sentiment Analysis: วิเคราะห์อารมณ์ (บวก/ลบ)
        # เราจะสุ่มเช็กประโยคสำคัญเพื่อดูทิศทางอารมณ์รวม
        sent_result = sentiment(text) 
        sent_label = "บวก (Positive)" if sent_result == "pos" else "ลบ (Negative)" if sent_result == "neg" else "ปกติ (Neutral)"
        
        all_data.append({
            "filename": file.name,
            "text": text,
            "keywords": Counter(filtered_words).most_common(10),
            "sentiment": sent_label
        })

    # ส่วนการแสดงผล
    for data in all_data:
        with st.expander(f"📊 ผลการวิเคราะห์ไฟล์: {data['filename']}"):
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.write("**ความรู้สึกโดยรวม:**", data['sentiment'])
                st.write("**สรุปเนื้อหา:**")
                try:
                    summ = summarize(data['text'], n=2)
                    for s in summ: st.write(f"- {s}")
                except: st.write("ไม่สามารถสรุปได้")

                # 3. Word Cloud: สร้างภาพกลุ่มคำ
                st.write("**Word Cloud:**")
                # สำหรับภาษาไทย ต้องใช้ Font ที่รองรับ (ระบบ Streamlit Cloud มักมี font มาตรฐานให้)
                wc = WordCloud(font_path=None, width=800, height=400, background_color="white", regexp=r"[\u0e00-\u0e7f]+").generate(" ".join(filtered_words))
                fig, ax = plt.subplots()
                ax.imshow(wc, interpolation='bilinear')
                ax.axis("off")
                st.pyplot(fig)

            with col2:
                st.write("**คำสำคัญที่พบบ่อย:**")
                df = pd.DataFrame(data['keywords'], columns=['คำ', 'จำนวน'])
                st.bar_chart(df.set_index('คำ'))

    # ตารางเปรียบเทียบระหว่างเคส
    st.divider()
    st.subheader("📑 ตารางเปรียบเทียบระหว่างเคส (Cross-Case Comparison)")
    compare_df = pd.DataFrame([{"ชื่อไฟล์": d['filename'], "อารมณ์": d['sentiment']} for d in all_data])
    st.table(compare_df)

else:
    st.info("กรุณาอัปโหลดไฟล์บทสัมภาษณ์ตั้งแต่ 1 ไฟล์ขึ้นไปเพื่อเริ่มการวิเคราะห์")
