import streamlit as st
from pythainlp.summarize import summarize
from pythainlp.tokenize import word_tokenize
from pythainlp.corpus import thai_stopwords
import pandas as pd
from collections import Counter

st.set_page_config(layout="wide", page_title="Thai Research Tool")
st.title("📂 เครื่องมือวิเคราะห์บทสัมภาษณ์ภาษาไทย")
st.write("ระบบวิเคราะห์เนื้อหาและคำสำคัญสำหรับงานวิจัยภาคสนาม")

uploaded_file = st.file_uploader("เลือกไฟล์บทสัมภาษณ์ภาษาไทย (.txt)", type=['txt'])

if uploaded_file:
    # อ่านไฟล์
    text = uploaded_file.read().decode("utf-8")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📄 ข้อความต้นฉบับ")
        st.text_area("Original Content", text, height=500)
        
    with col2:
        st.subheader("🔍 ผลการสกัดใจความสำคัญ")
        
        # 1. สรุปประโยคหลัก
        st.info("**สรุปประเด็นหลัก (AI Summary):**")
        try:
            summary = summarize(text, n=3)
            for s in summary:
                st.write(f"📌 {s}")
        except:
            st.write("ไม่สามารถสรุปความได้ในขณะนี้")
            
        st.divider()
        
        # 2. วิเคราะห์คำที่พบบ่อย (แทนที่ระบบ extract ที่ error)
        st.info("**คำที่พบบ่อยที่สุด (Keywords/Top Terms):**")
        tokens = word_tokenize(text, keep_whitespace=False)
        stopwords = list(thai_stopwords())
        # กรองคำฟุ่มเฟือยและคำสั้นเกินไปออก
        filtered_words = [t for t in tokens if t not in stopwords and len(t) > 1]
        
        # นับจำนวนคำ
        word_counts = Counter(filtered_words).most_common(10)
        
        for word, count in word_counts:
            st.write(f"🔑 `{word}` (พบ {count} ครั้ง)")

        # แสดงเป็นกราฟให้ดูง่ายๆ
        df = pd.DataFrame(word_counts, columns=['Word', 'Count'])
        st.bar_chart(df.set_index('Word'))