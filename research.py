import streamlit as st
from pythainlp.summarize import summarize
from pythainlp.tokenize import word_tokenize
from pythainlp.corpus import thai_stopwords
from pythainlp.sentiment import sentiment
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
import re

# ตั้งค่าหน้าจอ
st.set_page_config(layout="wide", page_title="Advanced Thai Research Tool")

# ส่วนหัวโปรแกรม
st.title("📂 ระบบวิเคราะห์บทสัมภาษณ์งานวิจัย (Advanced Version)")
st.markdown("---")

# ส่วนอัปโหลดไฟล์ (รองรับหลายไฟล์)
uploaded_files = st.file_uploader("เลือกไฟล์บทสัมภาษณ์ (.txt)", type=['txt'], accept_multiple_files=True)

if uploaded_files:
    # สำหรับเก็บข้อมูลสรุปเพื่อทำตารางเปรียบเทียบตอนท้าย
    summary_data = []

    for file in uploaded_files:
        # อ่านข้อความจากไฟล์
        text = file.read().decode("utf-8")
        
        # เตรียมข้อมูลสำหรับการแสดงผลรายไฟล์
        with st.expander(f"📋 ผลการวิเคราะห์ไฟล์: {file.name}", expanded=True):
            col1, col2 = st.columns([1, 1])
            
            # การประมวลผลคำ (Tokenization)
            tokens = word_tokenize(text, keep_whitespace=False)
            stopwords = list(thai_stopwords())
            filtered_words = [t for t in tokens if t not in stopwords and len(t) > 1 and not re.match(r'[0-9]+', t)]
            
            with col1:
                st.subheader("🔍 ข้อมูลเบื้องต้น")
                
                # 1. วิเคราะห์อารมณ์ (Sentiment)
                try:
                    sent_res = sentiment(text)
                    if sent_res == "pos":
                        sent_display = "บวก (Positive) 😊"
                    elif sent_res == "neg":
                        sent_display = "ลบ (Negative) 😟"
                    else:
                        sent_display = "ปกติ (Neutral) 😐"
                except:
                    sent_display = "ไม่สามารถวิเคราะห์ได้"
                
                st.write(f"**โทนความรู้สึก:** {sent_display}")
                
                # 2. สรุปใจความสำคัญ (Summary)
                st.write("**สรุปเนื้อหา:**")
                try:
                    brief = summarize(text, n=2)
                    for b in brief:
                        st.write(f"📌 {b}")
                except:
                    st.write("- ไม่สามารถสรุปได้เนื่องจากเนื้อหาสั้นเกินไป")

                # 3. สร้าง Word Cloud
                st.write("**Word Cloud:**")
                try:
                    # ใช้ RegExp เพื่อให้รองรับภาษาไทย
                    wc = WordCloud(
                        width=800, 
                        height=400, 
                        background_color="white", 
                        regexp=r"[\u0e00-\u0e7f]+",
                        colormap='viridis'
                    ).generate(" ".join(filtered_words))
                    
                    fig, ax = plt.subplots()
                    ax.imshow(wc, interpolation='bilinear')
                    ax.axis("off")
                    st.pyplot(fig)
                except:
                    st.error("ไม่สามารถสร้าง Word Cloud ได้")

            with col2:
                st.subheader("📊 สถิติคำสำคัญ")
                # นับคำสำคัญ
                word_counts = Counter(filtered_words).most_common(12)
                df_words = pd.DataFrame(word_counts, columns=['คำ', 'จำนวนครั้ง'])
                
                # แสดงเป็นกราฟแท่ง
                st.bar_chart(df_words.set_index('คำ'))
                
                # แสดงเป็นตาราง
                st.table(df_words)

        # เก็บข้อมูลไว้ทำตารางเปรียบเทียบล่างสุด
        summary_data.append({
            "ชื่อไฟล์": file.name,
            "ความรู้สึก": sent_display,
            "คำที่พบบ่อย": ", ".join([w[0] for w in word_counts[:3]])
        })

    # ส่วนตารางเปรียบเทียบรวม (Cross-Case Table)
    st.markdown("---")
    st.subheader("📑 ตารางสรุปการเปรียบเทียบระหว่างเคส (Case Comparison Table)")
    final_df = pd.DataFrame(summary_data)
    st.dataframe(final_df, use_container_width=True)

else:
    # หน้าจอตอนยังไม่ได้อัปโหลด
    st.info("กรุณาอัปโหลดไฟล์บทสัมภาษณ์ตั้งแต่ 1 ไฟล์ขึ้นไปทางแถบซ้ายมือ หรือคลิก Browse files")
    
    # คำแนะนำเล็กน้อย
    with st.expander("💡 คำแนะนำการใช้งาน"):
        st.write("1. เตรียมไฟล์บทสัมภาษณ์เป็นนามสกุล .txt")
        st.write("2. สามารถลากไฟล์หลายๆ ไฟล์มาวางพร้อมกันได้")
        st.write("3. คุณสามารถบันทึกกราฟหรือ Word Cloud ได้โดยการคลิกขวาที่รูปแล้วเลือก 'Save Image'")
