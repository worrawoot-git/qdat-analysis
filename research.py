import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import re

# พยายามนำเข้า Library ทีละตัวเพื่อความปลอดภัย
try:
    import pythainlp
    from pythainlp.tokenize import word_tokenize
    from pythainlp.corpus import thai_stopwords
    THAI_LIB = True
except ImportError:
    THAI_LIB = False

st.set_page_config(layout="wide", page_title="Stable Thai Research Tool")
st.title("📂 ระบบวิเคราะห์บทสัมภาษณ์ (Stable Cloud Version)")

if not THAI_LIB:
    st.error("❌ ระบบไม่สามารถติดตั้ง Library ภาษาไทยได้ กรุณากด Reboot App หรือเช็กไฟล์ requirements.txt")
    st.stop()

uploaded_files = st.file_uploader("อัปโหลดไฟล์บทสัมภาษณ์ (.txt)", type=['txt'], accept_multiple_files=True)

if uploaded_files:
    summary_list = []
    for file in uploaded_files:
        text = file.read().decode("utf-8")
        
        with st.expander(f"📑 วิเคราะห์ไฟล์: {file.name}", expanded=True):
            col1, col2 = st.columns(2)
            
            # การตัดคำ
            tokens = word_tokenize(text, keep_whitespace=False)
            stop_words = list(thai_stopwords())
            filtered = [t for t in tokens if t not in stop_words and len(t) > 1 and not re.match(r'[0-9]+', t)]
            
            with col1:
                st.subheader("💡 ผลการวิเคราะห์")
                
                # 1. วิเคราะห์ Sentiment (แบบดัก Error รายบรรทัด)
                try:
                    from pythainlp.sentiment import sentiment
                    s_val = sentiment(text)
                    s_label = "บวก 😊" if s_val == "pos" else "ลบ 😟" if s_val == "neg" else "ปกติ 😐"
                except:
                    s_label = "ไม่รองรับการวิเคราะห์อารมณ์"
                st.write(f"**โทนความรู้สึกรวม:** {s_label}")

                # 2. สรุปใจความ (แบบดัก Error)
                try:
                    from pythainlp.summarize import summarize
                    brief = summarize(text, n=2)
                    st.write("**สรุปเนื้อหา:**")
                    for b in brief: st.write(f"📌 {b}")
                except:
                    st.write("**สรุปเนื้อหา:** ไม่สามารถสรุปได้อัตโนมัติ")

                # 3. Word Cloud
                try:
                    from wordcloud import WordCloud
                    wc = WordCloud(width=800, height=400, background_color="white", regexp=r"[\u0e00-\u0e7f]+").generate(" ".join(filtered))
                    fig, ax = plt.subplots()
                    ax.imshow(wc, interpolation='bilinear')
                    ax.axis("off")
                    st.pyplot(fig)
                except:
                    st.write("⚠️ ไม่สามารถสร้าง Word Cloud ได้")

            with col2:
                st.subheader("📊 สถิติคำสำคัญ")
                counts = Counter(filtered).most_common(12)
                df = pd.DataFrame(counts, columns=['คำ', 'จำนวน'])
                if not df.empty:
                    st.bar_chart(df.set_index('คำ'))
                    st.table(df)

            summary_list.append({"ชื่อไฟล์": file.name, "อารมณ์": s_label})

    st.divider()
    st.subheader("📋 ตารางเปรียบเทียบเคส")
    st.dataframe(pd.DataFrame(summary_list), use_container_width=True)
else:
    st.info("กรุณาอัปโหลดไฟล์บทสัมภาษณ์เพื่อเริ่มการประมวลผล")
