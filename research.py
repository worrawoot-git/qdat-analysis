import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import re
import matplotlib as mpl
from wordcloud import WordCloud
from io import BytesIO
from docx import Document
import networkx as nx 

# --- 1. ตั้งค่าการแสดงผลภาษาไทย ---
font_path = "Kanit-Regular.ttf" 

def setup_font():
    try:
        mpl.font_manager.fontManager.addfont(font_path)
        prop = mpl.font_manager.FontProperties(fname=font_path)
        mpl.rc('font', family=prop.get_name(), size=12)
        mpl.rcParams['axes.unicode_minus'] = False 
        return prop
    except:
        return None

# --- 2. ฟังก์ชันวิเคราะห์อารมณ์ ---
def analyze_sentiment_thai(text):
    pos_words = ['ดี', 'เห็นด้วย', 'ภูมิใจ', 'สำเร็จ', 'ความสุข', 'พัฒนา', 'ประโยชน์', 'ยั่งยืน', 'พอเพียง']
    neg_words = ['ไม่ดี', 'ปัญหา', 'แย่', 'ยากลำบาก', 'ขาดแคลน', 'อุปสรรค', 'หนี้สิน', 'เดือดร้อน']
    pos_score = sum(1 for w in pos_words if w in text)
    neg_score = sum(1 for w in neg_words if w in text)
    if pos_score > neg_score: return "บวก 😊"
    elif neg_score > pos_score: return "ลบ 😟"
    else: return "ปกติ 😐"

# --- 3. ฟังก์ชันสร้าง Network Analysis ---
def plot_network(words, font_prop):
    G = nx.Graph()
    pairs = []
    for i in range(len(words)-1):
        if words[i] != words[i+1]:
            pairs.append(tuple(sorted((words[i], words[i+1]))))
    
    pair_counts = Counter(pairs).most_common(20)
    for pair, weight in pair_counts:
        G.add_edge(pair[0], pair[1], weight=weight)
    
    if len(G.nodes) == 0: return None

    fig, ax = plt.subplots(figsize=(10, 7))
    pos = nx.spring_layout(G, k=0.5, seed=42)
    weights = [G[u][v]['weight'] for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, width=weights, edge_color='skyblue', alpha=0.5)
    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='orange', alpha=0.8)
    
    for node, (x, y) in pos.items():
        ax.text(x, y, node, fontproperties=font_prop, fontsize=14, 
                ha='center', va='center', bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
    plt.axis('off')
    return fig

# --- 4. การนำเข้า Library ---
try:
    from pythainlp.tokenize import word_tokenize
    from pythainlp.corpus import thai_stopwords
    THAI_READY = True
except:
    THAI_READY = False

st.set_page_config(layout="wide", page_title="Advanced Research Analysis")
st.title("🕸️ ระบบวิเคราะห์งานวิจัยขั้นสูง (Full Edition)")

if not THAI_READY:
    st.error("❌ Library ไม่พร้อมใช้งาน")
    st.stop()

font_p = setup_font()
uploaded_files = st.file_uploader("อัปโหลดไฟล์บทสัมภาษณ์ (.txt)", type=['txt'], accept_multiple_files=True)

if uploaded_files:
    comparison_list = []
    for file in uploaded_files:
        text = file.read().decode("utf-8")
        tokens = word_tokenize(text, keep_whitespace=False)
        stop_words = list(thai_stopwords()) + ['เนาะ', 'นะ', 'ครับ', 'ค่ะ', 'คือ', 'แบบ', 'ว่า']
        
        filtered_words = [t.strip() for t in tokens if t.strip() and t not in stop_words and len(t.strip()) >= 5 and not re.match(r'^[0-9\W]+$', t)]
        word_counts = Counter(filtered_words)
        filtered_final = [w for w in filtered_words if word_counts[w] >= 3]
        
        s_label = analyze_sentiment_thai(text)
        comparison_list.append({"ไฟล์": file.name, "อารมณ์": s_label})

        with st.expander(f"📑 วิเคราะห์เชิงลึก: {file.name}", expanded=True):
            tab1, tab2, tab3 = st.tabs(["📊 สถิติ & Word Cloud", "🕸️ โครงข่ายความสัมพันธ์", "📄 ต้นฉบับ"])
            
            with tab1:
                c1, c2 = st.columns(2)
                with c1:
                    st.write(f"**โทนความรู้สึก:** {s_label}")
                    if filtered_final:
                        wc = WordCloud(width=600, height=300, background_color="white", regexp=r"[\u0e00-\u0e7f]+", font_path=font_path).generate(" ".join(filtered_final))
                        fig_wc, ax_wc = plt.subplots()
                        ax_wc.imshow(wc)
                        ax_wc.axis("off")
                        st.pyplot(fig_wc)
                        
                        # --- ปุ่มดาวน์โหลด PNG กลับมาแล้ว ---
                        buf_wc = BytesIO()
                        fig_wc.savefig(buf_wc, format="png")
                        st.download_button(label="💾 ดาวน์โหลด Word Cloud (PNG)", data=buf_wc.getvalue(), file_name=f"cloud_{file.name}.png", mime="image/png")
                with c2:
                    df_counts = pd.DataFrame(Counter(filtered_final).most_common(10), columns=['คำ', 'จำนวน'])
                    st.table(df_counts)

            with tab2:
                st.subheader("โหนดความสัมพันธ์ของคำสำคัญ")
                if len(filtered_words) > 5:
                    fig_net = plot_network(filtered_words, font_p)
                    if fig_net:
                        st.pyplot(fig_net)
                else:
                    st.warning("ข้อมูลน้อยเกินไปสำหรับการวิเคราะห์โครงข่าย")

            with tab3:
                # --- แสดงตัวอย่างข้อความ 5 ตัวอย่างแรก ---
                st.subheader("📄 ตัวอย่างข้อความจากไฟล์ (5 ตัวอย่างแรก)")
                sentences = text.split('\n')
                sample_text = "\n\n".join([s for s in sentences if s.strip()][:5])
                st.info(sample_text if sample_text else "ไม่พบข้อความในไฟล์")
                
                st.divider()
                st.write("**ข้อความทั้งหมด:**")
                st.text_area("Content Viewer", value=text, height=200)

    st.divider()
    st.subheader("📋 ตารางสรุปเปรียบเทียบทุกเคส")
    st.table(pd.DataFrame(comparison_list))
