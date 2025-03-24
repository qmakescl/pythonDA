import os
import re
import fitz  # PyMuPDF
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
import community as community_louvain  # Louvain 클러스터링을 위한 라이브러리, pip install python-louvain
from wordcloud import WordCloud
from collections import Counter
from nltk.util import bigrams
from konlpy.tag import Okt

# MacOS 기본 한글 폰트 설정
MAC_FONT_PATH = "/System/Library/Fonts/Supplemental/AppleGothic.ttf"
plt.rc("font", family="AppleGothic")  # Matplotlib에서 한글 폰트 적용
plt.rcParams["axes.unicode_minus"] = False  # 마이너스 기호 깨짐 방지

# 결과 저장 폴더 생성
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 한글 형태소 분석기
okt = Okt()

# 한글 불용어 리스트 (추가 가능)
korean_stopwords = set(["은", "는", "이", "가", "를", "에", "의", "도", "으로", "에서", "한", "있다", 
                        "한다", "것", "그리고", "그", "하지만", "수", "할", "잘", "하다", "하고", "지금",
                        "더", "등", "때", "대한", "오늘", "많은", "많이", "안", "위해", "또한", "모든",
                        "말했다", "않고", "않는", "않음", "않는다", "않는다고", "않는다는", "않는다며",
                        "귀하", "되다", "되어다", "되어", "되었다", "되었으며", "되었고", "되었는데",
                        "통해", "이라고", "이라는", "이라며", "이라면", "이라서", "이라", "이며", "이고",
                        "통한", "참가", "신청서", 
                        "사업", "내용", "지원", "기업", "사업자", "지역", "사업비", "사업계획서", "사업목적"])

# 단어 변환 딕셔너리 (동의어 통합)
word_replacements = {
    "강원도": "강원",
    "강원특별자치도": "강원"
}

# PDF 파일에서 텍스트 추출하는 함수
def extract_text_from_pdfs(directory):
    text_data = ""
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.lower().endswith(".pdf"):
                pdf_path = os.path.join(root, filename)
                doc = fitz.open(pdf_path)
                for page in doc:
                    text_data += page.get_text("text") + " "
    return text_data

# 한글 텍스트 전처리 함수
def preprocess_text(text):
    # 모든 단어 추출
    # tokens = okt.morphs(text, stem=True)  # 형태소 분석 및 원형 복원
    # 명사만 추출
    tokens = okt.nouns(text)  # 형태소 분석 및 원형 복원
    
    # 특수문자 제거: 한글, 영문자만 포함
    words = [word for word in tokens if re.match(r"^[가-힣a-zA-Z]+$", word)]

    # 단어 통합: 특정 단어를 하나의 단어로 변환, 유사어 처리
    words = [word_replacements.get(word, word) for word in words]
        
    words = [word for word in words if word not in korean_stopwords and len(word) > 1]  # 불용어 제거 및 한 글자 단어 제거
    return words

# 단어 빈도수를 계산하여 CSV로 저장하는 함수
def save_word_frequencies(words, output_file):
    word_counts = Counter(words)
    df = pd.DataFrame(word_counts.items(), columns=["Word", "Frequency"])
    df = df.sort_values(by="Frequency", ascending=False)  # 내림차순 정렬
    df.to_csv(output_file, index=False, encoding="utf-8-sig")  # CSV 저장

# Bi-gram을 추출하는 함수 (상위 20개 단어만 사용)
def get_top_bigrams(words, top_n=20):
    word_counts = Counter(words)
    top_words = {word for word, _ in word_counts.most_common(top_n)}  # 상위 20개 단어 선택
    bigram_list = [(word1, word2) for word1, word2 in bigrams(words) if word1 in top_words and word2 in top_words]
    return bigram_list

# Bi-gram을 추출하여 CSV로 저장하는 함수
def save_bigrams(words, output_file, top_n=20):
    word_counts = Counter(words)
    top_words = {word for word, _ in word_counts.most_common(top_n)}  # 상위 20개 단어 선택
    bigram_list = [(word1, word2) for word1, word2 in bigrams(words) if word1 in top_words and word2 in top_words]

    bigram_counts = Counter(bigram_list)
    df = pd.DataFrame(bigram_counts.items(), columns=["Bigram", "Frequency"])
    df["Word1"], df["Word2"] = zip(*df["Bigram"])  # Bi-gram 분리
    df.drop(columns=["Bigram"], inplace=True)  # 기존 Bi-gram 컬럼 삭제
    df = df.sort_values(by="Frequency", ascending=False)  # 내림차순 정렬
    df.to_csv(output_file, index=False, encoding="utf-8-sig")  # CSV 저장

# # Bi-gram을 시각화하고 저장하는 함수
# def save_bigram_graph(bigram_list, output_file):
#     if not bigram_list:
#         print(f"⚠ Warning: No bigrams found for {output_file}, skipping...")
#         return

#     bigram_counts = Counter(bigram_list)
#     bigram_graph = nx.Graph()

#     for (word1, word2), count in bigram_counts.items():
#         bigram_graph.add_edge(word1, word2, weight=count)

#     plt.figure(figsize=(10, 6))
#     pos = nx.spring_layout(bigram_graph, k=0.5)

#     edges = bigram_graph.edges(data=True)
#     edge_weights = [min(data["weight"] * 0.3, 3) for _, _, data in edges]  # 최대 굵기 3 제한

#     nx.draw(
#         bigram_graph, pos, with_labels=False, node_color="lightblue",
#         edge_color="yellow", width=edge_weights, alpha=0.8
#     )

#     labels = {node: node for node in bigram_graph.nodes()}
#     nx.draw_networkx_labels(bigram_graph, pos, labels, font_family="AppleGothic", font_size=10)

#     plt.savefig(output_file, format="png", dpi=300)
#     plt.close()

# Bi-gram을 시각화하고 저장하는 함수 (클러스터링 적용, 다른 클러스터 간 연결 점선 처리)
def save_bigram_graph(bigram_list, output_file):
    if not bigram_list:
        print(f"⚠ Warning: No bigrams found for {output_file}, skipping...")
        return

    bigram_counts = Counter(bigram_list)  # Bi-gram 빈도수 계산
    bigram_graph = nx.Graph()

    for (word1, word2), count in bigram_counts.items():
        bigram_graph.add_edge(word1, word2, weight=count)

    # ✅ Louvain 클러스터링 적용
    partition = community_louvain.best_partition(bigram_graph)
    unique_clusters = list(set(partition.values()))
    
    # ✅ 클러스터 개수를 제한 (최대 5개)
    if len(unique_clusters) > 5:
        print(f"⚠ Warning: Too many clusters ({len(unique_clusters)}), merging similar ones...")
        for node in partition:
            partition[node] = partition[node] % 5  # 최대 5개로 제한

    # ✅ 색상 설정 (클러스터별로 구분)
    cluster_colors = ["#FA7070", "#FFA725", "#FFF5E4", "#C1D8C3", "#6A9C89"]
    node_colors = [cluster_colors[partition[node] % len(cluster_colors)] for node in bigram_graph.nodes()]

    # ✅ 노드 위치 설정
    plt.figure(figsize=(10, 6))
    pos = nx.spring_layout(bigram_graph, k=0.5)

    # ✅ 엣지 스타일 구분 (같은 클러스터 vs 다른 클러스터)
    same_cluster_edges = []
    different_cluster_edges = []

    for node1, node2, data in bigram_graph.edges(data=True):
        if partition[node1] == partition[node2]:  # 같은 클러스터 내 연결
            same_cluster_edges.append((node1, node2, data))
        else:  # 다른 클러스터 간 연결
            different_cluster_edges.append((node1, node2, data))

    # ✅ 같은 클러스터 내 연결: 실선, 색상은 노란색, 투명도 80%
    edge_weights = [min(data["weight"] * 0.3, 3) for _, _, data in same_cluster_edges]
    nx.draw_networkx_edges(
        bigram_graph, pos, edgelist=[(u, v) for u, v, _ in same_cluster_edges],
        edge_color="yellow", width=edge_weights, alpha=0.8
    )

    # ✅ 다른 클러스터 간 연결: 점선, 색상은 회색, 투명도 20%
    nx.draw_networkx_edges(
        bigram_graph, pos, edgelist=[(u, v) for u, v, _ in different_cluster_edges],
        edge_color="gray", width=1, alpha=0.2, style="dashed"
    )

    # ✅ 노드 및 클러스터별 색상 적용
    nx.draw_networkx_nodes(bigram_graph, pos, node_color=node_colors, node_size=500)

    # ✅ 한글 폰트 적용하여 라벨 추가
    labels = {node: node for node in bigram_graph.nodes()}
    nx.draw_networkx_labels(bigram_graph, pos, labels, font_family="AppleGothic", font_size=8)

    # ✅ 이미지로 저장
    plt.savefig(output_file, format="png", dpi=300)
    plt.close()

# 한글 워드클라우드를 생성하고 저장하는 함수 (고해상도 설정)
def save_wordcloud(words, output_file):
    if not words:
        print(f"⚠ Warning: No words found for {output_file}, skipping...")
        return

    text = " ".join(words)
    wordcloud = WordCloud(
        font_path=MAC_FONT_PATH,
        width=1600, height=800, background_color="white"
    ).generate(text)

    plt.figure(figsize=(16, 8))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")

    plt.savefig(output_file, format="png", dpi=600)
    plt.close()


# 디렉토리별 개별 실행 및 전체 실행
base_directory = "/Users/yoonani/OneDrive - CultureLab/Works/2025/로컬벤쳐_창경/첨부3. 평가자료"
all_words = []  # 전체 데이터 저장

# 각 하위 디렉토리에 대해 개별 분석
for sub_dir in sorted(os.listdir(base_directory)):
    sub_dir_path = os.path.join(base_directory, sub_dir)

    if os.path.isdir(sub_dir_path):
        print(f"Processing directory: {sub_dir}")

        raw_text = extract_text_from_pdfs(sub_dir_path)
        processed_words = preprocess_text(raw_text)

        if processed_words:
            wordcloud_path = os.path.join(OUTPUT_DIR, f"{sub_dir}_wordcloud.png")
            bigram_path = os.path.join(OUTPUT_DIR, f"{sub_dir}_bigram.png")
            word_freq_path = os.path.join(OUTPUT_DIR, f"{sub_dir}_word_frequencies.csv")
            bigram_csv_path = os.path.join(OUTPUT_DIR, f"{sub_dir}_bigrams.csv")

            save_wordcloud(processed_words, wordcloud_path)
            bigrams_list = get_top_bigrams(processed_words)
            save_bigram_graph(bigrams_list, bigram_path)

            save_word_frequencies(processed_words, word_freq_path)
            save_bigrams(processed_words, bigram_csv_path)

            all_words.extend(processed_words)

# 모든 문서를 통합하여 분석
if all_words:
    print("Processing all documents together...")

    save_wordcloud(all_words, os.path.join(OUTPUT_DIR, "all_wordcloud.png"))
    all_bigrams_list = get_top_bigrams(all_words)
    save_bigram_graph(all_bigrams_list, os.path.join(OUTPUT_DIR, "all_bigram.png"))

    save_word_frequencies(all_words, os.path.join(OUTPUT_DIR, "all_word_frequencies.csv"))
    save_bigrams(all_words, os.path.join(OUTPUT_DIR, "all_bigrams.csv"))

print("✅ 모든 분석이 완료되었습니다. 결과는 'output' 폴더에서 확인하세요.")
