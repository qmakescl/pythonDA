import os
import fitz  # PyMuPDF
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import networkx as nx
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
    tokens = okt.morphs(text, stem=True)  # 형태소 분석 및 원형 복원
    words = [word for word in tokens if word not in korean_stopwords and len(word) > 1]  # 불용어 제거 및 한 글자 단어 제거
    return words

# Bi-gram을 추출하는 함수 (상위 20개 단어만 사용)
def get_top_bigrams(words, top_n=20):
    word_counts = Counter(words)
    top_words = {word for word, _ in word_counts.most_common(top_n)}  # 상위 20개 단어 선택
    bigram_list = [bigram for bigram in bigrams(words) if bigram[0] in top_words and bigram[1] in top_words]
    return bigram_list

# Bi-gram을 시각화하고 저장하는 함수 (엣지 굵기 및 투명도 조절)
def save_bigram_graph(bigram_list, output_file):
    if not bigram_list:  # 빈 리스트일 경우 처리
        print(f"⚠ Warning: No bigrams found for {output_file}, skipping...")
        return

    bigram_counts = Counter(bigram_list)  # Bi-gram 빈도수 계산
    bigram_graph = nx.Graph()

    # 그래프에 엣지 추가 (연결 강도 반영)
    for (word1, word2), count in bigram_counts.items():
        bigram_graph.add_edge(word1, word2, weight=count)  # 두 단어 연결 (edge)

    # 레이아웃 설정
    plt.figure(figsize=(10, 6))
    pos = nx.spring_layout(bigram_graph, k=0.5)  # 노드 위치 자동 배치

    # ✅ 엣지 가중치에 따라 선 굵기 설정 (최대 굵기 제한)
    edges = bigram_graph.edges(data=True)
    edge_weights = [min(data["weight"] * 0.3, 3) for _, _, data in edges]  # 최대 굵기 3 제한

    # ✅ 노드 및 엣지 그리기 (노란색 엣지, 투명도 80%)
    nx.draw(
        bigram_graph, pos, with_labels=False, node_color="lightblue",
        edge_color="yellow", width=edge_weights, alpha=0.8
    )

    # ✅ 한글 폰트 강제 적용하여 라벨 추가
    labels = {node: node for node in bigram_graph.nodes()}
    nx.draw_networkx_labels(bigram_graph, pos, labels, font_family="AppleGothic", font_size=10)

    # 이미지로 저장
    plt.savefig(output_file, format="png", dpi=300)
    plt.close()

# 한글 워드클라우드를 생성하고 저장하는 함수 (고해상도 설정)
def save_wordcloud(words, output_file):
    if not words:  # 빈 리스트이면 저장하지 않음
        print(f"⚠ Warning: No words found for {output_file}, skipping...")
        return

    text = " ".join(words)
    wordcloud = WordCloud(
        font_path=MAC_FONT_PATH,  # MacOS 한글 폰트 적용
        width=1600, height=800, background_color="white"
    ).generate(text)

    plt.figure(figsize=(16, 8))  # ✅ 더 큰 이미지 크기 설정
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")

    # ✅ DPI 값을 600으로 설정하여 고해상도 저장
    plt.savefig(output_file, format="png", dpi=600)
    plt.close()


# 디렉토리별 개별 실행 및 전체 실행
base_directory = "/Users/yoonani/OneDrive - CultureLab/Works/2025/로컬벤쳐_창경/첨부3. 평가자료"  # 메인 디렉토리 경로
all_words = []  # 전체 데이터 저장

# 각 하위 디렉토리에 대해 개별 분석
for sub_dir in sorted(os.listdir(base_directory)):  # 정렬하여 순차 처리
    sub_dir_path = os.path.join(base_directory, sub_dir)

    if os.path.isdir(sub_dir_path):  # 디렉토리인지 확인
        print(f"Processing directory: {sub_dir}")

        raw_text = extract_text_from_pdfs(sub_dir_path)
        processed_words = preprocess_text(raw_text)

        # 저장할 파일 경로 설정
        wordcloud_path = os.path.join(OUTPUT_DIR, f"{sub_dir}_wordcloud.png")
        bigram_path = os.path.join(OUTPUT_DIR, f"{sub_dir}_bigram.png")

        # 각 디렉토리별 WordCloud 저장
        save_wordcloud(processed_words, wordcloud_path)

        # 각 디렉토리별 Bi-gram 네트워크 저장
        bigrams_list = get_top_bigrams(processed_words)
        save_bigram_graph(bigrams_list, bigram_path)

        # 전체 분석을 위한 데이터 저장
        all_words.extend(processed_words)

# 모든 문서를 통합하여 분석
if all_words:
    print("Processing all documents together...")

    # 전체 WordCloud 저장
    all_wordcloud_path = os.path.join(OUTPUT_DIR, "all_wordcloud.png")
    save_wordcloud(all_words, all_wordcloud_path)

    # 전체 Bi-gram 네트워크 저장
    all_bigram_path = os.path.join(OUTPUT_DIR, "all_bigram.png")
    all_bigrams_list = get_top_bigrams(all_words)
    save_bigram_graph(all_bigrams_list, all_bigram_path)

print("✅ 모든 분석이 완료되었습니다. 결과는 'output' 폴더에서 확인하세요.")
