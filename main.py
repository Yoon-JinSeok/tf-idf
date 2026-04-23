import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

st.set_page_config(page_title="한국어 TF-IDF 분석기", layout="wide")
st.title("📝 한국어 TF-IDF 단계별 분석기")

# ──────────────────────────────────────────────
# 세션 상태 초기화
# ──────────────────────────────────────────────
for key in ["data", "data1", "df_tf", "df_tfidf", "pos_option"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ──────────────────────────────────────────────
# 1단계: 문장 입력
# ──────────────────────────────────────────────
st.header("1단계. 문장 입력")
st.caption("여러 문장을 줄바꿈(엔터)으로 구분해서 입력하세요. 빈 줄은 무시됩니다.")

text_input = st.text_area(
    "문장 입력",
    height=200,
    placeholder="예)\n나는 사과를 좋아한다\n바나나는 노랗고 달콤하다\n오늘 날씨가 맑다",
)

if st.button("1단계 실행 ▶", key="btn1"):
    lines = [line.strip() for line in text_input.split("\n") if line.strip()]
    if not lines:
        st.warning("입력된 문장이 없습니다.")
    else:
        st.session_state.data = lines
        # 이후 단계 결과 초기화
        st.session_state.data1 = None
        st.session_state.df_tf = None
        st.session_state.df_tfidf = None

if st.session_state.data is not None:
    st.success(f"총 {len(st.session_state.data)}개 문장이 저장되었습니다.")
    st.write("**data 리스트:**")
    st.write(st.session_state.data)

st.divider()

# ──────────────────────────────────────────────
# 2단계: 형태소 분석 (품사 선택)
# ──────────────────────────────────────────────
st.header("2단계. 형태소 분석 및 불용어 처리")

pos_option = st.radio(
    "추출할 품사를 선택하세요.",
    options=["명사만", "명사 + 형용사"],
    index=1,
    horizontal=True,
    key="pos_radio",
)

if st.button("2단계 실행 ▶", key="btn2"):
    if st.session_state.data is None:
        st.warning("먼저 1단계를 실행해 주세요.")
    else:
        try:
            from konlpy.tag import Okt
            okt = Okt()

            # 선택한 품사 집합 구성
            if pos_option == "명사만":
                target_pos = {"Noun"}
            else:
                target_pos = {"Noun", "Adjective"}

            data1 = []
            for sentence in st.session_state.data:
                words_list = okt.pos(sentence, stem=True)
                words = [w for w, p in words_list if p in target_pos]
                data1.append(" ".join(words))

            st.session_state.data1 = data1
            st.session_state.pos_option = pos_option
            # 이후 단계 결과 초기화
            st.session_state.df_tf = None
            st.session_state.df_tfidf = None
        except Exception as e:
            st.error(f"형태소 분석 중 오류가 발생했습니다: {e}")

if st.session_state.data1 is not None:
    st.success(f"형태소 분석 완료 (선택: {st.session_state.pos_option})")
    st.write("**data1 (추출된 단어):**")
    st.write(st.session_state.data1)

st.divider()

# ──────────────────────────────────────────────
# 3단계: TF(단어 빈도수)
# ──────────────────────────────────────────────
st.header("3단계. 단어 빈도수(TF) 표")

if st.button("3단계 실행 ▶", key="btn3"):
    if st.session_state.data1 is None:
        st.warning("먼저 2단계를 실행해 주세요.")
    else:
        try:
            vec = CountVectorizer()
            tf = vec.fit_transform(st.session_state.data1)
            df = pd.DataFrame(tf.toarray(), columns=vec.get_feature_names_out())
            st.session_state.df_tf = df
        except ValueError as e:
            st.error(f"TF 계산 오류: {e} (추출된 단어가 없을 수 있습니다.)")

if st.session_state.df_tf is not None:
    st.dataframe(st.session_state.df_tf, use_container_width=True)

st.divider()

# ──────────────────────────────────────────────
# 4단계: TF-IDF
# ──────────────────────────────────────────────
st.header("4단계. TF-IDF 표")

if st.button("4단계 실행 ▶", key="btn4"):
    if st.session_state.data1 is None:
        st.warning("먼저 2단계를 실행해 주세요.")
    else:
        try:
            vec1 = TfidfVectorizer()
            tfidf = vec1.fit_transform(st.session_state.data1)
            df1 = pd.DataFrame(tfidf.toarray(), columns=vec1.get_feature_names_out())
            st.session_state.df_tfidf = df1
        except ValueError as e:
            st.error(f"TF-IDF 계산 오류: {e}")

if st.session_state.df_tfidf is not None:
    st.dataframe(
        st.session_state.df_tfidf.style.format("{:.4f}"),
        use_container_width=True,
    )

st.divider()

# ──────────────────────────────────────────────
# 5단계: 행별 핵심 단어
# ──────────────────────────────────────────────
st.header("5단계. 행별 핵심 단어 (TF-IDF 최댓값)")

if st.button("5단계 실행 ▶", key="btn5"):
    if st.session_state.df_tfidf is None:
        st.warning("먼저 4단계를 실행해 주세요.")
    else:
        key_words = st.session_state.df_tfidf.idxmax(axis=1)
        result_df = pd.DataFrame({
            "원본 문장": st.session_state.data,
            "핵심 단어": key_words.values,
        })
        st.dataframe(result_df, use_container_width=True)
