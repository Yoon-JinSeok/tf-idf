import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

st.set_page_config(page_title="한글 TF-IDF 분석기", layout="wide")
st.title("📝 한글 문장 TF-IDF 분석기")
st.caption("문장을 입력하고 단계별 버튼을 눌러 분석을 진행하세요.")

# ---------------------------
# 세션 상태 초기화
# ---------------------------
for key in ["data", "data1", "df_tf", "df_tfidf"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ---------------------------
# 1단계: 문장 입력
# ---------------------------
st.header("1단계 · 문장 입력")
st.write("여러 문장을 입력하세요. **엔터(줄바꿈)** 를 기준으로 문장이 나누어집니다. 빈 줄은 무시됩니다.")

user_input = st.text_area(
    "문장 입력",
    height=200,
    placeholder="예시)\n오늘 날씨가 정말 맑고 좋다\n내일은 비가 올 것 같다\n\n주말에는 산책을 하고 싶다",
)

if st.button("1단계 실행: 문장 리스트 생성", type="primary"):
    data = [line.strip() for line in user_input.split("\n") if line.strip()]
    if not data:
        st.warning("입력된 문장이 없습니다. 문장을 입력한 후 다시 실행해 주세요.")
    else:
        st.session_state.data = data
        # 이후 단계 결과 초기화
        st.session_state.data1 = None
        st.session_state.df_tf = None
        st.session_state.df_tfidf = None
        st.success(f"총 {len(data)}개의 문장이 리스트에 저장되었습니다.")

if st.session_state.data is not None:
    st.write("**data 리스트:**")
    st.write(st.session_state.data)

st.divider()

# ---------------------------
# 2단계: 형태소 분석 (명사/형용사 추출)
# ---------------------------
st.header("2단계 · 명사·형용사 추출 (KoNLPy Okt)")
st.write("`Okt` 형태소 분석기를 이용하여 **명사(Noun)** 와 **형용사(Adjective)** 만 추출하고, 어간을 사전형으로 변환합니다.")

if st.button("2단계 실행: 형태소 분석"):
    if st.session_state.data is None:
        st.warning("먼저 1단계를 실행하세요.")
    else:
        try:
            from konlpy.tag import Okt
            okt = Okt()

            data1 = []
            for sentence in st.session_state.data:
                words_list = okt.pos(sentence, stem=True)
                words = [word for word, pos in words_list if pos in ("Noun", "Adjective")]
                result = " ".join(words)
                data1.append(result)

            st.session_state.data1 = data1
            st.session_state.df_tf = None
            st.session_state.df_tfidf = None
            st.success("형태소 분석이 완료되었습니다.")
        except Exception as e:
            st.error(f"형태소 분석 중 오류가 발생했습니다: {e}")
            st.info("Streamlit Cloud에서는 `packages.txt`에 `default-jdk`를 추가해야 KoNLPy가 동작합니다.")

if st.session_state.data1 is not None:
    st.write("**data1 리스트 (전처리 결과):**")
    st.write(st.session_state.data1)

st.divider()

# ---------------------------
# 3단계: TF (단어 빈도수)
# ---------------------------
st.header("3단계 · 단어 빈도수(TF) 벡터화")
st.write("`CountVectorizer` 를 이용하여 문장별 단어 빈도수를 벡터로 변환합니다.")

if st.button("3단계 실행: TF 계산"):
    if st.session_state.data1 is None:
        st.warning("먼저 2단계를 실행하세요.")
    else:
        # 빈 문자열만 있으면 오류 발생 가능
        if all(not s.strip() for s in st.session_state.data1):
            st.error("추출된 단어가 없습니다. 입력 문장을 다시 확인해 주세요.")
        else:
            try:
                vec = CountVectorizer()
                tf = vec.fit_transform(st.session_state.data1)
                df = pd.DataFrame(tf.toarray(), columns=vec.get_feature_names_out())
                st.session_state.df_tf = df
                st.session_state.df_tfidf = None
                st.success("TF 계산이 완료되었습니다.")
            except ValueError as e:
                st.error(f"TF 계산 오류: {e}")

if st.session_state.df_tf is not None:
    st.write("**단어 빈도수(TF) 표:**")
    st.dataframe(st.session_state.df_tf, use_container_width=True)

st.divider()

# ---------------------------
# 4단계: TF-IDF
# ---------------------------
st.header("4단계 · TF-IDF 계산")
st.write("`TfidfVectorizer` 를 이용하여 TF-IDF 값을 계산합니다.")

if st.button("4단계 실행: TF-IDF 계산"):
    if st.session_state.data1 is None:
        st.warning("먼저 2단계를 실행하세요.")
    else:
        if all(not s.strip() for s in st.session_state.data1):
            st.error("추출된 단어가 없습니다. 입력 문장을 다시 확인해 주세요.")
        else:
            try:
                vec1 = TfidfVectorizer()
                tfidf = vec1.fit_transform(st.session_state.data1)
                df1 = pd.DataFrame(tfidf.toarray(), columns=vec1.get_feature_names_out())
                st.session_state.df_tfidf = df1
                st.success("TF-IDF 계산이 완료되었습니다.")
            except ValueError as e:
                st.error(f"TF-IDF 계산 오류: {e}")

if st.session_state.df_tfidf is not None:
    st.write("**TF-IDF 표:**")
    st.dataframe(st.session_state.df_tfidf.style.format("{:.4f}"), use_container_width=True)

st.divider()

# ---------------------------
# 5단계: 핵심 단어 추출
# ---------------------------
st.header("5단계 · 문장별 핵심 단어 추출")
st.write("각 문장에서 **TF-IDF 값이 가장 큰 단어**를 핵심 단어로 추출합니다.")

if st.button("5단계 실행: 핵심 단어 추출"):
    if st.session_state.df_tfidf is None:
        st.warning("먼저 4단계를 실행하세요.")
    else:
        data = st.session_state.data
        keywords = st.session_state.df_tfidf.idxmax(axis=1)

        result_df = pd.DataFrame({
            "문장": data,
            "핵심 단어": keywords.values,
        })
        st.write("**원본 문장 리스트 (data):**")
        st.write(data)

        st.write("**각 문장의 핵심 단어:**")
        st.dataframe(result_df, use_container_width=True)
