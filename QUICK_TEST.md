# 🚀 AI 챗봇 빠른 통합 테스트 (15분)

## 현재 상태

✅ v9_app.py에 AI 어시스턴트 통합 완료
✅ requirements.txt 업데이트 완료
✅ ai_chatbot_hybrid.py 준비 완료

---

## 1단계: 의존성 설치 (2분)

```bash
pip install chromadb google-generativeai duckdb
```

**확인:**
```bash
python -c "import chromadb; import google.generativeai; print('✅ 설치 완료')"
```

---

## 2단계: Secrets 설정 (3분)

### 로컬 환경

```bash
# .streamlit 디렉토리 생성
mkdir -p .streamlit

# secrets.toml 생성
cp secrets.toml.example .streamlit/secrets.toml

# 편집기로 열어서 API 키 입력
code .streamlit/secrets.toml  # 또는 nano, vim 등
```

### secrets.toml 내용

```toml
[gemini]
api_key = "YOUR_ACTUAL_GEMINI_KEY"  # Google AI Studio에서 발급
generation_model = "gemini-2.0-flash-exp"  # 또는 "gemini-1.5-flash"
embedding_model = "text-embedding-004"

[chroma]
api_key = "YOUR_ACTUAL_CHROMA_KEY"  # Chroma Cloud에서 발급
tenant = "1468ca5f-87c8-4184-b5ec-bfcb1d03546b"
database = "SCM_DASHBOARD"
```

### API 키 발급

1. **Gemini API 키**
   - 사이트: https://aistudio.google.com/app/apikey
   - "Create API Key" 클릭
   - 무료 쿼터: 60 queries/min

2. **Chroma Cloud API 키**
   - 사이트: https://www.trychroma.com/
   - "Start for free" 계정 생성
   - Dashboard → API Keys에서 생성
   - 무료 플랜: 10GB storage

---

## 3단계: 로컬 실행 (1분)

```bash
streamlit run v9_app.py
```

**기대 결과:**
- 브라우저에 `http://localhost:8501` 자동 오픈
- 기존 대시보드 정상 표시
- 맨 아래 "🤖 AI 어시스턴트" 섹션 표시

**에러 시:**
- `ImportError`: ai_chatbot_hybrid.py 파일 위치 확인
- `KeyError`: .streamlit/secrets.toml 설정 확인
- `ConnectionError`: 인터넷 연결 확인

---

## 4단계: 기본 동작 테스트 (5분)

### 테스트 1: 정량 계산 ✅

**목표:** 판다스 계산이 정확한지 확인

1. 대시보드에서 필터 설정
   - 센터: REVVN, 태광KR (2개)
   - SKU: BA00021, BA00022 (2개)

2. 아래로 스크롤하여 "🤖 AI 어시스턴트" 섹션 이동

3. 질문 입력: **"총 재고는?"**

4. "💬 질문하기" 클릭

**기대 결과:**
```
🔖 질문 유형: quantitative | 대상: ['REVVN', '태광KR'] × ['BA00021', 'BA00022']

📊 답변
현재 총 재고는 5,500개입니다. 센터별로는 태광KR 3,000개, REVVN 2,500개로 구성되어 있습니다.

🔢 계산 상세
계산 유형: total
결과: 5500
센터별 분해:
- 태광KR: 3,000
- REVVN: 2,500
```

**확인 사항:**
- ✅ 숫자가 정확한가? (대시보드의 "재고 현황 테이블"과 비교)
- ✅ 2-3초 내 응답하는가?
- ✅ 에러 없이 완료되는가?

---

### 테스트 2: 탐색 검색 ✅

**목표:** 벡터 검색이 동작하는지 확인

1. 질문 입력: **"REVVN에는 어떤 SKU가 있나요?"**

2. "💬 질문하기" 클릭

**기대 결과:**
```
🔖 질문 유형: exploratory | 대상: ['REVVN'] × [전체 SKU]

📊 답변
REVVN 센터에는 BA00021(프로바이오틱스), BA00022(오메가3) 등 총 5개 SKU가 보유되어 있습니다.

🔎 근거 데이터
1. [SNAPSHOT] 날짜:2025-01-15 센터:REVVN SKU:BA00021 재고:1500개
2. [SNAPSHOT] 날짜:2025-01-15 센터:REVVN SKU:BA00022 재고:1000개
...
```

**확인 사항:**
- ✅ REVVN 관련 데이터만 표시되는가?
- ✅ 근거 데이터가 5개 표시되는가?
- ✅ 5-10초 내 응답하는가? (첫 실행은 인덱싱 포함 10초)

---

### 테스트 3: 센터별 집계 ✅

**목표:** 고급 집계 기능 확인

1. 질문 입력: **"센터별 재고는?"**

2. "💬 질문하기" 클릭

**기대 결과:**
```
📊 답변
센터별 재고 현황은 다음과 같습니다:
- 태광KR: 3,000개 (전체의 55%)
- REVVN: 2,500개 (45%)

태광KR이 가장 많은 재고를 보유하고 있습니다.

🔢 계산 상세
계산 유형: group_by_center
결과:
- 태광KR: 3,000
- REVVN: 2,500
```

**확인 사항:**
- ✅ 모든 선택된 센터가 표시되는가?
- ✅ 합계가 테스트 1의 총 재고와 일치하는가?
- ✅ 백분율이 표시되는가?

---

## 5단계: 에러 처리 테스트 (2분)

### 테스트 4: 비즈니스 판단 질문 (아직 미지원)

1. 질문 입력: **"재고가 부족한가요?"**

2. "💬 질문하기" 클릭

**기대 결과:**
```
🚧 비즈니스 판단 질문은 2단계에서 지원 예정입니다.
현재는 정량 계산과 탐색 검색만 가능합니다.
```

**확인 사항:**
- ✅ 우아하게 안내 메시지 표시
- ✅ 앱이 중단되지 않음

---

## 6단계: 필터 변경 테스트 (2분)

**목표:** 필터 캐싱이 동작하는지 확인

1. 사이드바에서 센터 변경: REVVN만 선택

2. 아래로 스크롤하여 AI 섹션

3. 자동 인덱싱 메시지 확인:
   ```
   📚 벡터 인덱스 준비 중...
   ✅ 800건 인덱싱 완료
   ```

4. 질문: **"총 재고는?"**

**기대 결과:**
- REVVN만의 재고 표시 (태광KR 제외)
- 숫자가 이전과 다름

**확인 사항:**
- ✅ 필터 변경 감지됨
- ✅ 자동 재인덱싱 완료
- ✅ 결과가 필터에 맞게 변경됨

---

## ✅ 테스트 완료 체크리스트

- [ ] 의존성 설치 완료
- [ ] Secrets 설정 완료
- [ ] 로컬 실행 성공
- [ ] 테스트 1: 정량 계산 ✅
- [ ] 테스트 2: 탐색 검색 ✅
- [ ] 테스트 3: 센터별 집계 ✅
- [ ] 테스트 4: 에러 처리 ✅
- [ ] 테스트 5: 필터 변경 ✅

---

## 🐛 문제 해결

### "ImportError: No module named 'chromadb'"
```bash
pip install chromadb google-generativeai duckdb
```

### "KeyError: 'gemini'" 또는 "KeyError: 'chroma'"
- `.streamlit/secrets.toml` 파일 확인
- API 키가 올바르게 입력되었는지 확인

### "Connection refused" 또는 "Timeout"
- 인터넷 연결 확인
- Gemini API 쿼터 확인: https://aistudio.google.com/app/apikey

### "AI 어시스턴트 모듈을 찾을 수 없습니다"
```bash
# ai_chatbot_hybrid.py 파일이 있는지 확인
ls -la ai_chatbot_hybrid.py

# 없으면 이전 커밋에서 복원
git checkout HEAD ai_chatbot_hybrid.py
```

### 인덱싱이 너무 느림 (30초 이상)
- 데이터가 너무 큼 (2000행 제한)
- 인터넷 속도 확인
- Chroma Cloud 대신 로컬 Chroma 사용 고려

---

## 📊 성능 벤치마크 (참고)

정상 동작 시:

| 작업 | 예상 시간 |
|------|----------|
| 첫 인덱싱 | 5-10초 |
| 정량 계산 질문 | 0.5-1초 |
| 탐색 검색 질문 | 2-3초 |
| 필터 변경 | 5-10초 (재인덱싱) |
| 동일 필터 재질문 | 2-3초 (캐싱) |

---

## ✅ 테스트 완료!

모든 테스트를 통과했다면:
1. ✅ 1.5단계 하이브리드 시스템 정상 동작
2. ✅ 2단계 개발 준비 완료

다음 단계:
→ 2단계 개발 시작 (영구 맥락 + 비즈니스 규칙)

---

**테스트 날짜:** ____________
**테스트 결과:** ✅ 성공 / ⚠️ 부분 성공 / ❌ 실패
**메모:** _________________________________
