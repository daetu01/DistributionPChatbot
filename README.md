# 🗨️ 도심 물류 최적화 Chatbot (AI 기반) 

## 📌 프로젝트 개요
도심 물류 최적화 시스템의 사용자 편의를 높이기 위해 AI 기반의 Chatbot을 개발하였습니다. 
이 Chatbot은 실시간 데이터 질의 및 자연어 응답을 제공하여, 물류 데이터 접근성을 향상시키고, 다양한 사용자 문제를 해결하는 데 도움을 줍니다.

---

## 🧐 문제 정의
- **다양한 사용자 문제**: 비전문가부터 전문가까지 물류 데이터를 필요로 하는 사용자가 다양함
- **복잡한 데이터 접근성**: 물류 시스템 데이터는 복잡한 쿼리가 필요하며 접근이 어려움
- **고객 불만사항**: 데이터 요청 대기 시간이 길어 실시간 답변이 필요함

---

## ✅ 해결 방안
Google 경량화 BERT 모델을 활용한 Fine-tuning을 통해 질문 유형을 분류하고, 최적의 응답 방식을 결정합니다. 훈련 데이터로는 **HuggingFace의 공개 데이터셋**을 활용하였습니다.

1️⃣ **질문 유형 분류** (Google 경량화 BERT Fine-tuning) - 용량 문제로 인해 모델 파일은 별도로 제공하지 않음. 
   - **평서문**: 일반적인 설명을 요구하는 질문
   - **SQL 질의 필요**: 데이터베이스 조회가 필요한 질문

2️⃣ **질문 유형에 따른 응답 프로세스**
   - **평서문** → OpenAI API를 통해 자연어 답변 생성 🧠
   - **SQL 질의** → LangChain의 `SQLAgent`를 활용하여 **쿼리문 작성 & DB 조회** 후 OpenAI API를 통해 자연어로 변환하여 응답 🏗️

---

## 🔄 Chatbot 응답 프로세스
```
사용자 질문 → BERT 기반 질문 유형 분류
└──> 평서문 → OpenAI API 응답
└──> SQL 질의 → LangChain SQLAgent → DB 조회 → OpenAI API를 통해 자연어 응답
```

---

## 🛠️ 기술 스택
- **훈련 데이터**: HuggingFace 공개 데이터셋
- **AI 모델**: Google 경량화 BERT (Fine-tuning)
- **자연어 처리**: OpenAI API (ChatGPT 기반 응답 생성)
- **데이터베이스 질의**: LangChain `SQLAgent`
- **Backend**: FastAPI
- **Database**: MySQL

---

## 🚀 주요 기능
- **Google BERT 기반 질문 유형 분류**
- **SQL 질의 및 LangChain을 통한 자동 쿼리 생성 & 조회**
- **OpenAI API를 활용한 자연어 응답 생성**
- **실시간 데이터 분석 및 답변 제공**
- **사용자 피드백을 반영한 지속적인 모델 개선**

---

## ⚠️ 한계점
- **서비스 통합 답변 미반영**: 아직 시스템 전반적인 DB 통합이 이루어지지 않음
- **한정된 답변**: 특정 도메인 내에서만 응답 가능
- **데이터 유출 우려**: 외부 API(OpenAI API) 사용으로 인한 데이터 보안 문제
- **느린 응답 속도**: OpenAI API 및 LangChain 연산으로 인한 응답 지연 가능
- **복잡한 쿼리문 불가**: 다중 테이블 조인이 필요한 복잡한 SQL 문 실행 불가

---

## 🔄 개선 방안
- **통합 데이터베이스 활성화**: 시스템 완성 후 DB 통합을 통해 모든 서비스에 걸친 답변 제공
- **고도화된 LangChain 구성**: LangGraph 활용으로 더욱 다양하고 전문적인 답변 제공 기대
- **데이터 유출 방지**: Open LLM 모델을 활용하여 외부 API 의존도를 줄이고 보안 강화
- **응답 속도 개선**: Vector 임베딩을 통한 유사도 검색을 적용하여 정형화된 응답을 빠르게 제공
- **복잡한 SQL 질의 처리**: Open LLM Model 기반의 학습된 TTS(TextToSQL)을 적용하여 고난이도 쿼리 처리 가능

---

## 🔄 지속적인 업데이트
이 프로젝트는 지속적으로 개선 및 업데이트가 진행되고 있습니다.
- 기능 추가 및 성능 개선
- 데이터베이스 확장 및 최적화
- 새로운 AI 모델 적용 가능성 검토

---

## 📂 프로젝트 구조
```bash
📦 chatbot_project
├── 📁 train
│   ├── ChatBot.ipynb # Google BERT 훈련 주피터 노트북 파일
├── .gitignore
├── Dockerfile
├── main.py
├── requirements_fastapi.txt
├── README.md  # 프로젝트 설명 파일
```

---

## ⚡ 설치 및 실행 방법
### 1️⃣ 프로젝트 클론
```bash
git clone https://github.com/your-repository/DistributionPChatbot.git
cd DistributionPChatbot
```

### 2️⃣ 가상환경 설정 및 패키지 설치
```bash
python -m venv venv
source venv/bin/activate  # (Windows: venv\Scripts\activate)
pip install -r requirements_fastapi.txt
```

### 3️⃣ FastAPI 서버 실행
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 4️⃣ 테스트 요청 예시 (Postman 또는 CURL 사용)
```bash
curl -X POST "http://localhost:8000/chat" -H "Content-Type: application/json" -d '{"question": "현재 서울의 물류량은?"}'
```

---

## 📊 시스템 모니터링
- **로그 분석**: FastAPI 내 로깅 시스템 활용
- **DB 성능 모니터링**: MySQL Query 분석

---
