from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os 
from langchain.llms import OpenAI
from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain_core import caches
from langchain_core.caches import BaseCache
from langchain.cache import InMemoryCache
from pydantic import BaseModel
from langchain.chains import create_sql_query_chain
from langchain_openai import ChatOpenAI
from openai import OpenAI
import json

# fastAPI Init
app = FastAPI()


chatbot_tokenizer = None
chatbot_model = None


# BaseCache를 상속받아 SimpleCache 구현
class SimpleCache(BaseCache):
    def __init__(self):
        self.cache = {}

    def lookup(self, key: str):
        return self.cache.get(key)

    def update(self, key: str, value: str):
        self.cache[key] = value

    def clear(self):
        self.cache = {}


# dev
host = os.environ['host']
port = os.environ['port']
username = os.environ['username']
password = os.environ['projectPassword']
database_schema = os.environ['default_schema']
mysql_uri = f"mysql+pymysql://{username}:{password}@{host}:{port}/{database_schema}"
@app.on_event("startup")

async def startup_event():
    global chatbot_tokenizer, chatbot_model
    # 초기화 작업 코드 작성
    print("애플리케이션 초기화 중...")
    chatbot_model_path = "./fine_tuned_model2"
    chatbot_tokenizer = AutoTokenizer.from_pretrained(chatbot_model_path)
    chatbot_model = AutoModelForSequenceClassification.from_pretrained(chatbot_model_path)
    # 모델 load 

class ChatbotDTO:
    class Post (BaseModel):
        text: str
    

@app.get("/")
def read_root():
    return {"hello":"World"}


# model 1 실행. 
# String만 받는걸로 프로트
@app.post("/chat")
async def divTextSql(chat: ChatbotDTO.Post): 
    rec = {}

    # 토크나이저로 입력 데이터 처리
    inputs = chatbot_tokenizer(chat.text, padding=True, truncation=True, return_tensors="pt")

    # 모델 추론 (Gradient 계산 비활성화)
    with torch.no_grad():
        outputs = chatbot_model(**inputs)
        logits = outputs.logits

    # 로짓 값을 확률로 변환 (Softmax)
    probs = torch.nn.functional.softmax(logits, dim=-1)

    # 가장 높은 확률의 클래스를 선택
    predicted_classes = torch.argmax(probs, dim=1)

    # 1일 경우 
    if predicted_classes == 0:
        # 0일 경우 평서문 출력.
        client = OpenAI()

        system_role = '''
            당신은 물류센터 관리자입니다.
            물류센터에서 사용되는 용어와 개념을 정확히 알려줘야 합니다. 
            응답은 다음의 형식을 지켜주세요.
            {"입력텍스트":\"입력된 텍스트\",
            "response":\"답변 내용\"}
        '''

        response = client.chat.completions.create(
            model = 'gpt-3.5-turbo',
            messages = [
                {
                    "role":"system",
                    "content":system_role
                },
                {
                    "role":"user",
                    "content":chat.text
                }
            ]
        )

        response_text = response.choices[0].message.content

        response_json = json.loads(response_text)

        rec['response'] = response_json.get("response", "응답을 처리할 수 없습니다.")
    else :
        
        db = SQLDatabase.from_uri(mysql_uri, sample_rows_in_table_info=2)
        
        # 테스트
        # db = SQLDatabase.from_uri("sqlite:///chinook.db")
        
        # 언어 모델 설정
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        # 체인 생성
        chain = create_sql_query_chain(llm, db)
        
        # 질문 결과 및 실행
        response = chain.invoke({"question": chat.text})
        print(response)
        
        # SQL 쿼리 결과 실행 및 결과 출력
        result = db.run(response)

         # 결과가 비어 있을 경우 처리
        if not result:
            formatted_result = "결과 없음"
        else:
            # 리스트 내부의 튜플에서 첫 번째 요소만 추출하여 문자열로 변환
            formatted_result = result 

        # 0일 경우 평서문 출력.
        client = OpenAI()

        system_role = '''
            당신은 물류센터 상담원입니다.
            주어진 질문과 결과값을 바탕으로 문장을 만들어서 자연스럽게 전달해야합니다. 
            응답은 다음의 형식을 지켜주세요.
            {"입력텍스트":\"입력된 텍스트\",
            "response":\"답변 내용\"}
        '''

        request_question = f"질문: {chat.text}, 결과값: {formatted_result}"

        response = client.chat.completions.create(
            model = 'gpt-3.5-turbo',
            messages = [
                {
                    "role":"system",
                    "content":system_role
                },
                {
                    "role":"user",
                    "content":request_question
                }
            ]
        )
       # 응답을 문자열로 받음
        response_text = response.choices[0].message.content
        
        response_json = json.loads(response_text)
    
        # "response" 키의 값만 추출
        rec['response'] = response_json.get("response", "응답을 처리할 수 없습니다.")
    
    # 결과 출력
    return rec



