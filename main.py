from fastapi import FastAPI
from pydantic import BaseModel 

# fastAPI Init
app = FastAPI()

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
    rec['response'] = chat.text
    return rec



