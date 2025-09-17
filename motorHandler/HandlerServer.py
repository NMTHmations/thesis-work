from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class MotorRevDuration(BaseModel):
    rev_count: int

@app.post("/rev/")
def rev_motor(item: MotorRevDuration):
    return {
        "rev duration": item.rev_count
    }