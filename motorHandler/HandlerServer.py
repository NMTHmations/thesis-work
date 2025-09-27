from fastapi import FastAPI
from pydantic import BaseModel
from motorController import motorController

app = FastAPI()

motorControll = motorController()

class MotorRevDuration(BaseModel):
    speed: int
    direction: bool
    duration: float

@app.post("/rev/")
def rev_motor(item: MotorRevDuration):
    if item.speed > 100:
        return {
            "message:": "Speed should not be exceed 100!"
        }
    motorControll.moveMotor(item.speed,item.direction,item.duration)
    return {
        "message": "ok"
    }