from multicam import MotorClient
from time import sleep

client = MotorClient.MotorClient()

for i in range(50,100):
    client.moveMotor(i,False,duration=1)
    sleep(0.1)