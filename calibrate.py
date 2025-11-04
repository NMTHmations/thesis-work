from multicam import MotorClient
from time import sleep
import keyboard

client = MotorClient.MotorClient()
rotate_count = 0
space_count = 0

print("Calibration of DC motor has started")

sleep(1)

while space_count < 1:
    for i in range(0,10):
        client.moveMotor(100,True,1)
    rotate_count += 10
    sleep(0.003)
    if keyboard.is_pressed("space"):
        print(f"Actual Max step: {rotate_count}")
        space_count += 1
    elif keyboard.is_pressed("esc"):
        break