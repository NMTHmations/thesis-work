import socket
from motorController import motorController
import traceback

controller = motorController()
INTERFACE = "eth0"
ETH_TYPE_CUSTOM = b'\x88\xb5'
s = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.htons(0x0003))
s.bind((INTERFACE, 0))

while True:
    packet, addr = s.recvfrom(65535)
    # Skip Ethernet header (14 bytes)
    if packet[12:14] != ETH_TYPE_CUSTOM:
        continue
    payload = packet[14:]
    # strip zero padding
    try:
        msg = payload.rstrip(b"\x00").decode('utf-8', errors='ignore')
        print("Message:", msg)
        parts = msg.split(';')
        controller.moveMotor(int(parts[0]), bool(parts[1]), float(parts[2].rstrip(b"\x00Rz").rstrip(b"\x00")))
    except UnicodeDecodeError:
        traceback.print_exc()
        print("Non-text payload")