import socket
from motorController import motorController

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
        msg = payload.rstrip(b'\x00').decode()
        print("Message:", msg)
        parts = msg.split(';')
        controller.moveMotor(int(parts[0]), bool(parts[1]), float(parts[2]))
    except UnicodeDecodeError:
        print("Non-text payload")