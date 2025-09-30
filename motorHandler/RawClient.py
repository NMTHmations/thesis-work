import socket
import time

INTERFACE = "eth0"
dst_mac = b'\x26\x6e\x90\x85\x4e\x9a'
src_mac = b'\x2c\xcf\x67\x9d\x12\x4d'
eth_type = b'\x88\xb5'
payload = b"0;1;0.5"


min_payload = 46
if len(payload) < min_payload:
    payload = payload + b'\x00' * (min_payload - len(payload))

frame = dst_mac + src_mac + eth_type + payload

s = socket.socket(socket.AF_PACKET, socket.SOCK_RAW)
s.bind((INTERFACE, 0))

count = 100
delay = 0.001

sent = 0
start = time.time()
try:
    for i in range(count):
        n = s.send(frame)
        if n == 0:
            print("Warning: send returned 0 bytes")
        sent += 1
finally:
    s.close()
end = time.time()
print(f"Sent {sent} frames in {end-start:.3f} s ({sent/(end-start):.1f} fps)")
