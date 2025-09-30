import socket
import threading
from motorController import motorController

HOST = "0.0.0.0"
PORT = 2000

controller = motorController()

def handle_client(conn, addr):
    print(f"[+] New connection from {addr}")
    with conn:
        while True:
            try:
                data = conn.recv(1024)
                if not data:
                    break
                parts = data.decode(errors='ignore').split(';')
                print(f"[{addr}] speed:{parts[0]} direction:{parts[1]} duration:{parts[2]}")
                controller.moveMotor(int(parts[0]), bool(parts[1]), float(parts[2]))
                conn.sendall(data)
            except (BrokenPipeError, ConnectionResetError):
                print(f"[!] Client {addr} disconnected unexpectedly")
                break
    print(f"[-] Connection closed: {addr}")

def start_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen()
    print(f"[*] Server listening on {HOST}:{PORT}")

    while True:
        conn, addr = server.accept()
        thread = threading.Thread(target=handle_client, args=(conn, addr))
        thread.daemon = True
        thread.start()

if __name__ == "__main__":
    start_server()