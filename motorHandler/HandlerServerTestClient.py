import socket

HOST = "127.0.0.1"  # server IP
PORT = 2000         # server port

def client():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))  # connect to server
        print(f"Connected to {HOST}:{PORT}")

        # Send some data
        string = b"1;True;0.3"
        s.sendall(string)

        # Receive echo back
        data = s.recv(1024)
        
        print(data.decode() == string.decode())

if __name__ == "__main__":
    client()