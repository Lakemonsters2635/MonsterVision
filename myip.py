
import socket

#print([l for l in ([ip for ip in socket.gethostbyname_ex(socket.gethostname())[2] if not ip.startswith("127.")][:1], [[(s.connect(("8.8.8.8", 53)), s.getsockname()[0], s.close()) for s in [socket.socket(socket.AF_INET, socket.SOCK_DGRAM)]][0][1]]) if l][0][0])

h = socket.gethostname()
ip = socket.gethostbyname(h)
print(ip)

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
x = s.connect(('10.26.35.2', 1))  # connect() for UDP doesn't send packets
print (x)
print (s.getsockname())
