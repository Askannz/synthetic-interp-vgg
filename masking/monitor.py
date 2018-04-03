import socket
import numpy as np
import cv2

PORT = 8888
BUFSIZE = 32000


def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


class Sender:

    def __init__(self):

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.address = ('localhost', PORT)

    def send(self, img):

        data = img.flatten().tostring()
        self.socket.sendto(b'\x0f', self.address)

        for c in chunks(data, BUFSIZE):
            self.socket.sendto(c, self.address)

        self.socket.sendto(b'\xff', self.address)


class Receiver:

    def __init__(self):

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        address = ('localhost', PORT)
        self.socket.bind(address)

    def receive_wait(self):

        header = b""
        while header != b'\x0f':
            header = self.socket.recv(BUFSIZE)

        data = b""
        msg = self.socket.recv(BUFSIZE)
        while msg != b'\xff':
            data += msg
            msg = self.socket.recv(BUFSIZE)

        img = np.fromstring(data, dtype=np.uint8)
        return img


if __name__ == "__main__":

    receiver = Receiver()

    while True:

        img = receiver.receive_wait().reshape(224, 224, 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)
