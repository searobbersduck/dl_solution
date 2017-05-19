import http.server
import socketserver

from http.server import BaseHTTPRequestHandler


from PIL import Image

from io import BytesIO

import io

PORT = 8002


class ImageHTTPRequestHandler(BaseHTTPRequestHandler):

    """Simple HTTP request handler with GET and HEAD commands.

    This serves files from the current directory and any of its
    subdirectories.  The MIME type for files is determined by
    calling the .guess_type() method.

    The GET and HEAD requests are identical except that the HEAD
    request omits the actual contents of the file.

    """
    def do_GET(self):
        data1 = self.rfile.read()
        print(self.rfile.read())
        f=open('2.jpg','wb')
        f.write(data1)
        stream = io.BytesIO(data1)
        img = Image.open(stream)
        print(img.show())

    def do_POST(self):
        data1 = self.rfile.read()
        print(self.rfile.read())
        f=open('3.jpg','wb')
        f.write(data1)
        stream = BytesIO(data1)
        img = Image.open(stream)
        print(img.size())



Handler = ImageHTTPRequestHandler

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print("serving at port", PORT)
    httpd.serve_forever()