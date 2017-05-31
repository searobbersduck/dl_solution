import http.server
import socketserver

from http.server import BaseHTTPRequestHandler
from http import HTTPStatus


from PIL import Image

from io import BytesIO

import io

PORT = 8002

import test_dr


test_dr.import_test()

classifier = test_dr.get_classifier()

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
        # print(self.rfile.read())
        # f=open('2.jpg','wb')
        # f.write(data1)
        stream = io.BytesIO(data1)
        img = Image.open(stream)
        # print(img.show())
        idx = classifier.classifyImage(img)
        print('dr image is level: {}'.format(idx))
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-type", 'text/plain')
        self.send_header("idx", 2)
        self.end_headers()
        # self.wfile.write(idx)

    def do_POST(self):
        data1 = self.rfile.read(int(self.headers['Content-Length']))
        # print(self.rfile.read())
        # f=open('3.jpg','wb')
        # f.write(data1)
        stream = BytesIO(data1)
        img = Image.open(stream)
        # print(img.size())
        idx = classifier.classifyImage(img)
        print('dr image is level: {}'.format(idx))
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-type", 'text/plain')
        self.send_header("idx", str(idx))
        self.end_headers()



Handler = ImageHTTPRequestHandler

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print("serving at port", PORT)
    httpd.serve_forever()



# 问题解决：http://stackoverflow.com/questions/20689958/cannot-get-response-body-from-post-request-python
    # def do_POST(self):
    #     content = bytes("TEST RESPONSE", "UTF-8")
    #     self.send_response(200)
    #     self.send_header("Content-type", "text/plain")
    #     self.send_header("Content-Length", len(content))
    #     self.end_headers()
    #     print(self.rfile.read(int(self.headers['Content-Length'])).decode("UTF-8"))
    #     self.wfile.write(content)