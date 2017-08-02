import http.server
import socketserver

from http.server import BaseHTTPRequestHandler
from http import HTTPStatus


from PIL import Image

from io import BytesIO

import io

PORT = 8002

import test_dr
from utils import mdb
import imagehash
import os


test_dr.import_test()

classifier = test_dr.get_classifier()

db, cursor = mdb.start_db_conn()
cursor.execute('select * from dr_image_tb')
print(cursor.fetchall())
# cmd_inserttb = """
#     insert into dr_image_tb values ('1341234', '/home/1.jpeg', 0, 2)
# """
# cursor.execute(cmd_inserttb)
# cursor.execute('select * from dr_image_tb')
# print(cursor.fetchall())

# image_root = '/home/data/zhizhen'
image_root = './zhizhen'

class ImageHTTPRequestHandler(BaseHTTPRequestHandler):

    """Simple HTTP request handler with GET and HEAD commands.

    This serves files from the current directory and any of its
    subdirectories.  The MIME type for files is determined by
    calling the .guess_type() method.

    The GET and HEAD requests are identical except that the HEAD
    request omits the actual contents of the file.

    """

    def do_GET(self):
        print('Content type: {0}'.format(self.headers['Content-type']))
        if self.headers['Content-type'] == 'image/jpeg':
            self._classify()
        elif self.headers['Content-type'] == 'text/plain':
            print('begin outside _doctor_confirm')
            self._doctor_confirm()

    def do_POST(self):
        data1 = self.rfile.read(int(self.headers['Content-Length']))
        # print(self.rfile.read())
        # f=open('3.jpg','wb')
        # f.write(data1)
        stream = BytesIO(data1)
        img = Image.open(stream)
        # print(img.size())
        idx,prop,prop1 = classifier.classifyImage(img)
        print('dr image is level: {}'.format(idx))
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-type", 'text/plain')
        self.send_header("idx", str(idx))
        self.send_header("prop", str(prop1))
        self.end_headers()

    def _classify(self):
        data1 = self.rfile.read(int(self.headers['Content-Length']))
        stream = BytesIO(data1)
        img = Image.open(stream)
        image_id = imagehash.average_hash(img)
        idx,prop,prop1 = classifier.classifyImage(img)
        print(prop1)
        cmd_query = """select * from dr_image_tb where id='{0}'""".format(image_id)
        cursor.execute(cmd_query)
        query = cursor.fetchall()
        assert len(query) <= 1
        if len(query) == 0:
            imagepath = os.path.join(image_root, '{}.jpeg'.format(image_id))
            img.save(imagepath)
            cmd_insert = """insert into dr_image_tb (id, imagepath, algolevel) values ('{0}', '{1}', {2})""".format(
                image_id, imagepath, idx
            )
        else:
            cmd_insert = """update dr_image_tb set algolevel={0} where id='{1}'""".format(
                idx, image_id
            )
        cursor.execute(cmd_insert)
        cursor.execute('commit')

        print('dr image is level: {}'.format(idx))
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-type", 'text/plain')
        self.send_header("idx", str(idx))
        self.send_header("prop", str(prop1))
        self.send_header("image_uid", str(image_id))
        self.end_headers()

    def _doctor_confirm(self):
        doctor_confirm_level = int(self.headers['level'])
        image_uid = self.headers['image_uid']
        cmd_inserttb = """update dr_image_tb set doctorlevel={0} where id='{1}'""".format(doctor_confirm_level, image_uid)
        cursor.execute(cmd_inserttb)
        cursor.execute('select * from dr_image_tb')
        print(cursor.fetchall())
        cursor.execute('commit')
        print('end inside _doctor_confirm')



Handler = ImageHTTPRequestHandler

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    try:
        print("serving at port", PORT)
        httpd.serve_forever()
    finally:
        cursor.execute('commit')

mdb.close_db_conn(db)



# 问题解决：http://stackoverflow.com/questions/20689958/cannot-get-response-body-from-post-request-python
    # def do_POST(self):
    #     content = bytes("TEST RESPONSE", "UTF-8")
    #     self.send_response(200)
    #     self.send_header("Content-type", "text/plain")
    #     self.send_header("Content-Length", len(content))
    #     self.end_headers()
    #     print(self.rfile.read(int(self.headers['Content-Length'])).decode("UTF-8"))
    #     self.wfile.write(content)