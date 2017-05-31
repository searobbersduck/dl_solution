# import http.client
#
# conn = http.client.HTTPConnection("127.0.0.1", port=8001)
#
# conn.request("GET", "/DS000F4J.JPG")
#
# r1 = conn.getresponse()
#
# print(r1.status, r1.reason)
#
# header = r1.getheaders()
#
# print(header)
#
# data1 = r1.read()
#
# file = open('1.jpg','wb')
#
# file.write(data1)
#
# print(len(data1))
#
# conn2 = http.client.HTTPConnection("127.0.0.1", port=8002)
#
# import urllib.parse
# # params = urllib.parse.urlencode({'@number': 12524, '@type': 'issue', '@action': 'show'})
# # headers = {"Content-type": "application/x-www-form-urlencoded", "Accept": "text/plain"}
#
# headers = {"Content-type": "application/x-www-form-urlencoded", "Accept": "text/plain", "Content-Length": str(len(data1))}
#
# # conn2.request('GET', "", data1, headers)
# conn2.request('POST', "", data1, headers)
#
# r2 = conn2.getresponse()
#
# print(r2.status, r2.reason)
#
# print('The image dr level is: {}'.format(r2.headers['idx']))
#
# # data2 = r2.read()
#
# conn.close()
# conn2.close()


import http.client

conn = http.client.HTTPConnection('127.0.0.1', port=8002)

data = open('2.jpg', 'rb').read()

headers = {"Content-type": "image/jpeg", "Accept": "q=0.6, image/jpeg", "Content-Length": str(len(data))}

conn.request('POST', "", data, headers)

r = conn.getresponse()

print('The image dr level is: {}'.format(r.headers['idx']))
