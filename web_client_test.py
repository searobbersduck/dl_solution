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

'''

before running this progress, do 'service mysqld restart'

'''

import http.client
# conn = http.client.HTTPConnection('127.0.0.1', port=8002)

# conn = http.client.HTTPConnection('yq01-idl-gpu-offline80.yq01.baidu.com', port=8002)

conn = http.client.HTTPConnection("face.baidu.com")

data = open('sample/3/16_right.jpeg', 'rb').read()

headers = {"Content-type": "image/jpeg", "Accept": "q=0.6, image/jpeg", "Content-Length": str(len(data))}

# conn.request('POST', "/test/for/medical", data, headers)
conn.request('GET', "/test/for/medical", data, headers)
# conn.request('POST', "", data, headers)
# conn.request('GET', "", data, headers)

r = conn.getresponse()

print('The image dr level is: {}'.format(r.headers['idx']))
print('The image dr level propobility is: {}'.format(r.headers['prop']))

image_uid = r.headers['image_uid']

print(image_uid)

headers = {"Content-type": "text/plain", "level":"3", "image_uid":image_uid}

# conn.request('GET', "", "", headers=headers)
conn.request('GET', "/test/for/medical", "", headers=headers)


# 这句似乎是必要的，'''r = conn.getresponse()'''，没有这句，服务器上不执行这一段
r = conn.getresponse()

print('end')

conn.close()
