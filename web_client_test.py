import http.client

conn = http.client.HTTPConnection("127.0.0.1", port=8001)

conn.request("GET", "/DS000F4J.JPG")

r1 = conn.getresponse()

print(r1.status, r1.reason)

header = r1.getheaders()

print(header)

data1 = r1.read()

file = open('1.jpg','wb')

file.write(data1)

print(len(data1))

conn2 = http.client.HTTPConnection("127.0.0.1", port=8002)

import urllib.parse
params = urllib.parse.urlencode({'@number': 12524, '@type': 'issue', '@action': 'show'})
headers = {"Content-type": "application/x-www-form-urlencoded", "Accept": "text/plain"}

conn2.request('GET', "", data1, headers)
