# imagehash --> reference: https://pypi.python.org/pypi/ImageHash

from PIL import Image
import imagehash

hash = imagehash.average_hash(Image.open('../2.jpg'))

print(hash)


# mysql --> reference:

# mac install reference: http://www.cnblogs.com/zhjsll/p/5702950.html
# mac install: brew install mysql
# mysql.server start
# mysql -uroot

# install python3 mysql: pip install PyMySQL
# install python2 mysql pip install MySQL-python
# note: when install mysql, be sure to stop mysql server service: 'mysql.server stop'

# start mysql server on mac: mysql.server start
# start mysql server on centos6: sudo service mysqld start, reference: https://www.linode.com/docs/databases/mysql/how-to-install-mysql-on-centos-6

import pymysql

cmd_showdb = 'show databases'
cmd_createdb = 'create database if not exists dr_images_db'
cmd_usedb = 'use dr_images_db'
cmd_showtb = 'show tables'
cmd_createtb = """create table if not exists dr_image_tb (
    id char(20) not null,
    imagepath char(200) not null,
    algolevel int,
    doctorlevel int
)"""


db = pymysql.connect('localhost', 'root')

cursor = db.cursor()

cursor.execute(cmd_showdb)

data = cursor.fetchall()

print(data)

cursor.execute(cmd_createdb)

cursor.execute(cmd_showdb)

data = cursor.fetchall()

print(data)

cursor.execute(cmd_usedb)

cursor.execute(cmd_showtb)

data = cursor.fetchall()

print(data)

cursor.execute(cmd_createtb)

cursor.execute(cmd_showtb)

print(cursor.fetchall())

cmd_inserttb = """
    insert into dr_image_tb values ('1341234', '/home/1.jpeg', 0, 2)
"""

cmd_inserttb1 = """
    insert into dr_image_tb (id, imagepath) values ('13412345', '/home/2.jpeg')
"""

cmd_selecttb = """
    select * from dr_image_tb where id='13412345'
"""

cursor.execute(cmd_inserttb)
cursor.execute(cmd_inserttb1)
cursor.execute(cmd_selecttb)

data = cursor.fetchall()

assert len(data) <= 1

query = data[0]

pre_id = query[0]
pre_path = query[1]
pre_algolvl = query[2]
pre_doctorlvl = query[3]

print(query)

cmd_update = """update dr_image_tb set algolevel={0}, doctorlevel={1}""".format(4,4)
cursor.execute(cmd_update)
print(cursor.fetchall())
cursor.execute(cmd_selecttb)
print(cursor.fetchall())






