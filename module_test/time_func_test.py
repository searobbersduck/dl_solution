import time

ticks = time.time()

print(ticks)

local_time = time.localtime(ticks)

print(local_time)

print(time.asctime(time.localtime(time.time())))