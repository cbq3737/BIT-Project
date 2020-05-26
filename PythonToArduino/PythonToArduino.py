
#원하는 결과입력
import serial
import numpy as np
import time
import random



ser = serial.Serial(
    port='com6',
    baudrate=9600,
)
time.sleep(2)
while True:
    ball = int(input("볼 선택: "))
    want = int(input("score: "))

    if ball==1:
        xy = np.loadtxt('random_pingpong.csv', delimiter=',', dtype=np.int32)
    else :
        xy = np.loadtxt('random_soft.csv', delimiter=',', dtype=np.int32)
    x_data = xy[:, 0:-1]
    y_data = xy[:, [-1]]

    total = len(x_data)

    while True:
        ran = random.randint(0,total-1)
        # print(ran, " ", y_data[ran][0])
        if y_data[ran][0] == int(want):
            break

    op1 = x_data[ran][0]
    op2 = x_data[ran][1]

    print(op1," ",op2)
    ser.write(str(op1).encode())
    res1 = ser.readline()
    print(res1.decode()[:len(res1) - 1])

    ser.write(str(op2).encode())
    res2 = ser.readline()
    print(res2.decode()[:len(res1) - 1])







# import serial
# import numpy as np
# import time
# xy = np.loadtxt('first_data.csv', delimiter=',', dtype=np.float32)
# x_data = xy[:, 0:-1]
# y_data = xy[:, [-1]]
# ser = serial.Serial(
#     port='com6',
#     baudrate=9600,
# )
# time.sleep(2)
# start = 0
# finish = len(x_data)
# while True:
#     #print("insert op :")  2.7버전은 이대로 2.7이상은 ("insert op:",end=' ')
#     if start==finish:
#         break
#     op1 = x_data[start][0]
#     op2 = x_data[start][1]
#
#     # print(op1," ",op2)
#     ser.write(str(op1).encode())
#     res1 = ser.readline()
#     print(res1.decode()[:len(res1) - 1])
#
#     ser.write(str(op2).encode())
#     res2 = ser.readline()
#     print(res2.decode()[:len(res1) - 1])
#
#     # input()
#     # print("받음", res1.decode()[:len(res1) - 1], res2.decode()[:len(res2) - 1])
#     start+=1
#     # sleep(100)


