import threading

def dooperation(args):
    print("Starting thread...",args )
    for i in range(10):
        print(f" {args} Thread: {i}")
    print("Thread completed...",args)

thread1 = threading.Thread(target=dooperation,args=(1,))
thread2 = threading.Thread(target=dooperation,args=(2,))
thread1.start()
thread2.start()
thread1.join()
thread2.join()
print("Main program completed...")