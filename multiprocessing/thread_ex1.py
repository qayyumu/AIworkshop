import threading

def dooperation():
    print("Starting thread...")
    for i in range(10):
        print(f"Thread: {i}")
    print("Thread completed...")

thread = threading.Thread(target=dooperation)
thread.start()
thread.join()
print("Main program completed...")