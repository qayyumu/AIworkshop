### shared resource and lock to synchronize the access to the resource
import asyncio

shared_resource = 0
lock = asyncio.Lock()

async def dooperation(i):
    print("Starting Async operation...",i)
    async with lock:
        global shared_resource
        shared_resource += 1
        print(f"Shared resource: {shared_resource}")
    await asyncio.sleep(1)
    print("Operation Async completed...",i)
    return f"Operation result of {i}"

async def main_operation():
    print("Main operation Started...")
    
    results = await asyncio.gather(dooperation(1),dooperation(2))
    
    for result in results:
        print(f"Result: {result}")
   


print("Main program Started...")
asyncio.run(main_operation())



