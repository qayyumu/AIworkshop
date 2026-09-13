### create task and await for the result
import asyncio

async def dooperation(i):
    print("Starting Async operation...",i)
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



