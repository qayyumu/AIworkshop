##asyncio library for writing asynchronous code
## operation used run and await to pause the execution of the code

import asyncio

async def dooperation(i):
    print("Starting Async operation...",i)
    await asyncio.sleep(1)
    print("Operation Async completed...",i)
    return f"Operation result of {i}"

async def main_operation():
    print("Main operation Started...")
    task1 = dooperation(1)
    task2 = dooperation(2)
    
    
    result1 = await task1
    print(f"Result: {result1}")
    
    result2 = await task2
    print(f"Result: {result2}")
   


print("Main program Started...")
asyncio.run(main_operation())



