##asyncio library for writing asynchronous code
## operation used run and await to pause the execution of the code

import asyncio

async def dooperation():
    print("Starting Async operation...")
    await asyncio.sleep(1)
    print("Operation Async completed...")
    return "Operation result"

async def main_operation():
    print("Main operation Started...")
    task = dooperation()
    
    result = await task
    print(f"Result: {result}")
    print("Main operation completed...")


print("Main program Started...")
asyncio.run(main_operation())



