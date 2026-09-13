##asyncio library for writing asynchronous code
## operation used run and await to pause the execution of the code

import asyncio

async def dooperation():
    print("Starting Async operation...")
    await asyncio.sleep(1)
    print("Operation Async completed...")
    return 1

print("Main program Started...")
### directly calling the function will not work and will return coroutine object
dooperation()

## async loop for running the code
asyncio.run(dooperation())
print("Main program completed...")

