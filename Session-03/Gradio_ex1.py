import gradio as gr

switch = 2

if switch == 1:
    ### example 1: simple greeting
    def greet(name):
        return "Hello " + name + "!"

    gr.Interface(fn=greet, inputs="text", outputs="text").launch()

elif switch == 2:
    ### example 2: simple calculator
    def calculate(num1, num2, operation):
        if operation == "add":
            return num1 + num2
        elif operation == "subtract":
            return num1 - num2
        elif operation == "multiply":
            return num1 * num2
        elif operation == "divide":
            if num2 == 0:
                return "Error: division by zero"
            return num1 / num2
        
    gr.Interface(fn=calculate,
        title="Simple Calculator",
        description="Enter two numbers and select an operation",
        inputs=[
            gr.Number(label="Number 1"),
            gr.Number(label="Number 2"),
            gr.Dropdown(["add", "subtract", "multiply", "divide"], label="Operation")
        ],
        outputs="text",
        examples=[
            [1, 2, "add"],
            [3, 4, "subtract"],
            [5, 6, "multiply"],
            [7, 8, "divide"]
        ]
    ).launch()

