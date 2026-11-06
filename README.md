# LLM skeleton (currently: MLP)

## Origin
This is me following Andrej Karpathy's Neural [Networks Zero to Hero](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) series, so it will strongly resemble his [Micrograd](https://github.com/karpathy/micrograd/tree/master), and [Makemore](https://github.com/karpathy/makemore) (it will be almost identical). 
I plan to apply this to different projects in the future.

## Usage

The neuron class initiates with the amount of inputs and creates random weights and biases for each, then can be called with your data set (i.e. x)

```
x = [2.0, 3.0, 4.0]
n = Neuron(3)
n(x)
```

The Layer class takes the input per neuron and total neurons so (3, 5) is five neurons with 3 each resulting in an array of five values.

```
x = [2.0, 3.0, 4.0]
l = Layer (3, 5)
l(x) # this would return an array of five output values
```

The MLP class takes number of inputs and array of layer size. The below example would create an MLP with 3 inputs two hidden layers with four neurons each which creates a single output.


```
x = [2.0, 3.0, 4.0]
ntw = MLP(3, [4, 4, 1])
ntw(x)
```

Andrej's example:

```
n = MLP(3, [4, 4, 1])
LR = 0.5

# dataset
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0]
]
#goal
ys = [1.0, -1.0, -1.0, 1.0]

# Running MLP for each data and its goal
for k in range(20):
    # forward pass
    ypred = [n(x) for x in xs]
    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))

    # backward pass
    for p in n.parameters():
        p.grad = 0.0
    loss.backward()

    # nudge in opposite direction of gradient by learning rate
    for p in n.parameters():
        p.data += -LR*p.grad
    
    print(k, loss.data)
```


### More Coming Soon!