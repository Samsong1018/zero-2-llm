# LLM skeleton (currently: MLP)

## Origin
This is me following Andrej Karpathy's [Neural Networks: Zero to Hero](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) series, so it will strongly resemble his [Micrograd](https://github.com/karpathy/micrograd/tree/master), and [Makemore](https://github.com/karpathy/makemore) (it will be almost identical). 
I plan to apply this to different projects in the future.

## Usage
### Bigram Engine with Embeddings
This follows part two of the makemore series, but using my tensors and numpy instead of torch functions. It's in the same folder as the bigram guesser, but it isn't a bigram model anymore: it predicts the next character from a context window of three characters and uses embeddings, which creates more human sounding names than the single character context seen in BigramEngine/main.py. The bigram model sees each character as a one-hot vector, while the new one looks each character up in a learned embedding table and gets a multidimensional vector for it. It includes multiple settings at the top.
```
BLOCK_SIZE = 3 # size of the context window (three characters)
EMBEDDING_DIMENSIONS = 10 # how many dimensions each character's embedding has
HIDDEN_LAYER = 300 # how many neurons the hidden layer contains
BATCH_SIZE = 32 # the batch size per optimization step
STEPS = 200000 # how many optimization steps are taken
```
The learning rate is 0.1 for the first 100,000 steps and 0.01 after that. Each run appends its settings, train/dev/test loss, and 20 generated names to BigramEngine/embedding_output.txt.

Multiple Tensor features were added as well: the shape property, so you can call Tensor.shape instead of Tensor.data.shape; division, exp, and log; tanh that works on whole arrays, unlike the Value class; and sum, which now takes axis and keepdims similar to pytorch. The Tensor now has a randn function that returns a Tensor of random values in the given shape (either separate numbers or a tuple). Indexing now passes gradients back, which is what lets the embedding table C[X] learn. The view function was implemented to reshape data for processing, and finally there's a cross entropy that uses log softmax to avoid overflowing.

### MLP with handmade Tensor class
After applying the MLP using the value class to the MNIST data set I realized it was painfully slow (about a quarter of the dataset in 10 hours), and started looking into how pytorch implemented features which made it considerably more efficient for general use cases. That's when I started working on the Tensor class to get the same effect with more efficiency.

The Tensor MLP works very similarly to the Value MLP, with the exception that it can run matrix multiplication and can run on the full datasets as opposed to individual items. The result is I could run 100 epochs for the full dataset (in batches of 32) in about 5min as opposed to the MLP using the value class. 

### MLP with Value class

The Neuron class initializes with the number of inputs and creates a random weight for each one and a bias that starts at 0, then can be called with your data set (i.e. x)

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

The MLP class takes the number of inputs and an array of layer sizes. The example below would create an MLP with 3 inputs and two hidden layers of four neurons each, which creates a single output. Finally, MLP takes an activation parameter to choose between tanh ("tanh") and sigmoid ("sig"); the default is tanh.


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
## Makemore MLP

This is a super simple bigram guesser similar to makemore, but I used my Tensor MLP design to guess the next character. weights.json was formed using a 27, [32, 32, 27] MLP and weightsSimplified.json was made with a 27, [27] MLP. Because we're using bigrams they create generally the same results. 
### More Coming Soon!