# LLM skeleton (currently: makemore)

## Origin
This is me following Andrej Karpathy's [Neural Networks: Zero to Hero](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) series, so it will strongly resemble his [Micrograd](https://github.com/karpathy/micrograd/tree/master) and [Makemore](https://github.com/karpathy/makemore) (it will be almost identical). The main difference is I'm using my own Tensor class and numpy instead of torch.
I plan to apply this to different projects in the future.

## Setup
All you need is Python 3 and numpy. Run the scripts from the repo root, for example:
```
python3 makemore/embedding_complex.py
```
Each script adds the root folder to its path so it can find `tensor.py`, `nn.py` and `mlp.py`.

## Layout
```
tensor.py     my Tensor class (autograd on top of numpy arrays)
nn.py         layer classes used for makemore parts three and four
mlp.py        simple MLP built on Tensor (used for MNIST and the bigram model)
micrograd/    the Value class and the MLP built on it
mnist/        MNIST training with the Value MLP and the Tensor MLP
makemore/     name generators, from bigram up to the WaveNet style model
```

## MLP with Value class (micrograd/)

The Neuron class initializes with the number of inputs and creates a random weight for each one and a bias that starts at 0, then can be called with your data set (i.e. x)

```
x = [2.0, 3.0, 4.0]
n = Neuron(3)
n(x)
```

The Layer class takes the inputs per neuron and total neurons, so (3, 5) is five neurons with 3 inputs each, resulting in an array of five values.

```
x = [2.0, 3.0, 4.0]
l = Layer(3, 5)
l(x) # this would return an array of five output values
```

The MLP class takes the number of inputs and an array of layer sizes. The example below would create an MLP with 3 inputs and two hidden layers of four neurons each, which creates a single output. MLP also takes an activation parameter to choose between tanh ("tanh") and sigmoid ("sig"), the default is tanh.

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

## MNIST (mnist/)
`mnist_24.npz` is MNIST shrunk to 24x24, so each image is flattened into 576 inputs. `train_value.py` uses the Value MLP and `train_tensor.py` uses the Tensor MLP. Both save their weights to json and the progress goes in `train.log`.

After running the Value MLP on MNIST I realized it was painfully slow (about a quarter of the dataset in 10 hours), and started looking into how pytorch made things considerably more efficient. That's when I started working on the Tensor class.

The Tensor MLP works very similarly to the Value MLP, except it does matrix multiplication and runs on whole batches instead of individual items. I could run 100 epochs of the full dataset (in batches of 32) in about 5 min.

## Tensor class (tensor.py)
It wraps a numpy array and tracks gradients the same way Value does, just for whole arrays at once. Broadcasted adds and multiplies send gradients back in the right shape. Currently it has:
- `+ - * / ** @`, plus `exp`, `log`, `tanh` and `sigmoid`
- `sum`, `mean`, `var` and `std`, which take axis and keepdims like pytorch (var and std take a correction too)
- `shape`, `view` and `squeeze` for reshaping
- indexing that passes gradients back, which is what lets the embedding table learn
- `Tensor.randn` for random tensors in a given shape (separate numbers or a tuple)
- `cross_entropy`, which uses log softmax so it doesn't overflow
- `batch_norm` with its own backward pass, so it doesn't build a huge graph
- `no_grad()` for evaluating without building a graph

## nn.py
Layer classes that copy the way pytorch's are set up: `Linear`, `BatchNorm1d`, `Tanh`, `Embedding`, `Flatten` and `Sequential`. `BatchNorm1d` keeps a running mean and variance for when `training` is set to False. `Flatten(n)` groups every n characters together, which is what the WaveNet style model uses.

## makemore (makemore/)
All of these train on `names.txt` and try to make up new names. The embedding scripts split the names 80/10/10 into train/dev/test and have their settings at the top. Each run appends its settings, the losses and 20 generated names to `embedding_output.txt`.

### Bigram (bigram.py)
This is a super simple bigram guesser, but using my Tensor MLP to guess the next character. weights.json was formed using a 27, [32, 32, 27] MLP and weightsSimplified.json was made with a 27, [27] MLP. Because they're both bigram models they create generally the same results.

### Embeddings (embedding_mlp.py, part two)
This isn't a bigram model anymore: it predicts the next character from a context window of three characters and uses embeddings, which creates more human sounding names. The bigram model sees each character as a one-hot vector, while this one looks each character up in a learned embedding table and gets a multidimensional vector for it. The learning rate is 0.1 for the first 100,000 steps and 0.01 after that.

### Batch norm (embedding_improved.py, part three)
Same model, but with batch norm written out by hand and better initialization. The hidden layer weights are scaled by (5/3)/sqrt(fan_in) and the output weights are scaled down so the starting loss isn't crazy high. I dropped the first bias since batch norm subtracts it right back out.

### Using classes (embedding_classes.py, part three)
The part three model rebuilt with `nn.py` and `Sequential`, five hidden layers of Linear, BatchNorm and Tanh. It evaluates in chunks so it doesn't run out of memory on the full dataset.

### WaveNet style (embedding_complex.py, part four)
It uses a context of 8 characters, and instead of squishing all of them into one layer it merges them two at a time using `Flatten(2)`, so it goes 8 → 4 → 2 → 1 over three layers.

### Results so far
| Model | Dev loss |
| --- | --- |
| Embeddings (part two) | 2.17 |
| Batch norm by hand (part three) | 2.10 |
| Using classes (part three) | 2.08 |
| WaveNet style (part four) | 2.02 |

### More Coming Soon!
