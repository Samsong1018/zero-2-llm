# LLM skeleton (currently: GPT)

## Origin
This is me following Andrej Karpathy's [Neural Networks: Zero to Hero](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) series, so it will strongly resemble his [Micrograd](https://github.com/karpathy/micrograd/tree/master) and [Makemore](https://github.com/karpathy/makemore) (it will be almost identical). The main difference is I'm using my own Tensor class and numpy instead of torch.
I've also started applying it to my own projects, the first one being a 2x2 Rubik's cube solver.

## Setup
All you need is Python 3 and numpy. Run the scripts from the repo root, for example:
```
python3 makemore/embedding_complex.py
```
Each script adds the root folder to its path so it can find `tensor.py`, `nn.py` and `mlp.py`.

Everything runs on the CPU by default. If you have an NVIDIA card and CuPy installed (`pip install cupy-cuda13x`) you can run on the GPU instead:
```
MLP_DEVICE=gpu python3 GPT/Shakespear/gpt_shakespear.py
```
The GPT script sets this to gpu on its own, so change it at the top of the file if you don't have CuPy.

## Layout
```
tensor.py     my Tensor class (autograd on top of numpy arrays)
backend.py    picks numpy or cupy for tensor.py and nn.py
nn.py         layer classes and the Adam optimizer
mlp.py        simple MLP built on Tensor (used for MNIST and the bigram model)
micrograd/    the Value class and the MLP built on it
mnist/        MNIST training with the Value MLP and the Tensor MLP
makemore/     name generators, from bigram up to the WaveNet style model
rubiks/       2x2 Rubik's cube solver (my own project)
GPT/          character level GPT trained on Shakespeare (in Shakespear/)
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
- `+ - * / ** @`, plus `exp`, `log`, `tanh`, `sigmoid` and `relu`
- `sum`, `mean`, `var` and `std`, which take axis and keepdims like pytorch (var and std take a correction too)
- `shape`, `view`, `squeeze` and `transpose` for reshaping, and `Tensor.cat` to join tensors
- indexing that passes gradients back, which is what lets the embedding table learn
- `Tensor.randn` for random tensors in a given shape (separate numbers or a tuple)
- `softmax`, and `cross_entropy`, which uses log softmax so it doesn't overflow
- `batch_norm` and `layer_norm` with their own backward passes, so they don't build a huge graph
- `no_grad()` for evaluating without building a graph

## nn.py
Layer classes that copy the way pytorch's are set up: `Linear`, `BatchNorm1d`, `LayerNorm`, `Tanh`, `ReLU`, `Dropout`, `Embedding`, `Flatten` and `Sequential`. `BatchNorm1d` keeps a running mean and variance for when `training` is set to False. `Flatten(n)` groups every n characters together, which is what the WaveNet style model uses. `Dropout` only drops while `Dropout.training` is True, so it gets turned off during eval.

There's also an `Adam` optimizer, which the GPT uses instead of plain gradient descent.

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

## 2x2 Rubik's cube (rubiks/)
This is the first thing I built on my own with all of this. A 2x2 only has 3,674,160 possible states, so `build_dataset.py` does a BFS out from the solved cube and saves every state with its distance from solved and the move that gets it one step closer. The dataset is too big for the repo, so run that first:
```
python3 rubiks/build_dataset.py
python3 rubiks/2b2.py
```
`cube.py` has the moves and the model, which is an embedding for each sticker's color then nine Linear, BatchNorm and Tanh layers of 512. It predicts which move to make next (or stop). `2b2.py` trains it in stages, starting with cubes 1 move away and working up to 11 (the furthest any 2x2 can be from solved). The full run took about 10 hours on my CPU.

Just picking the best move every time only solved 13% of the depth 11 cubes, so the solver uses a beam search instead. It keeps the K best partial solutions after each move, and if any of them hits solved it's done. With K=200 it solved everything I tested it on.

| K (beam width) | d5 | d7 | d8 | d9 | d10 | d11 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 90% | 74% | 33% | 24% | 17% | 9% |
| 10 | 100% | 99% | 97% | 83% | 63% | 48% |
| 50 | 100% | 100% | 99% | 100% | 99% | 94% |
| 200 | 100% | 100% | 100% | 100% | 100% | 100% |

`test_network.py` runs those tests and `solve.py` opens a little tkinter GUI where you can scramble or paint the cube yourself and watch it solve.

## GPT (GPT/Shakespear/)
Part seven of the series, a character level GPT that trains on `input.txt` (all of tiny Shakespeare) and writes new Shakespeare. It's Andrej's model rebuilt on my Tensor class, 6 blocks of multi head self attention and feed forward, 6 heads, 384 embedding dimensions and a context of 256 characters. That comes out to about 10.8 million parameters.

This is what `backend.py` was for. Running it on numpy would've taken forever, so tensor.py and nn.py can swap to CuPy and run on the GPU without changing anything else.

```
python3 GPT/Shakespear/gpt_shakespear.py
```
It trains for up to 15,000 steps with a cosine learning rate and checks the loss every 1,000. It saves the weights whenever val loss hits a new best, and stops early if it doesn't improve for 3 checks in a row. Every check goes to `training_log.txt`, and when it's done it generates text with the best weights and appends it to `output.txt`.

The best val loss so far is 1.486 (train was 1.11, so it's starting to overfit by the end). To get more text out of the saved weights without training again, run `python3 GPT/Shakespear/sample.py`. A sample from `output.txt`:
```
MARCIUS:
I cannot me would not to seek for a woman!

First Senator:
A fitten i' the tremity; come to choose his grave,
justice and breeds with poison spark, but like these nurses,
```
Not exactly Shakespeare but it's got the format down.

### More Coming Soon!
