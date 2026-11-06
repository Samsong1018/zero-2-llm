from tensor import Tensor
import numpy as np

class Layer:
    def __init__(self, nin, nout, activation=True):
        self.W = Tensor(np.random.randn(nin, nout) * (1/nin) ** 0.5)
        self.activation = activation
        self.b = Tensor(np.zeros(nout))
        
    def __call__(self, x):
        out = x @ self.W + self.b
        return out.sigmoid() if self.activation else out

    def parameters(self):
        return [self.W, self.b]

class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [
            Layer(sz[i], sz[i+1], activation=True) for i in range(len(nouts))
            ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    
    def train(self, lr, batch_size, epochs, x, y, lrd=False, on_epoch_end=None):
        out_dim = self.layers[-1].W.data.shape[1]
        y_onehot = np.eye(out_dim)[y]

        params = self.parameters()
        history = []
        for epoch in range(epochs):
            if lrd:
                cur_lr = lr * 0.5 * (1 + np.cos(np.pi * epoch / epochs))
            else:
                cur_lr = lr
            total_cost, correct, seen = 0.0, 0, 0

            order = np.random.permutation(len(x))
            for step, s in enumerate(range(0, len(order), batch_size), start=1):
                idx = order[s:s + batch_size]
                xb = Tensor(x[idx])
                yb = Tensor(y_onehot[idx])

                out = self(xb)
                loss = ((out - yb) ** 2).sum() * (1/len(idx))

                for p in params:
                    p.zero_grad()
                loss.backward()
                for p in params:
                    p.data -= cur_lr * p.grad

                total_cost += loss.data * len(idx)
                seen += len(idx)
                correct += (np.argmax(out.data, axis=1) == y[idx]).sum()

                if step % 300 == 0:
                    print(f"  step {step}: cost {total_cost / seen:.4f} | acc {correct / seen:.1%}")

            avg_cost, acc = total_cost / seen, correct / seen
            history.append({"epoch": epoch, "cost": avg_cost, "acc": acc, "lr": cur_lr})
            print(f"epoch {epoch}: cost {avg_cost:.4f} | acc {acc:.1%} | lr {cur_lr:.4f}")

            if on_epoch_end is not None:
                on_epoch_end()

        return history
