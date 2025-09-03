# Zero‑to‑Hero практикум — рабочая тетрадка

> Учебная тетрадка по мотивам курса **Andrej Karpathy — NN: Zero‑to‑Hero**.  
> Здесь — компактные конспекты + практические задания и полностью воспроизводимые
> мини‑реализации: *микро‑autograd*, *char‑LM (makemore)*, *внимание* и *минимальный Transformer‑блок*.
>
> **Формат:** обычный `Markdown`. Его можно читать на GitHub, а для запуска кода
> – просто копировать блоки в Python/Colab/Jupyter. При желании конвертируйте в
> ноутбук: `jupytext --to ipynb notebooks.md`.

---

## 0. Как пользоваться

1. Идёте по разделам сверху вниз.  
2. В каждом разделе есть **код**, **задания (✅)** и **вопросы для самопроверки (❓)**.  
3. Всё работает **без внешних данных** — мы генерируем игрушечные корпуса.

**Рекомендуемая версия Python:** 3.10+  
**Зависимости:** для PyTorch‑части (разделы 3–4) установите `torch`.

```bash
pip install torch jupytext  # опционально
```

---

## 1. Микро‑autograd: вычислительный граф и обратное распространение

В этом разделе реализуем **скалярную** автодифференциацию в духе *micrograd*.
Векторизацию и PyTorch подключим позже.

### 1.1. Узлы вычислительного графа

```python
import math

class Value:
    def __init__(self, data, _prev=(), _op='', label=''):
        self.data = float(data)
        self.grad = 0.0
        self._prev = set(_prev)
        self._op = _op
        self.label = label
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    # --- базовые операции ---
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data  * out.grad
        out._backward = _backward
        return out

    def __neg__(self):        # -a
        return self * -1
    def __sub__(self, other): # a - b
        return self + (-other)
    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self * other**-1
    def __pow__(self, p):
        assert isinstance(p, (int,float))
        out = Value(self.data**p, (self,), f'**{p}')
        def _backward():
            self.grad += (p * (self.data**(p-1))) * out.grad
        out._backward = _backward
        return out
    def tanh(self):
        t = (math.exp(2*self.data) - 1) / (math.exp(2*self.data) + 1)
        out = Value(t, (self,), 'tanh')
        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out

def topological_sort(root):
    """Обход графа для корректного backprop"""
    seen, order = set(), []
    def build(v):
        if v not in seen:
            seen.add(v)
            for child in v._prev: build(child)
            order.append(v)
    build(root)
    return order

def backward(loss):
    for v in topological_sort(loss)[::-1]:
        v._backward()
```
**Проверим градиенты** на простом выражении:

```python
# f = (a*b + c)^2
a, b, c = Value(2.0), Value(-3.0), Value(1.5)
f = (a*b + c)**2
# прямой прогон
f.grad = 1.0
backward(f)
print(a, b, c, f)
```

### ✅ Задания
1. Добавьте операции `exp`, `relu` и проверьте градиенты на выражении вашего выбора.
2. Реализуйте линейный слой `Linear1(x) = w*x + b` для скаляров (по одному нейрону).

### ❓ Самопроверка
- Чем вычислительный граф отличается от статической символической дифференциации?
- Почему нужен **топологический порядок** при backprop?

---

## 2. Мини‑MLP на нашем autograd

Соберём крошечную сеть и обучим её аппроксимировать простую функцию.

```python
import random

random.seed(0)

class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(0.0)

    def __call__(self, x):  # x: list[float] или list[Value]
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out

    def parameters(self):
        return self.w + [self.b]

class MLP:
    def __init__(self, nin, nout, nh=16):
        self.h = [Neuron(nin) for _ in range(nh)]
        self.out = [Neuron(nh) for _ in range(nout)]

    def __call__(self, x):
        h = [n(x) for n in self.h]
        y = [n(h) for n in self.out]
        return y

    def parameters(self):
        ps = []
        for n in self.h + self.out:
            ps.extend(n.parameters())
        return ps

# Игрушечные данные: y = sin(x0) + cos(x1)
def make_batch(bs=32):
    import math, random
    X, Y = [], []
    for _ in range(bs):
        x0, x1 = random.uniform(-3,3), random.uniform(-3,3)
        y = math.sin(x0) + math.cos(x1)
        X.append([Value(x0), Value(x1)])
        Y.append(Value(y))
    return X, Y

net = MLP(nin=2, nout=1, nh=16)

def mse(yhat, y):
    return sum(( (yh - yt)**2 for yh,yt in zip(yhat, y) ), Value(0.0)) / len(y)

# Тренировка
for step in range(200):
    X, Y = make_batch(32)
    Yhat = [net(x)[0] for x in X]
    loss = mse(Yhat, Y)

    # обнулить градиенты
    for p in net.parameters(): p.grad = 0.0
    loss.grad = 1.0
    backward(loss)

    # шаг оптимизации
    for p in net.parameters():
        p.data += -0.05 * p.grad

    if step % 50 == 0:
        print(f"step {step:03d} loss={loss.data:.4f}")
```

### ✅ Задания
1. Добавьте L2‑регуляризацию.  
2. Сравните таргет‑функции: `tanh` vs `relu`. Где сходится быстрее и почему?

---

## 3. Char‑LM (*makemore*): n‑граммы → MLP → сэмплинг

Теперь перейдём к **символьной языковой модели**. Начнём с n‑грамм и простой MLP.

### 3.1. Датасет (генерируем имена)

```python
import random, string

random.seed(1)
NAMES = ["anna","maria","pavel","boris","lena","olga","roman","dima","ivan","nina","lola","max","katya","petya","sara"]
# спец. символы начала/конца
BOS, EOS = "<", ">"

def build_vocab(names):
    chars = sorted(set("".join(names)))
    itos = {i+1:c for i,c in enumerate(chars)}  # 0 оставим под BOS
    itos[0] = BOS
    itos[len(itos)] = EOS
    stoi = {c:i for i,c in itos.items()}
    return stoi, itos

stoi, itos = build_vocab(NAMES)
vocab_size = len(itos)
vocab_size
```

### 3.2. Обучающие пары для биграмм

```python
def make_bigrams(names):
    pairs = []
    for name in names:
        s = BOS + name + EOS
        for ch_prev, ch_next in zip(s, s[1:]):
            pairs.append((stoi[ch_prev], stoi[ch_next]))
    return pairs

pairs = make_bigrams(NAMES)

# Матрица подсчётов (MLE)
import numpy as np
C = np.ones((vocab_size, vocab_size), dtype=np.float32)  # +1 — сглаживание
for i,j in pairs: C[i,j] += 1.0
P = C / C.sum(axis=1, keepdims=True)  # эмпирические вероятности перехода
```

### 3.3. Сэмплирование имён по биграммной модели

```python
import numpy as np

def sample_bigram(n=10):
    out = []
    for _ in range(n):
        s = BOS
        while True:
            i = stoi[s[-1]]
            j = np.random.choice(vocab_size, p=P[i])
            ch = itos[j]
            if ch == EOS: break
            s += ch
        out.append(s[1:])  # убрать BOS
    return out

print(sample_bigram(10))
```

### 3.4. MLP‑модель для (контекст→следующий символ)

```python
import torch, torch.nn as nn

K = 3  # длина контекста
# построим датасет (X: [N,K], Y: [N])
def build_dataset(names, K):
    X, Y = [], []
    for name in names:
        s = BOS + name + EOS
        ctx = [0]*K
        for ch in s:
            ix = stoi[ch]
            X.append(ctx[:])
            Y.append(ix)
            ctx = ctx[1:] + [ix]
    return torch.tensor(X), torch.tensor(Y)

Xtr, Ytr = build_dataset(NAMES, K)

model = nn.Sequential(
    nn.Embedding(vocab_size, 16),
    nn.Flatten(),
    nn.Linear(16*K, 64), nn.Tanh(),
    nn.Linear(64, vocab_size)
)

opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
lossf = nn.CrossEntropyLoss()

for step in range(2000):
    logits = model(Xtr)
    loss = lossf(logits, Ytr)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 400 == 0:
        print(step, loss.item())
```

**Сэмплирование с MLP:**

```python
import torch
def sample_mlp(n=10):
    res = []
    for _ in range(n):
        ctx = [0]*K
        out = ""
        while True:
            X = torch.tensor([ctx])
            logits = model(X)
            ix = torch.distributions.Categorical(logits=logits).sample().item()
            ch = itos[ix]
            if ch == EOS: break
            out += ch
            ctx = ctx[1:] + [ix]
        res.append(out)
    return res

print(sample_mlp(10))
```

### ✅ Задания
1. Поиграйте с `K` и размерами слоёв; посчитайте перплексию на обучении/валидации.  
2. Добавьте **dropout** и сравните качество.  
3. Реализуйте *temperature sampling* при генерации.  

---

## 4. Внимание (Scaled Dot‑Product) — игрушечный пример

Сделаем один слой **самовнимания** (self‑attention) поверх эмбеддингов.

```python
import torch, torch.nn as nn, math

torch.manual_seed(42)

B, T, C = 2, 5, 8  # batch, time, channels
x = torch.randn(B, T, C)

d_k = C
Wq = nn.Linear(C, C, bias=False)
Wk = nn.Linear(C, C, bias=False)
Wv = nn.Linear(C, C, bias=False)

Q = Wq(x)  # (B,T,C)
K = Wk(x)  # (B,T,C)
V = Wv(x)  # (B,T,C)

att = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)  # (B,T,T)
mask = torch.tril(torch.ones(T, T)) == 1
att = att.masked_fill(~mask, float('-inf'))  # каузальная маска
w = torch.softmax(att, dim=-1)               # веса внимания
y = w @ V                                     # (B,T,C)
y.shape
```

### ❓ Вопросы
- Зачем делить на `sqrt(d_k)`?  
- Чем **self‑attention** отличается от обычного внимания «запрос→память»?

---

## 5. Минимальный Transformer‑блок

Соберём **каузальный** однослойный блок: *Multi‑Head Attention → FFN → LayerNorm + residuals*.

```python
class Head(nn.Module):
    def __init__(self, C, head_size, T):
        super().__init__()
        self.key = nn.Linear(C, head_size, bias=False)
        self.query = nn.Linear(C, head_size, bias=False)
        self.value = nn.Linear(C, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(T, T)))
        self.scale = head_size ** -0.5
    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x); q = self.query(x); v = self.value(x)
        att = q @ k.transpose(-2,-1) * self.scale
        att = att.masked_fill(self.tril[:T,:T]==0, float('-inf'))
        w = att.softmax(dim=-1)
        return w @ v

class MultiHead(nn.Module):
    def __init__(self, C, num_heads, head_size, T):
        super().__init__()
        self.heads = nn.ModuleList([Head(C, head_size, T) for _ in range(num_heads)])
        self.proj  = nn.Linear(num_heads*head_size, C)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(out)

class Block(nn.Module):
    def __init__(self, C, num_heads, T, ff=4):
        super().__init__()
        head_size = C // num_heads
        self.sa = MultiHead(C, num_heads, head_size, T)
        self.ln1 = nn.LayerNorm(C)
        self.ffn = nn.Sequential(nn.Linear(C, ff*C), nn.ReLU(), nn.Linear(ff*C, C))
        self.ln2 = nn.LayerNorm(C)
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x
```

**Применим блок к char‑LM:**

```python
class TinyTransformer(nn.Module):
    def __init__(self, vocab, C=64, T=16, H=4):
        super().__init__()
        self.T = T
        self.token = nn.Embedding(vocab, C)
        self.pos   = nn.Embedding(T, C)
        self.block = Block(C=C, num_heads=H, T=T)
        self.lm    = nn.Linear(C, vocab)
    def forward(self, idx):
        B, T = idx.shape
        x = self.token(idx) + self.pos(torch.arange(T))
        x = self.block(x)
        return self.lm(x)  # (B,T,vocab)

# игрушечные данные для LM
def batch_lm(names, T=16, bs=32):
    alltext = BOS + (EOS+BOS).join(names) + EOS
    ids = torch.tensor([stoi[c] for c in alltext], dtype=torch.long)
    X, Y = [], []
    for i in range(len(ids)-T):
        X.append(ids[i:i+T])
        Y.append(ids[i+1:i+T+1])
    X = torch.stack(X); Y = torch.stack(Y)
    ix = torch.randint(0, len(X), (bs,))
    return X[ix], Y[ix]

torch.manual_seed(0)
model = TinyTransformer(vocab_size, C=64, T=16, H=4)
opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
lossf = nn.CrossEntropyLoss()

for step in range(800):
    X,Y = batch_lm(NAMES, T=16, bs=64)
    logits = model(X)
    loss = lossf(logits.view(-1, vocab_size), Y.view(-1))
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 200 == 0:
        print(step, loss.item())
```

### ✅ Задания
1. Сделайте **много голов** и измерьте влияние на лосс.  
2. Добавьте **dropout** в MHA и FFN.  
3. Реализуйте *weight tying* (`lm.weight = token.weight`).

---

## 6. Эксперименты и сравнение

| Модель | Длина контекста | Параметров | Вал. лосс* |
|---|---:|---:|---:|
| Биграммы (MLE) | 1 | ~`V^2` | — |
| MLP (K=3) | 3 | ~50k | … |
| TinyTransformer | 16 | ~110k | … |

\* Добавьте свою валидацию (рандомный сплит имен 80/20).

---

## 7. Вопросы для собеседования (короткий лист)

- Чем языковая модель отличается от seq2seq?  
- Почему **каузальная маска** важна для autoregressive‑LM?  
- Зачем нужен `LayerNorm` и residual‑соединения?  
- Что такое *exposure bias* при генерации?

---

## 8. Полезные команды

```bash
# Конвертация в ноутбук (по желанию)
pip install jupytext
jupytext --to ipynb notebooks.md

# Запуск локально
python - <<'PY'
print("Проверка окружения ок")
PY
```

---

## 9. Откуда идеи

- Курс **NN: Zero‑to‑Hero** (Andrej Karpathy) — рекомендуем пройти целиком.  
- Реализации в этой тетрадке написаны «с нуля» для учебных целей и
  не повторяют код курса дословно.

Удачи! 🚀
