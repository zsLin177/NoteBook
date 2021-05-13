# pytorch代码编写

## 损失函数

* `torch.nn.``CrossEntropyLoss`(*weight: Optional[torch.Tensor] = None*, *size_average=None*, *ignore_index: int = -100*, *reduce=None*, *reduction: str = 'mean'*)

  The input is expected to contain raw, unnormalized scores for each class. 不需要去做softmax，直接将分数输入

  用于单标签分类

  input:(N,C)，target:(C)

* 多标签分类

  `torch.nn.``MultiLabelSoftMarginLoss`(*weight: Optional[torch.Tensor] = None*, *size_average=None*, *reduce=None*, *reduction: str = 'mean'*)

  
## 数据的处理，加载

* Field(supar)

  定义一个域来处理数据集中的一个列。

  比如怎么把一句话的这一列转换为一个tensor；还有要不要这一列要不要建词典，用于之后数字化

  supar中比如：

  ```python
  WORD = Field('words', pad=pad, unk=unk, lower=True)
  FEAT = SubwordField('chars', pad=pad, unk=unk, fix_len=args.fix_len)
  FEAT = SubwordField('bert',pad=tokenizer.pad_token,unk=tokenizer.unk_token,fix_len=args.fix_len,tokenize=tokenizer.tokenize)
  EDGE = ChartField('edges', use_vocab=False, fn=CoNLL.get_edges)
  LABEL = ChartField('labels', fn=CoNLL.get_labels)
  # field.build(dataset):从dataset建立一个词典
  FEAT.build(train)
  
  ```
  
  chartfield：







* Transform(supar)

  transform是由一些field组成的类，这些field对数据中的对应的列进行处理，针对不同结构的数据，组成不同的transform。

  比如CoNLL是transform的一个子类，有十列fields = ['ID', 'FORM', 'LEMMA', 'CPOS', 'POS', 'FEATS', 'HEAD', 'DEPREL', 'PHEAD', 'PDEPREL']

  ```python
  transform = CoNLL(FORM=(WORD, FEAT), PHEAD=(EDGE, LABEL))
  transform = CoNLL(FORM=WORD, CPOS=FEAT, PHEAD=(EDGE, LABEL))
  transform.load(path) # 读取数据集，返回[conll_sentence],每个conll_sentence包含代表一句话，
  					 # conll_sentence有一个values=[(),(),...],每一个()包含一列的所有内容
  ```







* Dataset(supar)

  加载数据和数字化、生成dataloader（以此生成batch）都在这里面

  ```python
  train = Dataset(transform, args.train) # 第二个参数是数据的路径
  ```

---
# 生成tensor

* x = torch.rand(2,3,5)

  返回tensor的size为[2,3,5]

* torch.ones_like(x)

  ```python
  x = torch.rand(5, 10, 600)
  y = torch.ones_like(x[..., :1])
  z = torch.ones_like(x)
  print(z.shape)
  print(y.shape)
  >>
  torch.Size([5, 10, 600])
  torch.Size([5, 10, 1])
  
  torch.ones(5, 10, 600)
  
  ```

---

# Biaffine

```python
def forward(self, x, y):
        r"""
        Args:
            x (torch.Tensor): ``[batch_size, seq_len, n_in]``.
            y (torch.Tensor): ``[batch_size, seq_len, n_in]``.

        Returns:
            ~torch.Tensor:
                A scoring tensor of shape ``[batch_size, n_out, seq_len, seq_len]``.
                If ``n_out=1``, the dimension for ``n_out`` will be squeezed automatically.
        """

        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)
        # [batch_size, n_out, seq_len, seq_len]
        s = torch.einsum('bxi,oij,byj->boxy', x, self.weight, y)
        # remove dim 1 if n_out == 1
        s = s.squeeze(1)

        return s
```
---
# Mask

* ***把pad的mask掉***

  ```python
  x = torch.tensor([[2, 1, 432, 52, 0], [2, 5, 234, 0, 0]]) # [2,5]
  mask = x.ne(0) # [2,5]
  >>mask
  tensor([[ True,  True,  True,  True, False],
          [ True,  True,  True, False, False]])
  mask1 = mask.unsqueeze(1) #[2,1,5]
  mask2 = mask.unsqueeze(2) #[2,5,1]
  >>mask1
  tensor([[[ True,  True,  True,  True, False]],
  
          [[ True,  True,  True, False, False]]])
  >>mask2
  tensor([[[ True],
           [ True],
           [ True],
           [ True],
           [False]],
  
          [[ True],
           [ True],
           [ True],
           [False],
           [False]]])
  mask3 = mask2 & mask1 #[2,5,5] 五行五列
  >>mask3
  tensor([[[ True,  True,  True,  True, False],
           [ True,  True,  True,  True, False],
           [ True,  True,  True,  True, False],
           [ True,  True,  True,  True, False],
           [False, False, False, False, False]],
  
          [[ True,  True,  True, False, False],
           [ True,  True,  True, False, False],
           [ True,  True,  True, False, False],
           [False, False, False, False, False],
           [False, False, False, False, False]]])
  
  ```

* ***把每句话的第一个词 bos mask掉***

  ```python
  >>mask
  tensor([[[ True,  True,  True,  True, False],
           [ True,  True,  True,  True, False],
           [ True,  True,  True,  True, False],
           [ True,  True,  True,  True, False],
           [False, False, False, False, False]],
  
          [[ True,  True,  True, False, False],
           [ True,  True,  True, False, False],
           [ True,  True,  True, False, False],
           [False, False, False, False, False],
           [False, False, False, False, False]]])
  mask[:, 0] = 0 #“:”对第0维度的所有进行操作，[:, 0]对所有二维矩阵的第0行赋值0
  >>mask
  tensor([[[False, False, False, False, False],
           [ True,  True,  True,  True, False],
           [ True,  True,  True,  True, False],
           [ True,  True,  True,  True, False],
           [False, False, False, False, False]],
  
          [[False, False, False, False, False],
           [ True,  True,  True, False, False],
           [ True,  True,  True, False, False],
           [False, False, False, False, False],
           [False, False, False, False, False]]])
  >>print(mask[1, 1])
  tensor([ True,  True,  True, False, False])
  ```

* ***set the indices larger than num_embeddings to unk_index***

  ```python
  if hasattr(self, 'pretrained'):
              ext_mask = words.ge(self.word_embed.num_embeddings) # 生成一个mask，size与words相同，在words中大于self.word_embed.num_embeddings的地方，对应mask中位置为True，别的位置为False
              ext_words = words.masked_fill(ext_mask, self.unk_index) # words不变，根据ext_mask来将位置为True的位置设置为self.unk_index，返回新的ext_words
              
  ```

* ***pack_padded_sequence, pad_packed_sequence(待研究)***

```python
x = pack_padded_sequence(embed, mask.sum(1), True, False)
x, _ = self.lstm(x)
x, _ = pad_packed_sequence(x, True, total_length=seq_len)
x = self.lstm_dropout(x)
```

* ***根据mask去取值***

  ```python
  >>mask.shape
  [145,22,22]
  >>s_egde.shape
  [145,22,22,2]
  >>egdes.shape
  [145,22,22]
  x = s_egde[mask]
  # x:[k,2]
  y = edges[mask]
  # y:[k]   k是mask中为True的数量
  ```
  
* ***supar的Metric多看看***

  ```python
  def __call__(self, preds, golds):
          pred_mask = preds.ge(0)
          gold_mask = golds.ge(0)
          span_mask = pred_mask & gold_mask
          self.pred += pred_mask.sum().item()
          self.gold += gold_mask.sum().item()
          self.tp += (preds.eq(golds) & span_mask).sum().item()
          self.utp += span_mask.sum().item()
          return self
  ```

---
# nn.Embedding

* ```python
  embed = nn.Embedding(num_embeddings=3, embedding_dim=10)
  trans = torch.randint(0, 3, (4, 3))
  x = embed(trans)
  print(x.shape) #[4, 3, 10]
  ```

---

# torch.sum

* ```python
  x = torch.ones(1, 2, 3, 10)
  x = torch.sum(x, dim=2) 
  print(x.shape)
  >>[1, 2, 10]
  ```
---
# expand

* ```python
  y = torch.ones(1, 1, 10)
  y_ = y.expand(2, -1, -1)
  # 表示第0维变成2，别的是-1，表示不变(复制)
  ```
---
# torch.gather()

* `torch.``gather`(*input*, *dim*, *index*, ***, *sparse_grad=False*, *out=None*) → Tensor

  - **input** ([*Tensor*](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – the source tensor
  - **dim** ([*int*](https://docs.python.org/3/library/functions.html#int)) – the axis along which to index
  - **index** (*LongTensor*) – the indices of elements to gather
  - **sparse_grad** ([*bool*](https://docs.python.org/3/library/functions.html#bool)*,**optional*) – If `True`, gradient w.r.t. `input` will be a sparse tensor.
  - **out** ([*Tensor*](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)*,* *optional*) – the destination tensor

  ```python
  x = torch.cat([x, null_hidden], dim=1)
  # x:[batch_size, seq_len+1, hidden_size*2]
  idx = torch.randint(0, seq_len+1, (batch_size, transition_num, id_num))
  # idx:[batch_size, transition_num, id_num]
  # 我希望得到的是[batch_size, transition_num, id_num, hidden_size*2], 也就是说，batch对应到batch，用id去抽取
  idx = idx.view(batch_size, transition_num*id_num).unsqueeze(-1).expand(-1,-1, hidden_size*2)
  states_hidden = torch.gather(x, 1, idx)
  # [batch_size, transition_num*d_num, hidden_size*2]
  states_hidden = states_hidden.view(batch_size, transition_num, id_num,-1)
  ```

  idx的维度的数量必须和输入源的维度数量一致，最终的输出size和idx的size是一样的。

  值得多看


---
# Optimizer&Scheduler

* ***clip gradient(待研究)***

  ```python
  nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
  ```
---
# masked_fill和tensor切片的一些坑

* 切片

  这些切片实际上并没有新建一个内存块出来，只是一个视图，按步长去跳给你看，实际上还是就只存了那一块地方的东西，你改了切出来的东西，源头也变了

  ```python
  states = torch.ones(2, 8, 4, 4) * (-1)
  windowed = states[..., :2]
  win_states = windowed[:, :, 0:3, :]
  ```

* masked_fill

  ```python
  null_lstm_mask = win_states.eq(-1)
  win_states.masked_fill_(null_lstm_mask, 10)
  # !!!这个就地操作改了states，也改了windowed，如果之后还要用到就有可能会发生错误，
  # 不能使用就地操作，应该：
  win_states = win_states.masked_fill(null_lstm_mask, 10)
  ```

* 以后建议都不要轻易用就地操作，即使你觉得没关系，不需要梯度，改了也没事的地方，不到万不得已不用就地。


---

# 索引改的一个操作

```python
>>> lr_action
tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
>>> arc
tensor([0, 2])
# 我需要把第0行的第0列，第1行的第2列改成1
>>> lr_action[torch.range(0,1,dtype=torch.long),arc]=1
>>> lr_action
tensor([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0.]])
```
---
# 得到向量中值为某个值的索引
* 先mask.eq(value)
```python
torch.nonzero(torch.tensor([1, 1, 1, 0, 1]), as_tuple=True)
```
---
# torch.unbind(dim=0)
用来将原来的tensor按照维度dim拆分成由小tensor组成的tuple