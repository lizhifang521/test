# 数据集：[['man', 'women'], ['black', 'white'], ['king', 'queen'], ['girl', 'boy'], ['up', 'dowan'], ['high', 'low'], ["love", "hate"]]
# 4.正确的导入必须的包，并通过给定的数据集构建出：单词—>下标，下标->单词两个字典，并在两个字典中加入S，E，P（注：S表开始，E表结束，P表填充）
# 9.正确的构建输入和标签
# 10.对下x，y进行处理，填充到统一长度
# 11.文本向量化处理
import numpy as np
import torch
from torch import nn,optim

data = [['man', 'women'], ['black', 'white'], ['king', 'queen'], ['girl', 'boy'], ['up', 'dowan'], ['high', 'low'], ["love", "hate"]]
corpus = 'SEPabcdefghijklmnopqrstuvwxyz'
word2id = {w: i for i,w in enumerate(corpus)}
id2word = {i: w for i,w in enumerate(corpus)}
vocab_size = len(word2id)
hidden_size = 256
maxlen = max([max(len(i[0]),len(i[1])) for i in data])

def make_data(data):
    encoder_input,decoder_input,labels = [],[],[]
    for i in data:
        encoder_input.append(np.eye(vocab_size)[[word2id[j] for j in i[0]] + [word2id['P']] * (maxlen - len(i[0]))])
        decoder_input.append(np.eye(vocab_size)[[word2id['S']] + [word2id[j] for j in i[1]] + [word2id['P']] * (maxlen - len(i[1]))])
        labels.append([word2id[j] for j in i[1]] + [word2id['P']] * (maxlen - len(i[1])) + [word2id['E']])
    return torch.Tensor(encoder_input),torch.Tensor(decoder_input),torch.LongTensor(labels)

encoder_input,decoder_input,labels = make_data(data)

# 5.正确的搭建网络模型结构seq2seq
# 6.正确构建编码器和解码器使用lstm
# 7.正确定义出参数，要求使用双层的lstm
class Seq2Seq(nn.Module):
    def __init__(self):
        super(Seq2Seq, self).__init__()
        self.encoder = nn.LSTM(input_size=vocab_size,hidden_size=hidden_size,num_layers=2,dropout=0.5,batch_first=True)
        self.decoder = nn.LSTM(input_size=vocab_size,hidden_size=hidden_size,num_layers=2,dropout=0.5,batch_first=True)
        self.layer = nn.Linear(in_features=hidden_size*2,out_features=vocab_size)
    def forward(self,encoder_input,decoder_input):
        encoder_out,states = self.encoder(encoder_input)
        out = []
        # 8.在编解码模型的基础上加入Attention机制进行调优
        for i in range(maxlen + 1):
            decoder_out,states = self.decoder(decoder_input[:,i,:].unsqueeze(1),states)
            attention_weights = torch.softmax(torch.matmul(encoder_out,decoder_out.transpose(1,2)),dim=1)
            attention_vector = torch.matmul(encoder_out.transpose(1,2),attention_weights).squeeze()
            context_vector = torch.concat([attention_vector,decoder_out.squeeze()],dim=-1)
            out.append(self.layer(context_vector))
        return torch.stack(out)

model = Seq2Seq()

# 12.定义代价和优化器，并且进行梯度的裁剪
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(lr=0.01,params=model.parameters())

# 13.循环5000次，每100次输出一下代价
for i in range(5000):
    optimizer.zero_grad()
    h = model(encoder_input,decoder_input)
    loss = criterion(h.permute(1,2,0),labels)
    loss.backward()
    optimizer.step()
    if (i + 1) % 100 == 0:
        print(loss.item())

# 14.展示并且输出最后结果
x_data = [['man', 'P']]
encoder_input,decoder_input,_ = make_data(x_data)
y_pre = model(encoder_input,decoder_input)
y_pre = torch.argmax(y_pre,dim=-1).squeeze().detach().numpy()
print(''.join([id2word[i] for i in y_pre]).replace('E', '').replace('S', ''))


