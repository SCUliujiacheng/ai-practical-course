"""B16—B30 作者源；算法始终展开写入每例 solution.py 和 Notebook。"""
from textwrap import dedent
import argparse
import ast
from course_builder import write_case

parser=argparse.ArgumentParser(description='生成 B16—B30，--ids 可只重建待修复案例')
parser.add_argument('--ids',default='')
args=parser.parse_args()
SELECTED={int(value.removeprefix('B')) for value in args.ids.split(',') if value} if args.ids else None

COMMON = '''
import json
import time
import numpy as np
started = time.perf_counter()
def finish(metrics):
    metrics["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
def barplot(labels, values, title, filename="comparison.png"):
    fig, ax = plt.subplots()
    ax.bar(labels, values)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename)
    plt.close(fig)
'''

def ex(title, prompt, answer, starter=None):
    answer=ast.unparse(ast.parse(dedent(answer).strip()))
    checks=['# '+line.strip() for line in answer.splitlines() if line.strip().startswith('assert ')]
    scaffold='# 本单元可直接使用上方载入的本例函数与数据。\n# TODO：先写预期结果，再实现题目要求。\n\n# 完成后取消下面检查的注释；运行空白练习不代表已完成。\n'+'\n'.join(checks)
    return dict(title=title, prompt=prompt, starter=starter or scaffold, answer=answer)

def make(n, slug, title, intro, sections, exercises, sources, prereqs, pitfalls, data_note=None, extra_files=None):
    if SELECTED is not None and n not in SELECTED:return None
    # 展开作者稿中的紧凑语句，使学习者看到逐行的标准 Python 控制流。
    sections=[(md,ast.unparse(ast.parse(dedent(code).strip()))) for md,code in sections]
    sections[0]=(sections[0][0],dedent(COMMON).strip()+'\n\n'+sections[0][1])
    return write_case('B',n,slug,title,intro,sections,exercises,sources,prereqs,pitfalls,
        decision='先检查 outputs/metrics.json 和实际图表。说明基线、改善或退步、代价和适用条件；不以一次运行的分数承诺真实业务效果。',
        data_note=data_note or '由本例完整代码生成的教学模拟数据；不是企业生产记录。随机种子、生成公式与划分方式均可检查。',extra_files=extra_files)

make(16,'手写神经网络训练','怎样亲手写出神经网络的训练过程？',
'任务：从带噪声的传感器读数学习非线性关系。你将看清预测、损失、梯度、更新四步，而不是把 fit 当作黑箱。',[
('## 1. 把数据变成形状明确的 Tensor\n输入 x 是 N×1，目标 y 也是 N×1。保留测试集，只用训练集学习；同形状避免广播偷偷产生 N×N 的损失。Python 的切片用于划分，float32 是本例的计算类型。','''
import torch
from torch import nn
torch.set_num_threads(2)
torch.manual_seed(16)
rng = np.random.default_rng(16)
x = rng.uniform(-2, 2, (240, 1)).astype('float32')
y = (0.8*x*x + 0.5*x + rng.normal(0, 0.12, x.shape)).astype('float32')
x_train, y_train = torch.from_numpy(x[:160]), torch.from_numpy(y[:160])
x_test, y_test = torch.from_numpy(x[160:]), torch.from_numpy(y[160:])
assert x_train.shape == y_train.shape == (160, 1)
np.savez(DATA_DIR / 'sensor.npz', x=x, y=y)
print('训练形状：', x_train.shape)
'''),
('## 2. 从一个导数到网络参数\nrequires_grad=True 告诉 PyTorch 记录运算。先核对 (w−3)² 在 w=1 的梯度为 −4，再建立 1→24→1 网络；ReLU 提供非线性，参数不是手工指定的答案。','''
w = torch.tensor(1.0, requires_grad=True)
((w-3)**2).backward()
assert w.grad.item() == -4.0
model = nn.Sequential(nn.Linear(1,24), nn.ReLU(), nn.Linear(24,1))
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.025)
print('可训练参数：', sum(p.numel() for p in model.parameters()))
'''),
('## 3. 写出每一次更新\nzero_grad 清掉上次梯度，backward 计算当前梯度，step 更新参数。缺少清零会累加梯度。每次循环都记录训练损失，但不据测试损失挑选轮数。','''
losses = []
for epoch in range(220):
    model.train()
    prediction = model(x_train)
    loss = criterion(prediction,y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(float(loss.detach()))
assert np.isfinite(losses).all() and losses[-1] < losses[0]
'''),
('## 4. 在未参与训练的数据上检查\neval 切换推理模式，no_grad 不建立梯度图。把网络与只猜训练均值的基线比较；保存权重时同时保留模型结构。观察拟合曲线比只看最终损失更直观。','''
model.eval()
with torch.no_grad():
    predicted = model(x_test).numpy().ravel()
baseline = np.full(len(predicted),float(y_train.mean()))
mse = float(np.mean((predicted-y[160:,0])**2))
baseline_mse = float(np.mean((baseline-y[160:,0])**2))
torch.save(model.state_dict(),OUT_DIR / 'network.pt')
fig,axes=plt.subplots(1,2,figsize=(10,4))
axes[0].plot(losses);axes[0].set_title('训练损失');axes[0].set_xlabel('更新次数')
order=np.argsort(x[160:,0]);axes[1].scatter(x[160:,0],y[160:,0],s=12,label='观测')
axes[1].plot(x[160:,0][order],predicted[order],color='red',label='预测');axes[1].legend()
fig.tight_layout();fig.savefig(OUT_DIR/'training.png');plt.close(fig)
finish({'test_mse':mse,'mean_baseline_mse':baseline_mse,'train_first_loss':losses[0],'train_last_loss':losses[-1]})
''')],[
ex('补全：核对梯度与一步更新','对 (a−5)² 在 a=2 求导，按学习率 0.1 手动更新，断言新值为 2.6。','''a=torch.tensor(2.,requires_grad=True)
((a-5)**2).backward()
with torch.no_grad(): a -= 0.1*a.grad
assert abs(a.item()-2.6)<1e-6
print(a.item())'''),
ex('修改：改为绝对误差','用当前预测计算 MAE，并与直接 NumPy 计算核对；解释它为何不像 MSE 一样放大大误差。','''with torch.no_grad():
    mae=float(nn.L1Loss()(model(x_test),y_test))
assert np.isclose(mae,np.abs(predicted-y[160:,0]).mean())
assert mae>=0
print('测试 MAE',mae)'''),
ex('迁移：学习线性计量关系','另生成 y=3x−2，无噪声。只用一个 Linear 层训练，最后检查斜率、截距；不要复用非线性答案。','''torch.manual_seed(2)
xx=torch.linspace(-1,1,80).reshape(-1,1); yy=3*xx-2
linear=nn.Linear(1,1); opt=torch.optim.SGD(linear.parameters(),lr=0.15)
for _ in range(150):
    opt.zero_grad(); loss=((linear(xx)-yy)**2).mean();loss.backward();opt.step()
assert abs(linear.weight.item()-3)<0.02 and abs(linear.bias.item()+2)<0.02
print(linear.state_dict())''')],
[('PyTorch：优化参数','https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html')],
'B13 数组与 B07 回归；补充 Tensor、自动求导、for 循环。','训练损失小不能证明泛化；不得把测试集用于调轮数。传感器公式是教学假设。')

make(17,'手写数字与数据加载','怎样用神经网络识别手写数字？',
'任务：把 8×8 手写数字转成分类结果，理解 Dataset、DataLoader 和批训练，并保存错误样例。',[
('## 1. 读图像并划分样本\n使用 sklearn 随包提供的真实 digits 小数据，不下载 MNIST。每张图 64 个灰度值除以 16；stratify 保持各数字比例。训练和测试的样本索引互不相交。','''
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
torch.set_num_threads(2);torch.manual_seed(17)
digits=load_digits();X=digits.data.astype('float32')/16;y=digits.target
train_idx,test_idx=train_test_split(np.arange(len(y)),test_size=.25,random_state=17,stratify=y)
assert not set(train_idx)&set(test_idx)
dataset=TensorDataset(torch.tensor(X[train_idx]),torch.tensor(y[train_idx],dtype=torch.long))
loader=DataLoader(dataset,batch_size=64,shuffle=True,generator=torch.Generator().manual_seed(17))
print('一个批次',next(iter(loader))[0].shape)
'''),
('## 2. 定义输出十个分数的分类器\n最后一层输出 batch×10 logits，CrossEntropyLoss 会内部处理 log-softmax；目标是 long 类型类别编号，不是 one-hot。逻辑回归作为可解释的强基线。','''
baseline=LogisticRegression(max_iter=400).fit(X[train_idx],y[train_idx])
model=nn.Sequential(nn.Linear(64,64),nn.ReLU(),nn.Linear(64,10))
loss_fn=nn.CrossEntropyLoss();opt=torch.optim.Adam(model.parameters(),lr=.008)
assert model(torch.zeros(2,64)).shape==(2,10)
'''),
('## 3. 批次训练与正确的平均损失\n每个批次更新一次。累计 loss×样本数再除以总数，可避免最后一个短批次被赋予过大权重。固定 14 轮，测试集只在最后使用。','''
history=[]
for epoch in range(14):
    model.train();total=0.0
    for bx,by in loader:
        opt.zero_grad();loss=loss_fn(model(bx),by);loss.backward();opt.step()
        total+=loss.item()*len(by)
    history.append(total/len(dataset))
'''),
('## 4. 观察错分而不是只追准确率\nargmax 沿类别轴取分数最高类别。混淆矩阵的行是真实数字、列是预测数字；错误列表用于后续找规律，不能据此反复改模型再声称独立测试。','''
model.eval()
with torch.no_grad(): pred=model(torch.tensor(X[test_idx])).argmax(1).numpy()
base_pred=baseline.predict(X[test_idx]);cm=confusion_matrix(y[test_idx],pred)
errors=[{'sample':int(i),'actual':int(a),'predicted':int(p)} for i,a,p in zip(test_idx,y[test_idx],pred) if a!=p]
(OUT_DIR/'errors.json').write_text(json.dumps(errors,indent=2),encoding='utf-8')
fig,ax=plt.subplots();im=ax.imshow(cm,cmap='Blues');ax.set_xlabel('预测');ax.set_ylabel('真实');ax.set_title('数字混淆矩阵');fig.colorbar(im,ax=ax)
fig.tight_layout();fig.savefig(OUT_DIR/'confusion.png');plt.close(fig)
torch.save(model.state_dict(),OUT_DIR/'digits_mlp.pt')
finish({'mlp_accuracy':float(accuracy_score(y[test_idx],pred)),'logistic_accuracy':float(accuracy_score(y[test_idx],base_pred)),'test_samples':len(test_idx),'errors':len(errors)})
''')],[
ex('补全：批次形状','取一批数据，检查输入为二维、标签为一维，并检查模型输出每个样本有 10 个分数。','''bx,by=next(iter(loader))
assert bx.ndim==2 and by.ndim==1 and bx.shape[1]==64
assert model(bx).shape==(len(by),10)
print(bx.shape,by.shape)'''),
ex('修改：每个数字的召回率','由混淆矩阵算每类召回率，找出最弱数字，分母使用真实样本数。','''recall=np.diag(cm)/cm.sum(axis=1)
assert recall.shape==(10,) and np.all((recall>=0)&(recall<=1))
print('最弱数字',int(recall.argmin()),'召回率',float(recall.min()))'''),
ex('迁移：改成奇偶分类','保留同一划分，标签改成 y%2，用逻辑回归训练二分类并检查输出类别。','''binary=LogisticRegression(max_iter=400).fit(X[train_idx],y[train_idx]%2)
bp=binary.predict(X[test_idx])
assert set(np.unique(bp))<={0,1}
score=accuracy_score(y[test_idx]%2,bp)
assert score>.7
print('奇偶准确率',score)''')],
[('PyTorch：数据集与加载器','https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html'),('sklearn digits','https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html')],
'B16 训练循环；补充 Dataset、批次、类别轴。','这里是 8×8 digits，不是 MNIST；不能直接移用论文中的 MNIST 成绩。',
'真实小型手写数字数据由 scikit-learn load_digits 随包提供，源于 UCI Optical Recognition of Handwritten Digits；样本、字段、引用与许可见 sklearn 官方文档和根目录第三方声明。')

make(18,'卷积网络与增强','怎样用卷积网络改善图像识别？',
'任务：比较全连接网络与卷积网络，学习卷积共享局部特征的方式，也允许结果显示小数据上简单模型更好。',[
('## 1. 保留空间结构与验证集\n卷积输入是 N×C×H×W，本例为 N×1×8×8；全连接输入才展平。先划出测试集，再从训练部分划验证集，验证集用于选择训练轮次。','''
import copy
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
torch.set_num_threads(2);torch.manual_seed(18)
d=load_digits();X=(d.images/16).astype('float32')[:,None,:,:];y=d.target
rest,te=train_test_split(np.arange(len(y)),test_size=.2,random_state=18,stratify=y)
tr,va=train_test_split(rest,test_size=.25,random_state=18,stratify=y[rest])
assert set(tr).isdisjoint(va) and set(rest).isdisjoint(te)
print(X[tr].shape)
'''),
('## 2. 写出卷积、池化和分类头\n3×3 卷积加 padding=1 保持 8×8；池化降到 4×4；第二卷积输出 16×4×4，因此 Linear 输入 256。逐层检查形状能快速定位维度错误。','''
def build_cnn():
    return nn.Sequential(nn.Conv2d(1,8,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
        nn.Conv2d(8,16,3,padding=1),nn.ReLU(),nn.Flatten(),nn.Linear(16*4*4,10))
def build_mlp():
    return nn.Sequential(nn.Flatten(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,10))
def shift_right(batch):
    result=torch.zeros_like(batch);result[:,:,:,1:]=batch[:,:,:,:-1]
    return result
assert build_cnn()(torch.zeros(3,1,8,8)).shape==(3,10)
'''),
('## 3. 只对训练图像增强\n用零填充的小平移模拟轻微位置变化，不用会把边缘绕回来的 roll。两个模型使用相同划分与 10 轮训练预算；验证损失最低的权重被保留。增强并不必然有益。分类标签显式指定 torch.long，因为交叉熵需要整数类别编号，不能依赖不同系统的 NumPy 默认整数位宽。','''
def train_network(network,augment=False,epochs=10):
    torch.manual_seed(18)
    loader=DataLoader(TensorDataset(torch.tensor(X[tr]),torch.tensor(y[tr],dtype=torch.long)),batch_size=64,shuffle=True)
    opt=torch.optim.Adam(network.parameters(),lr=.006);criterion=nn.CrossEntropyLoss()
    best=float('inf');state=None;trace=[]
    for epoch in range(epochs):
        network.train()
        for bx,by in loader:
            if augment:
                mask=torch.rand(len(bx))<.2;bx=bx.clone();bx[mask]=shift_right(bx[mask])
            opt.zero_grad();loss=criterion(network(bx),by);loss.backward();opt.step()
        network.eval()
        with torch.no_grad(): val=float(criterion(network(torch.tensor(X[va])),torch.tensor(y[va],dtype=torch.long)))
        trace.append(val)
        if val<best:best=val;state=copy.deepcopy(network.state_dict())
    network.load_state_dict(state);return network,trace
mlp,mlp_history=train_network(build_mlp())
cnn,cnn_history=train_network(build_cnn(),augment=True)
'''),
('## 4. 最后比较独立测试结果\n比较准确率和验证曲线，不能因为叫 CNN 就宣布更好。记录参数量帮助理解复杂度；真实场景还要检查分布变化和图像采集条件。','''
with torch.no_grad():
    p_mlp=mlp(torch.tensor(X[te])).argmax(1).numpy()
    p_cnn=cnn(torch.tensor(X[te])).argmax(1).numpy()
fig,ax=plt.subplots();ax.plot(mlp_history,label='MLP');ax.plot(cnn_history,label='CNN + 平移');ax.set_title('验证损失');ax.legend()
fig.tight_layout();fig.savefig(OUT_DIR/'validation.png');plt.close(fig)
finish({'mlp_accuracy':float(accuracy_score(y[te],p_mlp)),'cnn_accuracy':float(accuracy_score(y[te],p_cnn)),
        'mlp_parameters':sum(p.numel() for p in mlp.parameters()),'cnn_parameters':sum(p.numel() for p in cnn.parameters()),'selected_by':'validation loss'})
''')],[
ex('补全：计算特征图形状','把一个全零批次逐层送进 CNN，断言第一卷积和池化之后的形状。','''net=build_cnn();z=torch.zeros(2,1,8,8)
z=net[0](z);assert z.shape==(2,8,8,8)
z=net[2](net[1](z));assert z.shape==(2,8,4,4)
print(z.shape)'''),
ex('修改：移到左边','写一个零填充左平移函数，用非对称小图核对，不得环绕边缘。','''def shift_left(batch):
    out=torch.zeros_like(batch);out[:,:,:,:-1]=batch[:,:,:,1:];return out
a=torch.arange(8.).reshape(1,1,2,4)
b=shift_left(a)
assert torch.equal(b[0,0,0],torch.tensor([1.,2.,3.,0.]))
assert b[:,:,:,-1].sum()==0
print(b)'''),
ex('迁移：检查采集位置变化','在测试图像统一右移一格后评价同一个 CNN；不重新训练，比较原始和扰动准确率。','''with torch.no_grad():
    shifted=cnn(shift_right(torch.tensor(X[te]))).argmax(1).numpy()
score=float(accuracy_score(y[te],shifted))
assert len(shifted)==len(te) and 0<=score<=1
print('原始',accuracy_score(y[te],p_cnn),'右移',score)''')],
[('PyTorch 图像分类教程','https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html')],
'B16—B17；补充四维张量、卷积和验证集。','增强只能作用于训练样本；本例模型不同，比较的是完整方案，并非单独隔离卷积的因果作用。',
'使用 sklearn 自带真实 digits 数据。模型结构与零填充平移为原创教学改编，未下载 CIFAR-10。')

make(19,'冻结预训练图像模型','图片不多时，怎样利用已有模型完成分类？',
'任务：在每类只有少量训练图片时，将本地 ImageNet ResNet18 当作特征提取器，和原像素分类比较。重点是迁移流程，不保证跨领域迁移改善精度。',[
('## 1. 固定少样本训练与测试\n每类只抽 15 张训练图片、另 10 张测试图片。所有图片来自 digits；转成 RGB 不会凭空增加信息。这里不是自然图片业务的精度实验。','''
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet18
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
torch.set_num_threads(2);torch.manual_seed(19)
d=load_digits();X=(d.images/16).astype('float32');y=d.target
rng=np.random.default_rng(19);tr=[];te=[]
for label in range(10):
    idx=rng.permutation(np.flatnonzero(y==label));tr.extend(idx[:15]);te.extend(idx[15:25])
tr=np.array(tr);te=np.array(te)
assert set(tr).isdisjoint(te)
'''),
('## 2. 从本地加载权重并冻结\nweights=None 阻止 torchvision 在线下载，随后显式读取已配套权重。requires_grad=False 禁止更新骨干；eval 固定 BatchNorm 行为。去掉原来的 1000 类头，输出 512 维表征。','''
weights_path=COURSE_ROOT/'assets/models/resnet18.pth'
if not weights_path.is_file(): raise FileNotFoundError(f'缺少离线权重：{weights_path}')
backbone=resnet18(weights=None)
backbone.load_state_dict(torch.load(weights_path,map_location='cpu',weights_only=True))
backbone.fc=nn.Identity()
for p in backbone.parameters():p.requires_grad=False
backbone.eval()
assert all(not p.requires_grad for p in backbone.parameters())
'''),
('## 3. 统一预处理后提取特征\n把灰度复制到三通道、缩放到 112×112，再用 ImageNet 均值和标准差规范化。112 是 CPU 教学加速设置，区别于常见 224 输入，应记录这一简化。特征提取不需要梯度。','''
def extract_features(images):
    pieces=[]
    mean=torch.tensor([.485,.456,.406]).reshape(1,3,1,1)
    std=torch.tensor([.229,.224,.225]).reshape(1,3,1,1)
    with torch.no_grad():
        for start in range(0,len(images),25):
            batch=torch.tensor(images[start:start+25])[:,None].repeat(1,3,1,1)
            batch=F.interpolate(batch,size=(112,112),mode='bilinear',align_corners=False)
            pieces.append(backbone((batch-mean)/std).numpy())
    return np.concatenate(pieces)
train_features=extract_features(X[tr]);test_features=extract_features(X[te])
assert train_features.shape==(150,512)
np.savez(OUT_DIR/'features.npz',train=train_features,test=test_features)
'''),
('## 4. 只训练新分类头，并解释迁移结果\n逻辑回归在冻结表征上学习 10 类。与原像素逻辑回归使用同一训练样本；保存准确率和错误数量。即使迁移方案更差，也完成了有效的选型实验。','''
raw=LogisticRegression(max_iter=500).fit(X[tr].reshape(len(tr),-1),y[tr])
head=LogisticRegression(max_iter=600,C=.5).fit(train_features,y[tr])
raw_pred=raw.predict(X[te].reshape(len(te),-1));transfer_pred=head.predict(test_features)
scores=[float(accuracy_score(y[te],raw_pred)),float(accuracy_score(y[te],transfer_pred))]
barplot(['原像素','冻结 ResNet18'],scores,'少样本迁移：测试准确率')
finish({'raw_pixel_accuracy':scores[0],'frozen_resnet_accuracy':scores[1],'train_samples':len(tr),'test_samples':len(te),'backbone_trainable_parameters':0,'input_size':112,'domain_warning':'ImageNet to digits; teaching workflow only'})
''')],[
ex('补全：冻结检查','确认骨干没有可训练参数，且 3 张图片得到 3×512 特征。','''assert sum(p.numel() for p in backbone.parameters() if p.requires_grad)==0
assert extract_features(X[te[:3]]).shape==(3,512)
print('骨干已冻结')'''),
ex('修改：只训练一个神经分类头','冻结 train_features，训练 Linear(512,10) 15 步；检查骨干未获梯度而分类头获梯度。','''torch.manual_seed(19)
new_head=nn.Linear(512,10);opt=torch.optim.Adam(new_head.parameters(),lr=.01)
ft=torch.tensor(train_features);labels=torch.tensor(y[tr],dtype=torch.long)
for _ in range(15):
    opt.zero_grad();loss=nn.CrossEntropyLoss()(new_head(ft),labels);loss.backward();opt.step()
assert new_head.weight.grad is not None
assert all(p.grad is None for p in backbone.parameters())
print(float(loss))'''),
ex('迁移：特征用于奇偶分类','不重新提取图像特征，只把新任务的标签改成奇偶。说明这样节省的是哪个环节。','''parity=LogisticRegression(max_iter=500).fit(train_features,y[tr]%2)
result=parity.predict(test_features)
assert set(np.unique(result))<={0,1} and len(result)==len(te)
print('奇偶准确率',accuracy_score(y[te]%2,result))''')],
[('PyTorch 迁移学习教程','https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html'),('torchvision ResNet18','https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html')],
'B17—B18；补充参数冻结、特征提取与迁移边界。','ImageNet 与数字领域差异很大；迁移并不必然胜过原像素。完整骨干微调是拓展，必须另留验证集、比较预算。',
'真实 digits 图片与本地 ResNet18 ImageNet 权重。权重文件来自 torchvision 官方资源，需首次按根目录安装指南下载至 assets/models；来源、哈希和许可见 assets 清单。运行时不下载。')

make(20,'情感分类与文本表示','怎样训练一个评论情感分类器？',
'任务：用一份可检查的小型英文评论语料比较 TF-IDF 线性模型和词向量平均网络；中文解释帮助你观察分词、词表和否定句错误。',[
('## 1. 先写清语料和划分\n此语料由课程作者编写，不是真实用户数据。训练和测试句子分别列出，测试包含否定、同义改写和混合评价。很小的语料只能学习流程，不能给出产品质量结论。','''
import re
import torch
from torch import nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
torch.set_num_threads(2);torch.manual_seed(20)
positive=['great food and friendly service','excellent quality and fast delivery','I love this useful product','wonderful experience with helpful staff','good value and reliable design','the room was clean and comfortable','very happy with the purchase','delicious meal and nice atmosphere','works perfectly and feels sturdy','amazing support solved my problem','easy to use and good performance','I recommend this excellent book']
negative=['terrible food and rude service','poor quality and slow delivery','I hate this useless product','awful experience with unhelpful staff','bad value and unreliable design','the room was dirty and uncomfortable','very unhappy with the purchase','disgusting meal and noisy atmosphere','does not work and feels fragile','horrible support ignored my problem','hard to use and bad performance','I regret buying this boring book']
texts=positive+negative;labels=np.array([1]*len(positive)+[0]*len(negative))
test_texts=['friendly staff and excellent food','slow delivery and terrible quality','good product with reliable performance','rude staff and dirty room','not good at all','not a terrible experience','helpful support but slow delivery','I love the clean comfortable room']
test_labels=np.array([1,0,1,0,0,1,0,1])
(DATA_DIR/'reviews.json').write_text(json.dumps({'train':list(zip(texts,labels.tolist())),'test':list(zip(test_texts,test_labels.tolist()))},ensure_ascii=False,indent=2),encoding='utf-8')
'''),
('## 2. TF-IDF 基线与只来自训练集的词表\n词表把 token 映射到整数，0 是补齐、1 是未知词。不能先用测试语料建立词表。TF-IDF 使用一元和二元词组，给否定搭配保留一些表达空间。','''
vectorizer=TfidfVectorizer(ngram_range=(1,2))
baseline=LogisticRegression(C=2,max_iter=300).fit(vectorizer.fit_transform(texts),labels)
def tokenize(text):return re.findall(r"[a-z]+",text.lower())
vocab={'<pad>':0,'<unk>':1}
for text in texts:
    for token in tokenize(text):
        if token not in vocab:vocab[token]=len(vocab)
def encode(sentences,max_len=16):
    result=np.zeros((len(sentences),max_len),dtype='int64')
    for i,text in enumerate(sentences):
        ids=[vocab.get(word,1) for word in tokenize(text)[:max_len]];result[i,:len(ids)]=ids
    return torch.tensor(result)
assert encode(['unseenword'])[0,0].item()==1
'''),
('## 3. Embedding 加掩码平均\nEmbedding 将编号变成 24 维可学习向量。mask 排除 padding，分母 clamp 防止空句除零。平均池化会丢失顺序，所以这是一个特意简单、有明确局限的网络。交叉熵的类别标签使用 torch.long，保持与损失函数的类型约定一致。','''
class SentimentNet(nn.Module):
    def __init__(self,size):
        super().__init__();self.embedding=nn.Embedding(size,24,padding_idx=0);self.head=nn.Linear(24,2)
    def forward(self,ids):
        mask=(ids!=0).unsqueeze(-1)
        pooled=(self.embedding(ids)*mask).sum(1)/mask.sum(1).clamp(min=1)
        return self.head(pooled)
network=SentimentNet(len(vocab));opt=torch.optim.Adam(network.parameters(),lr=.025)
train_ids=encode(texts);train_labels=torch.tensor(labels,dtype=torch.long);losses=[]
for epoch in range(70):
    opt.zero_grad();loss=nn.CrossEntropyLoss()(network(train_ids),train_labels);loss.backward();opt.step();losses.append(float(loss.detach()))
'''),
('## 4. 保存逐条预测和失败类型\n输出两种预测以便人工检查“not good”等否定句。准确率只是这八句上的描述统计，不能当总体保证；复杂模型在小数据上很可能过拟合。','''
network.eval()
with torch.no_grad():neural_pred=network(encode(test_texts)).argmax(1).numpy()
linear_pred=baseline.predict(vectorizer.transform(test_texts))
rows=[{'text':t,'label':int(y),'tfidf':int(a),'embedding':int(b)} for t,y,a,b in zip(test_texts,test_labels,linear_pred,neural_pred)]
(OUT_DIR/'predictions.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
scores=[float(accuracy_score(test_labels,linear_pred)),float(accuracy_score(test_labels,neural_pred))]
barplot(['TF-IDF','Embedding 平均'],scores,'小语料独立测试准确率')
finish({'tfidf_accuracy':scores[0],'embedding_accuracy':scores[1],'test_samples':len(test_labels),'vocabulary_size':len(vocab),'training_loss':losses[-1]})
''')],[
ex('补全：未知词比例','计算测试语料中未知词占全部词的比例，检查分母非零，打印有哪些未知词。','''tokens=[w for t in test_texts for w in tokenize(t)]
unknown=[w for w in tokens if w not in vocab]
ratio=len(unknown)/len(tokens)
assert 0<=ratio<=1
print(ratio,sorted(set(unknown)))'''),
ex('修改：补齐不应改变结果','同一句话编码到长度 10 和 20，检查掩码平均产生的分数一致。','''with torch.no_grad():
    a=network(encode(['good food'],10));b=network(encode(['good food'],20))
assert torch.allclose(a,b,atol=1e-6)
print(a)'''),
ex('迁移：另建中文词级基线','用分好词的中文服务评价训练一个新分类器，不混用英文词表；核对正负对照。','''cn=['服务 很 好','态度 友好 推荐','服务 很 差','态度 粗鲁 失望']
cv=TfidfVectorizer(tokenizer=str.split,token_pattern=None)
cl=LogisticRegression(C=10).fit(cv.fit_transform(cn),[1,1,0,0])
cp=cl.predict(cv.transform(['友好 推荐','粗鲁 失望']))
assert cp.tolist()==[1,0]
print(cp)''')],
[('PyTorch Embedding','https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html'),('sklearn 文本特征','https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction')],
'B08 文本向量与 B16 训练；补充词表、未知词、padding 掩码。','混合评价的标签依赖任务定义；本例将物流差的混合评价标负面，必须在真实任务中和业务方约定。')

make(21,'本地预训练Transformer','怎样使用预训练 Transformer 处理文本？',
'任务：使用本地 FLAN-T5-small 按指令判断英文情感，检查分词、掩码与格式错误。轻量词典是对照，模型无需联网。',[
('## 1. 定义任务与规则对照\n目标输出只能是 positive 或 negative。测试句含否定，模型只看文本，不看人工标签；词典很便宜，但不知道句法。','''
import re
import torch
from transformers import AutoTokenizer,AutoModelForSeq2SeqLM
torch.set_num_threads(2)
examples=[('I love this wonderful book.',1),('The service was terrible and slow.',0),('It is not good at all.',0),('A reliable and useful product.',1),('I regret buying this item.',0),('The staff were friendly and helpful.',1)]
def lexicon_predict(text):
    tokens=set(re.findall('[a-z]+',text.lower()))
    return int(len(tokens&{'love','wonderful','good','reliable','useful','friendly','helpful'})>=len(tokens&{'terrible','slow','regret','bad','hate'}))
baseline_pred=[lexicon_predict(t) for t,_ in examples]
'''),
('## 2. 本地加载与批分词\nTokenizer 将文本变成 token ID；attention_mask=0 表示补齐。FLAN-T5 是编码器—解码器模型，必须使用相应模型类。local_files_only 禁止运行时下载。','''
model_path=COURSE_ROOT/'assets/models/flan-t5-small'
if not (model_path/'config.json').is_file():raise FileNotFoundError(f'缺少本地模型：{model_path}')
tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
model=AutoModelForSeq2SeqLM.from_pretrained(model_path,local_files_only=True);model.eval()
prompts=['Classify the sentiment as positive or negative. Answer with one word. Text: '+text for text,_ in examples]
batch=tokenizer(prompts,padding=True,truncation=True,max_length=128,return_tensors='pt')
assert batch['input_ids'].shape==batch['attention_mask'].shape
assert set(batch['attention_mask'].unique().tolist())<={0,1}
print('输入形状',batch['input_ids'].shape)
'''),
('## 3. 生成与独立解析\n固定贪心解码和最大新 token 数。解析器只接受约定标签，其他文本记为 invalid；格式合法不代表语义正确。','''
t0=time.perf_counter()
with torch.inference_mode():generated=model.generate(**batch,max_new_tokens=8,do_sample=False,num_beams=1)
raw_outputs=tokenizer.batch_decode(generated,skip_special_tokens=True);generation_seconds=time.perf_counter()-t0
def parse_label(text):return {'positive':1,'negative':0}.get(text.strip().lower().strip('.! '),-1)
model_pred=[parse_label(t) for t in raw_outputs]
'''),
('## 4. 保留真实输出与错误\n分别报告格式通过率、准确率与耗时。六句仅能说明本例行为，不能代表全部评论；不得通过改测试句掩盖失败。','''
truth=np.array([label for _,label in examples])
rows=[{'text':t,'expected':y,'raw':raw,'parsed':int(p),'baseline':int(b)} for (t,y),raw,p,b in zip(examples,raw_outputs,model_pred,baseline_pred)]
(OUT_DIR/'predictions.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
scores=[float(np.mean(truth==baseline_pred)),float(np.mean(truth==model_pred))]
barplot(['词典规则','FLAN-T5'],scores,'本地 Transformer 与基线')
finish({'baseline_accuracy':scores[0],'transformer_accuracy':scores[1],'format_valid_rate':float(np.mean(np.array(model_pred)!=-1)),'generation_seconds':round(generation_seconds,3),'examples':len(examples),'model':'google/flan-t5-small'})
''')],[
ex('补全：解析边界','检查大小写、句点和未知输出。','''assert parse_label(' Positive. ')==1
assert parse_label('negative')==0
assert parse_label('I am not certain')==-1
print('通过')'''),
ex('修改：观察补齐','两句不同长度输入一批，核对短句的真实长度较小。','''b=tokenizer(['good','This is a much longer sentence about a product'],padding=True,return_tensors='pt')
assert b['attention_mask'][0].sum()<b['attention_mask'][1].sum()
ids=b['input_ids'][0][b['attention_mask'][0].bool()]
assert len(ids)==int(b['attention_mask'][0].sum())
print(tokenizer.decode(ids,skip_special_tokens=True))'''),
ex('迁移：资料内抽取地点','让模型从资料抽取 Room 204，保留真实输出，检查是否命中人工答案。','''inp=tokenizer('Context: The workshop takes place in Room 204. Question: In which room is the workshop? Answer briefly.',return_tensors='pt')
with torch.inference_mode():out=model.generate(**inp,max_new_tokens=10,do_sample=False)
answer=tokenizer.decode(out[0],skip_special_tokens=True)
assert isinstance(answer,str) and len(answer)>0
print('真实输出',answer,'包含标准答案','204' in answer)''')],
[('FLAN-T5-small 模型卡','https://huggingface.co/google/flan-t5-small'),('Transformers 生成','https://huggingface.co/docs/transformers/main_classes/text_generation')],
'B20；分词、掩码和生成。','微调为拓展；不得用测试标签调提示词。该模型主要用于英文教学对照。',
'作者编写的英文测试句；FLAN-T5-small 权重位于 assets/models/flan-t5-small，来源与许可见课程第三方清单。')

make(22,'从笔记检索段落','怎样从一批笔记中找到相关段落？',
'任务：将中文手册切成带来源片段，用 TF-IDF 检索，并用预先标注的问题检查 Recall@k 和 MRR。',[
('## 1. 文档与人工相关性标签\n内容是虚构校园规则；字典键为来源 ID。评测问题提前指定相关文档，不让检索器自己给自己标答案。','''
from sklearn.feature_extraction.text import TfidfVectorizer
documents={
 'D1':'实验室预约需要提前一天提交申请。每次预约最长两小时。取消预约请在开始前四小时通知管理员。管理员会检查设备使用资格。',
 'D2':'图书馆工作日晚上22点关闭。周末晚上20点关闭。借书期限为30天，续借需要通过图书管理系统办理。',
 'D3':'校园开放日的签到地点是一号楼大厅。志愿者早上8点集合。访客请携带报名二维码。活动遇到大雨时转入体育馆。',
 'D4':'共享单车故障时请停止骑行并上报车辆编号。不能把车停在消防通道。每周维护人员会检查刹车和轮胎。',
 'D5':'报销材料包括发票、支付凭证和项目编号。财务在每周五处理申请。交通费与餐费应分别填写。',
 'D6':'服务器训练任务最长运行六小时。提交前需要记录随机种子和依赖版本。GPU预约冲突时优先安排已批准的课程实验。'}
queries=[('实验室预约最长多久','D1'),('周末图书馆几点关闭','D2'),('下大雨开放日去哪里','D3'),('共享单车故障需要报告什么','D4'),('报销需要哪些材料','D5'),('训练任务最长多长时间','D6')]
(DATA_DIR/'notes.json').write_text(json.dumps(documents,ensure_ascii=False,indent=2),encoding='utf-8')
'''),
('## 2. 切分保留来源与字符位置\n采用 40 字窗口和 10 字重叠，使这些短手册也实际产生多个片段。保存 start/end，使片段可以回查；真实长文可改用句子边界，防止割裂条件。','''
def chunk_text(text,width=40,overlap=10):
    if not 0<=overlap<width:raise ValueError('要求 0 <= overlap < width')
    result=[]
    for start in range(0,len(text),width-overlap):
        result.append((start,min(start+width,len(text)),text[start:start+width]))
        if start+width>=len(text):break
    return result
chunks=[]
for doc_id,text in documents.items():
    for start,end,part in chunk_text(text):chunks.append({'doc_id':doc_id,'start':start,'end':end,'text':part})
assert all(documents[c['doc_id']][c['start']:c['end']]==c['text'] for c in chunks)
assert len(chunks)>len(documents)
'''),
('## 3. 字符 n-gram 关键词索引\n中文无空格，因此用字符 2—4 gram，不依赖外部分词器。规范化 TF-IDF 的点积等于余弦相似度；零匹配和空查询不能返回假命中。','''
vectorizer=TfidfVectorizer(analyzer='char',ngram_range=(2,4))
matrix=vectorizer.fit_transform([c['text'] for c in chunks])
def retrieve(query,k=3):
    if not query.strip():return []
    scores=(matrix@vectorizer.transform([query]).T).toarray().ravel()
    order=np.argsort(-scores,kind='stable')[:k]
    return [{**chunks[i],'score':float(scores[i])} for i in order if scores[i]>0]
print(retrieve(queries[0][0]))
'''),
('## 4. 排名评价与失败记录\nRecall@k 判断正确来源是否进入前 k；MRR 是首个相关结果名次倒数。这里按来源判断相关，不意味着每个命中片段都已包含答案。','''
results=[]
for query,gold in queries:
    hits=retrieve(query,3);ranks=[i+1 for i,h in enumerate(hits) if h['doc_id']==gold]
    results.append({'query':query,'gold':gold,'hits':hits,'hit1':bool(hits and hits[0]['doc_id']==gold),'hit3':bool(ranks),'rr':1/min(ranks) if ranks else 0.0})
(OUT_DIR/'retrieval_results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
scores=[float(np.mean([r['hit1'] for r in results])),float(np.mean([r['hit3'] for r in results])),float(np.mean([r['rr'] for r in results]))]
barplot(['Recall@1','Recall@3','MRR@3'],scores,'人工标注问题的检索表现')
finish({'recall_at_1':scores[0],'recall_at_3':scores[1],'mrr_at_3':scores[2],'chunks':len(chunks),'evaluation_questions':len(queries)})
''')],[
ex('补全：重叠切分','窗口 6、重叠 2，核对 ABCDEFGHIJ 的片段以及非法参数。','''parts=chunk_text('ABCDEFGHIJ',6,2)
assert [p[2] for p in parts]==['ABCDEF','EFGHIJ']
try:chunk_text('abc',2,2)
except ValueError:pass
else:raise AssertionError('非法参数未拒绝')
print(parts)'''),
ex('修改：无匹配处理','检查空查询与完全不在语料里的查询。','''assert retrieve('')==[]
assert retrieve('zzzzzz')==[]
assert retrieve('实验室预约',1)[0]['doc_id']=='D1'
print('通过')'''),
ex('迁移：加入新文档','新增会议室规则并重建独立索引，核对新来源可找到。','''new_docs={**documents,'D7':'会议室预约必须填写参会人数。会议结束后应关闭投影仪。'}
v=TfidfVectorizer(analyzer='char',ngram_range=(2,4));ids=list(new_docs)
m=v.fit_transform(list(new_docs.values()));score=(m@v.transform(['会议室参会人数']).T).toarray().ravel()
assert ids[int(score.argmax())]=='D7'
print(ids[int(score.argmax())])''')],
[('sklearn 文本特征','https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction'),('检索与重排','https://sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html')],
'B08、B20；切片、稀疏矩阵和排名。','虚构规则不是实际学校规定；命中来源不等于正确回答。')

make(23,'关键词与向量检索','提问和原文用词不同，怎样找到相关内容？',
'任务：在同一份英文笔记和人工标签上，比较关键词检索与本地 MiniLM；用同语言对照避免把语言差异误当算法优势。',[
('## 1. 固定笔记与同义改写问题\n每个问题有一个提前指定的人工相关文档。模型从未看到本例标签，但预训练语料不可完全检查，因此不声称严格的新领域泛化。','''
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import torch
torch.set_num_threads(2)
notes=[('D1','Cancel a laboratory booking at least four hours before it starts.'),
       ('D2','The library closes at ten in the evening on weekdays.'),
       ('D3','During heavy rain the outdoor event moves to the sports hall.'),
       ('D4','Report a broken bicycle and include its identification number.'),
       ('D5','Reimbursement requires an invoice and proof of payment.'),
       ('D6','Save random seeds and software versions before training a model.')]
questions=[('How do I withdraw my reservation?',0),('When must readers leave on Monday night?',1),
           ('Where will people gather if it pours?',2),('What should I do about a damaged bike?',3),
           ('Which receipts are needed to get my money back?',4),('What helps make an experiment repeatable?',5)]
texts=[t for _,t in notes];qs=[q for q,_ in questions];gold=np.array([g for _,g in questions])
(DATA_DIR/'semantic_notes.json').write_text(json.dumps({'notes':notes,'questions':questions},indent=2),encoding='utf-8')
'''),
('## 2. 关键词基线与零匹配\nTF-IDF 只从当前笔记建立词表；完全没有共同词时所有分数可能为零，不能把排序返回的第一个文档算作有效命中。','''
v=TfidfVectorizer(stop_words='english',ngram_range=(1,2))
lexical_matrix=v.fit_transform(texts)
t0=time.perf_counter();lexical_scores=(v.transform(qs)@lexical_matrix.T).toarray();lexical_seconds=time.perf_counter()-t0
lexical_top=lexical_scores.argmax(axis=1)
lexical_hit=(lexical_top==gold)&(lexical_scores.max(axis=1)>0)
'''),
('## 3. 本地向量编码与余弦相似度\nnormalize_embeddings=True 使向量长度为 1，矩阵乘法得到所有问题对所有笔记的相似度。分别记录模型加载、建索引和查询耗时，不用缓存后速度掩盖初次成本。','''
model_path=COURSE_ROOT/'assets/models/minilm'
if not (model_path/'config.json').is_file():raise FileNotFoundError(f'缺少嵌入模型：{model_path}')
t0=time.perf_counter();encoder=SentenceTransformer(str(model_path),device='cpu',local_files_only=True);load_seconds=time.perf_counter()-t0
t0=time.perf_counter();embeddings=encoder.encode(texts,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False);index_seconds=time.perf_counter()-t0
t0=time.perf_counter();query_embeddings=encoder.encode(qs,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False)
semantic_scores=query_embeddings@embeddings.T;query_seconds=time.perf_counter()-t0
semantic_top=semantic_scores.argmax(axis=1)
assert np.allclose(np.linalg.norm(embeddings,axis=1),1,atol=1e-5)
'''),
('## 4. 比较效果与成本\n保留逐问题的预测来源，失败可回查。此处没有无答案阈值，语义相似度再高也不能替代相关性判断；中文任务需换成同语言评测集。','''
rows=[{'query':q,'gold':notes[g][0],'lexical_top':notes[int(a)][0],'lexical_max_score':float(lexical_scores[i].max()),'semantic_top':notes[int(b)][0],'semantic_max_score':float(semantic_scores[i].max())} for i,((q,g),a,b) in enumerate(zip(questions,lexical_top,semantic_top))]
(OUT_DIR/'comparison.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
scores=[float(lexical_hit.mean()),float(np.mean(semantic_top==gold))]
barplot(['关键词','MiniLM 向量'],scores,'同义改写检索 Recall@1')
finish({'lexical_recall_at_1':scores[0],'semantic_recall_at_1':scores[1],'model_load_seconds':round(load_seconds,3),'index_seconds':round(index_seconds,3),'semantic_query_seconds':round(query_seconds,3),'lexical_query_seconds':round(lexical_seconds,3),'language':'English controlled comparison'})
''')],[
ex('补全：余弦相似度','计算 [3,4] 与 [4,3] 的规范化点积，核对相同向量相似度。','''a=np.array([3.,4.]);b=np.array([4.,3.]);a/=np.linalg.norm(a);b/=np.linalg.norm(b)
assert np.isclose(a@a,1) and np.isclose(a@b,.96)
print(a@b)'''),
ex('修改：Recall@3','用原分数矩阵计算 Recall@3，不能重新选择问题。','''top3=np.argsort(-semantic_scores,axis=1)[:,:3]
recall3=float(np.mean([g in row for g,row in zip(gold,top3)]))
assert recall3>=float(np.mean(semantic_top==gold))
print(recall3)'''),
ex('迁移：排名融合','实现 reciprocal rank fusion，每种方法按 1/(60+排名) 累加；零匹配不贡献票数，避免词典顺序制造证据。','''def rrf(a,b,k=60):
    combined=np.zeros_like(a)
    for scores in [a,b]:
        for i,row in enumerate(np.argsort(-scores,axis=1)):
            for rank,j in enumerate(row,1):
                if scores[i,j]>0:combined[i,j]+=1/(k+rank)
    return combined
fused=rrf(lexical_scores,semantic_scores)
assert fused.shape==semantic_scores.shape and np.all(fused>=0) and np.all(fused.sum(axis=1)>0)
print('融合 Recall@1',np.mean(fused.argmax(1)==gold))''')],
[('SentenceTransformers 语义搜索','https://sbert.net/examples/sentence_transformer/applications/semantic-search/README.html'),('MiniLM 模型卡','https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2')],
'B22；向量规范化、矩阵乘法与查询成本。','仅六个问题，不足以选定生产模型。中文模型与小候选重排为拓展；无关文本也会产生向量相似度。',
'人工编写英文笔记与改写问题；本地 all-MiniLM-L6-v2 权重位于 assets/models/minilm，来源、revision 与许可见清单。')

make(24,'中文本地资料问答','怎样让本地语言模型参考资料回答问题？',
'任务：使用虚构开放日手册完成中文本地 RAG。检索、生成、证据检查和拒答分别评价；保留原始输出，不以模板答案替代失败的生成。',[
('## 1. 手册、验证查询与独立测试题\n验证查询只用于选择拒答阈值。测试有两道可回答题和两道资料外问题；gold=None 明确表示资料没有答案。规则仅是教学虚构。','''
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
torch.set_num_threads(2)
docs=[('D1','开放日签到地点是一号楼大厅。'),('D2','志愿者集合时间是早上8点。'),('D3','遇到大雨，活动转入体育馆。'),('D4','报销材料必须包含发票和支付凭证。')]
validation=[('开放日在哪里签到',0),('志愿者几点集合',1),('今晚股票会涨吗',None),('校长的生日是哪天',None)]
tests=[('签到地点在哪里',0,'一号楼'),('下大雨活动转到哪里',2,'体育馆'),('明天股票价格是多少',None,None),('校长的生日是什么时候',None,None)]
v=TfidfVectorizer(analyzer='char',ngram_range=(1,3));index=v.fit_transform([t for _,t in docs])
def find_evidence(question):
    scores=(index@v.transform([question]).T).toarray().ravel();i=int(scores.argmax());return i,float(scores[i])
validation_scores=[find_evidence(q)[1] for q,_ in validation]
candidates=np.arange(.1,.81,.05)
threshold=float(min(candidates,key=lambda t:sum((s>=t)!=(g is not None) for s,(_,g) in zip(validation_scores,validation))))
print('验证集选定阈值',threshold)
'''),
('## 2. 因果语言模型与提示模板\nQwen 使用 AutoModelForCausalLM。chat template 采用非思考模式，CPU float32、最多32个新 token。模型文件缺失明确失败，所有加载使用 local_files_only。','''
model_path=COURSE_ROOT/'assets/models/qwen3-0.6b'
if not (model_path/'config.json').is_file():raise FileNotFoundError(f'缺少本地生成模型：{model_path}')
tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
model=AutoModelForCausalLM.from_pretrained(model_path,local_files_only=True,torch_dtype=torch.float32);model.eval()
def generate_answer(question,doc_index):
    doc_id,text=docs[doc_index]
    messages=[{'role':'system','content':'你是资料助手。只依据资料回答，用一句简短中文，不补充其他事实。'},
              {'role':'user','content':f'资料[{doc_id}]：{text} 问题：{question} 直接给出答案。'}]
    prompt=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    inputs=tokenizer(prompt,return_tensors='pt')
    with torch.inference_mode():out=model.generate(**inputs,max_new_tokens=32,do_sample=False,pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0,inputs['input_ids'].shape[1]:],skip_special_tokens=True).strip()
'''),
('## 3. 门控、生成和支持检查\n低分直接拒答；高分只是允许生成。引用绑定实际检索来源。人工标注的关键短语检查只是支持率代理，无法识别额外编造或矛盾，必须保存原文供人工复核。','''
rows=[]
for question,gold,expected in tests:
    i,score=find_evidence(question);abstained=score<threshold;t0=time.perf_counter()
    raw='资料不足，无法回答。' if abstained else generate_answer(question,i)
    supported=bool(not abstained and gold==i and expected is not None and expected in raw and expected in docs[i][1])
    rendered=raw if abstained else raw+f' [来源:{docs[i][0]}]'
    rows.append({'question':question,'gold_doc':None if gold is None else docs[gold][0],'retrieved_doc':docs[i][0],
                 'retrieval_score':score,'abstained':abstained,'raw_generation':raw,'answer_with_source':rendered,
                 'expected_phrase':expected,'phrase_support_proxy':supported,'seconds':round(time.perf_counter()-t0,3)})
    print(question,rendered)
'''),
('## 4. 结果拆开评价\n分别报告有答案题检索命中、关键短语支持代理、无答案拒答和错误拒答。高代理分数不证明所有陈述正确；四道题也不能证明生产可靠性。','''
answerable=[r for r in rows if r['gold_doc'] is not None];unanswerable=[r for r in rows if r['gold_doc'] is None]
recall=float(np.mean([r['retrieved_doc']==r['gold_doc'] for r in answerable]))
support=float(np.mean([r['phrase_support_proxy'] for r in answerable]))
refusal=float(np.mean([r['abstained'] for r in unanswerable]));false_refusal=float(np.mean([r['abstained'] for r in answerable]))
(OUT_DIR/'answers_for_review.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
(DATA_DIR/'handbook.json').write_text(json.dumps(docs,ensure_ascii=False,indent=2),encoding='utf-8')
barplot(['来源命中','短语支持代理','无答案拒答','错误拒答'],[recall,support,refusal,false_refusal],'RAG 分项评价')
finish({'answerable_retrieval_recall':recall,'phrase_support_proxy_rate':support,'unanswerable_refusal_rate':refusal,'false_refusal_rate':false_refusal,'threshold':threshold,'test_questions':len(tests),'human_full_claim_review':'required; phrase proxy is not full verification','model':'Qwen/Qwen3-0.6B'})
''')],[
ex('补全：核对引用','非拒答答案的引用必须指向实际检索来源。','''for row in rows:
    if not row['abstained']:assert '[来源:'+row['retrieved_doc']+']' in row['answer_with_source']
assert len(rows)==4
print('引用绑定通过；不是事实正确性证明')'''),
ex('修改：提高拒答阈值','仅在验证集比较原阈值与0.8导致的拒答数。解释错误拒答可能增加。','''original=sum(s<threshold for s in validation_scores);strict=sum(s<.8 for s in validation_scores)
assert strict>=original
print('原阈值拒答',original,'严格阈值拒答',strict)'''),
ex('迁移：受约束的计算工具','只允许 add，参数必须为有限数值；拒绝未知工具和字符串参数，不执行代码字符串。','''def call_tool(name,args):
    if name!='add':raise ValueError('不允许的工具')
    if set(args)!={'a','b'}:raise ValueError('参数字段错误')
    if any(type(x) not in (int,float) or not np.isfinite(x) for x in args.values()):raise ValueError('数值无效')
    return args['a']+args['b']
assert call_tool('add',{'a':2,'b':3})==5
for name,args in [('shell',{}),('add',{'a':'2','b':3})]:
    try:call_tool(name,args)
    except ValueError:pass
    else:raise AssertionError('非法调用未拒绝')
print('工具边界检查通过')''')],
[('Qwen3-0.6B 模型卡','https://huggingface.co/Qwen/Qwen3-0.6B'),('检索与重排','https://sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html')],
'B21—B23；RAG、拒答、引用与人工核验。','字符相似度不能发现所有资料外问题；支持代理会漏掉矛盾和额外事实。原始答案必须人工复核，资料内指令不能作为程序指令。',
'作者编写虚构中文手册、验证集和测试集；Qwen3-0.6B 权重需首次按根目录安装指南下载至 assets/models。运行时不访问网络。')

make(25,'A星寻找路线','怎样让程序在有障碍的地图上找到路线？',
'任务：给配送小车寻找四方向可通行的最短路线。先定义合法移动，再用 Dijkstra 对照 A*，检查启发式为什么可用。',[
('## 1. 地图、状态与合法动作\n状态是 (行,列) 元组，0 可走、1 障碍，每步成本 1。边界与障碍检查集中写在 neighbors 中；没有对角移动，曼哈顿距离才是对应的下界。','''
import heapq
from collections import deque
grid=np.array([[0,0,0,1,0,0],[1,1,0,1,0,1],[0,0,0,0,0,0],[0,1,1,1,1,0],[0,0,0,0,0,0]],dtype=int)
start=(0,0);goal=(4,5)
def neighbors(node,board):
    r,c=node
    for dr,dc in [(1,0),(-1,0),(0,1),(0,-1)]:
        nr,nc=r+dr,c+dc
        if 0<=nr<board.shape[0] and 0<=nc<board.shape[1] and board[nr,nc]==0:yield (nr,nc)
def manhattan(a,b):return abs(a[0]-b[0])+abs(a[1]-b[1])
assert manhattan(start,goal)==9
np.savetxt(DATA_DIR/'map.csv',grid,fmt='%d',delimiter=',')
'''),
('## 2. 优先队列与 g+h\ng 保存当前已知的实际代价，h 是剩余代价下界。heapq 每次取最小 f；更短路径出现时更新父指针。旧队列项可能过时，要跳过，避免重复扩展污染统计。','''
def astar(board,source,target,use_heuristic=True):
    for p in [source,target]:
        if not (0<=p[0]<board.shape[0] and 0<=p[1]<board.shape[1]) or board[p]!=0:raise ValueError('起终点必须可通行')
    h=(lambda p:manhattan(p,target)) if use_heuristic else (lambda p:0)
    queue=[(h(source),0,source)];g={source:0};parent={};expanded=0
    while queue:
        _,distance,node=heapq.heappop(queue)
        if distance!=g[node]:continue
        expanded+=1
        if node==target:
            path=[node]
            while node!=source:node=parent[node];path.append(node)
            return list(reversed(path)),expanded
        for nxt in neighbors(node,board):
            tentative=distance+1
            if tentative<g.get(nxt,float('inf')):
                g[nxt]=tentative;parent[nxt]=node;heapq.heappush(queue,(tentative+h(nxt),tentative,nxt))
    return [],expanded
path,expanded=astar(grid,start,goal)
'''),
('## 3. 独立验证每一步与最短长度\n检查路径从起点到终点、每步邻接、不穿障碍。再用不含启发式的 Dijkstra 对照最短长度；A* 能减少扩展是经验结果，不预设它在所有地图都更快。','''
def validate_path(board,path,source,target):
    return bool(path and path[0]==source and path[-1]==target and all(board[p]==0 for p in path) and all(b in set(neighbors(a,board)) for a,b in zip(path,path[1:])))
reference,reference_expanded=astar(grid,start,goal,use_heuristic=False)
assert validate_path(grid,path,start,goal)
assert len(path)==len(reference)
blocked=np.array([[0,1],[1,0]])
assert astar(blocked,(0,0),(1,1))[0]==[]
'''),
('## 4. 输出路线和搜索工作量\n图上展示坐标路径，JSON 保存可重放坐标。报告步数和扩展节点数，不把路径长度与运行时间混成一个指标。','''
fig,ax=plt.subplots();ax.imshow(grid,cmap='Greys',vmin=0,vmax=1)
ax.plot([p[1] for p in path],[p[0] for p in path],'-o',color='tab:red');ax.set_title('A* 最短可行路线')
ax.text(start[1],start[0],'S',color='blue');ax.text(goal[1],goal[0],'G',color='blue')
fig.tight_layout();fig.savefig(OUT_DIR/'route.png');plt.close(fig)
(OUT_DIR/'path.json').write_text(json.dumps(path),encoding='utf-8')
finish({'steps':len(path)-1,'astar_expanded':expanded,'dijkstra_expanded':reference_expanded,'path_valid':True,'unreachable_checked':True})
''')],[
ex('补全：合法邻居','核对左上角只有右邻居可行，终点的所有邻居均在边界内且可通行。','''assert set(neighbors((0,0),grid))=={(0,1)}
assert all(grid[p]==0 for p in neighbors(goal,grid))
print(list(neighbors(goal,grid)))'''),
ex('修改：封闭关键路口','把 (1,2) 封闭后重新规划，检查原路线无效且新图中起点无法到终点。','''closed=grid.copy();closed[1,2]=1
assert not validate_path(closed,path,start,goal)
new_path,_=astar(closed,start,goal)
assert new_path==[]
print('道路封闭导致无路可走')'''),
ex('迁移：空仓库地图','新建 7×7 无障碍地图，从 (0,0) 到 (6,6)，独立核对最短步数 12。','''warehouse=np.zeros((7,7),dtype=int)
route,_=astar(warehouse,(0,0),(6,6))
assert validate_path(warehouse,route,(0,0),(6,6)) and len(route)-1==12
print(route)''')],
[('NetworkX A* 算法文档','https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.shortest_paths.astar.astar_path.html')],
'B05、B13；元组、字典、生成器与 heapq。','最短性依赖非负边代价与有效启发式；改成对角移动、带电量状态或动态路况后必须重审建模和启发式。')

make(26,'Qlearning学习迷宫','怎样让智能体通过尝试学会走迷宫？',
'任务：在偶尔执行错方向的网格环境里学习到达终点。规则与奖励明确可查，训练和评价使用独立随机数。',[
('## 1. 环境状态、动作与随机转移\n状态是网格位置，四个动作对应上下左右；10% 概率执行随机动作。碰墙留在原地。奖励由任务定义：终点 +10、普通移动 −0.1、撞墙 −0.3。','''
board=np.array([[0,0,0,0,0],[0,1,1,1,0],[0,0,0,1,0],[1,1,0,0,0],[0,0,0,1,0]])
start=(0,0);goal=(4,4);moves=[(-1,0),(1,0),(0,-1),(0,1)]
def transition(state,action,rng,grid=board,slip=.1):
    if rng.random()<slip:action=int(rng.integers(4))
    dr,dc=moves[action];candidate=(state[0]+dr,state[1]+dc)
    valid=0<=candidate[0]<grid.shape[0] and 0<=candidate[1]<grid.shape[1] and grid[candidate]==0
    nxt=candidate if valid else state;done=nxt==goal
    return nxt,10. if done else (-.1 if valid else -.3),done
def state_id(state):return state[0]*board.shape[1]+state[1]
'''),
('## 2. Bellman 更新与探索\nQ(s,a) 估计采取动作后的折扣回报。非终止目标 r+gamma×maxQ，真正到达终点时不再 bootstrap。epsilon-greedy 初期探索多，后期降低；截断上限是采样限制，不伪装成终点。','''
def learn(episodes=1600,seed=26,gamma=.95):
    rng=np.random.default_rng(seed);Q=np.zeros((board.size,4));returns=[];successes=[]
    for episode in range(episodes):
        state=start;total=0.;done=False;epsilon=max(.05,.9*(1-episode/episodes))
        for step in range(70):
            sid=state_id(state)
            action=int(rng.integers(4)) if rng.random()<epsilon else int(Q[sid].argmax())
            nxt,reward,done=transition(state,action,rng)
            target=reward if done else reward+gamma*Q[state_id(nxt)].max()
            Q[sid,action]+=.15*(target-Q[sid,action]);state=nxt;total+=reward
            if done:break
        returns.append(total);successes.append(done)
    return Q,np.array(returns),np.array(successes)
Q,returns,successes=learn()
assert np.isfinite(Q).all()
'''),
('## 3. 冻结策略后独立评价\n评价不再更新 Q，也不使用训练探索率。随机策略作为基线。记录成功率、步数和总奖励，避免某种奖励设计让“高分”掩盖未完成任务。','''
def evaluate(Q,random_policy=False,seed=1026,episodes=180,grid=board):
    rng=np.random.default_rng(seed);rows=[]
    for _ in range(episodes):
        state=start;total=0.;done=False
        for step in range(70):
            action=int(rng.integers(4)) if random_policy else int(Q[state_id(state)].argmax())
            state,reward,done=transition(state,action,rng,grid);total+=reward
            if done:break
        rows.append((int(done),step+1,total))
    return np.array(rows)
learned=evaluate(Q);random_result=evaluate(Q,random_policy=True)
assert learned.shape==(180,3)
'''),
('## 4. 学习曲线与策略输出\n训练回报受探索影响，不能替代独立评价。保存 Q 表和每个状态的贪心动作；随机成功不等于保证最优。','''
fig,ax=plt.subplots();ax.plot(np.convolve(returns,np.ones(80)/80,mode='valid'));ax.set_xlabel('训练回合');ax.set_ylabel('80 回合平均回报');ax.set_title('Q-learning 学习曲线')
fig.tight_layout();fig.savefig(OUT_DIR/'learning.png');plt.close(fig)
np.save(OUT_DIR/'q_table.npy',Q)
finish({'learned_success_rate':float(learned[:,0].mean()),'random_success_rate':float(random_result[:,0].mean()),'learned_mean_steps':float(learned[:,1].mean()),'learned_mean_return':float(learned[:,2].mean()),'training_episodes':len(returns),'evaluation_episodes':len(learned)})
''')],[
ex('补全：终止目标','终点奖励 10，旧Q=2，学习率0.2。终止时不得加下一状态价值，更新结果应为3.6。','''old=2.;reward=10.;alpha=.2;done=True;next_value=100.
target=reward if done else reward+.95*next_value
updated=old+alpha*(target-old)
assert np.isclose(updated,3.6)
print(updated)'''),
ex('修改：检查确定性动作','slip=0，从起点向右应到 (0,1)，向上撞墙应留原地，核对奖励。','''rng=np.random.default_rng(0)
s,r,done=transition(start,3,rng,slip=0)
assert s==(0,1) and np.isclose(r,-.1) and not done
s,r,done=transition(start,0,rng,slip=0)
assert s==start and np.isclose(r,-.3)
print('环境边界检查通过')'''),
ex('迁移：道路改变后的旧策略','封闭 (0,3)，用旧 Q 表在新图评价，不重新训练；打印性能变化，解释何时需要重新规划或学习。','''changed=board.copy();changed[0,3]=1
snapshot=Q.copy();shifted=evaluate(Q,seed=2026,grid=changed)
assert np.array_equal(Q,snapshot) and shifted.shape==(180,3)
print('原成功率',learned[:,0].mean(),'变更后',shifted[:,0].mean())''')],
[('Gymnasium FrozenLake 与环境建模','https://gymnasium.farama.org/environments/toy_text/frozen_lake/'),('Sutton 与 Barto 强化学习教材','http://incompleteideas.net/book/the-book-2nd.html')],
'B16 更新循环、B25 网格状态。','环境由作者定义，真实拥堵更复杂。固定小地图优先用规划；RL 的价值需与交互代价比较，不声称学得即最优。')

make(27,'改进是否可靠','修改模型后，怎样判断改进是否可靠？',
'任务：比较线性模型与加入三次多项式的分类器，用训练子样本重复实验、配对误差和区间避免只展示最好一次。',[
('## 1. 固定测试与比较问题\n用已知非线性机制的双月牙数据。训练和测试一次划分后固定；比较路线预先指定，不看测试结果调多项式阶数。标签是教学生成，不是真实业务事实。','''
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
from sklearn.linear_model import LogisticRegression
X,y=make_moons(n_samples=1200,noise=.25,random_state=27)
tr,te=train_test_split(np.arange(len(y)),test_size=.3,stratify=y,random_state=27)
seeds=list(range(8));base_errors=[];new_errors=[];timings=[]
'''),
('## 2. 同一训练子样本上的配对比较\n每个种子抽取训练集的80%，两种模型共享该子样本和测试集。去掉多项式就是消融对照；不同种子的测试结果不是完全独立，因为共享测试样本。','''
for seed in seeds:
    rng=np.random.default_rng(seed);subset=rng.choice(tr,size=int(.8*len(tr)),replace=False)
    base=make_pipeline(StandardScaler(),LogisticRegression(max_iter=500))
    improved=make_pipeline(PolynomialFeatures(3,include_bias=False),StandardScaler(),LogisticRegression(C=1,max_iter=500))
    t0=time.perf_counter();base.fit(X[subset],y[subset]);base_time=time.perf_counter()-t0
    t0=time.perf_counter();improved.fit(X[subset],y[subset]);new_time=time.perf_counter()-t0
    base_errors.append((base.predict(X[te])!=y[te]).astype(float));new_errors.append((improved.predict(X[te])!=y[te]).astype(float));timings.append((base_time,new_time))
base_errors=np.array(base_errors);new_errors=np.array(new_errors)
per_seed_gain=base_errors.mean(1)-new_errors.mean(1)
'''),
('## 3. 对测试样本做配对 bootstrap\n先固定已训练的模型，再对每个测试样本的平均误差差值重采样。区间描述这些固定模型在独立同分布测试样本上的不确定性，不是完整训练算法总体保证；时间序列需区块重采样。','''
per_case_gain=(base_errors-new_errors).mean(axis=0)
def paired_bootstrap(values,seed=27,repeats=1500):
    rng=np.random.default_rng(seed)
    samples=rng.choice(values,size=(repeats,len(values)),replace=True).mean(1)
    return np.quantile(samples,[.025,.975]),samples
interval,bootstrap=paired_bootstrap(per_case_gain)
assert interval[0]<=interval[1]
'''),
('## 4. 报告收益、变异与成本\n正差值表示新方案少犯错；若区间跨0，需要承认证据不足。训练成本、测试集构成和每次结果同时交付，不只保留最佳种子。','''
fig,axes=plt.subplots(1,2,figsize=(10,4));axes[0].bar(seeds,per_seed_gain);axes[0].axhline(0,color='black');axes[0].set_title('各训练子样本的准确率提升')
axes[1].hist(bootstrap,bins=30);axes[1].axvline(interval[0],color='red');axes[1].axvline(interval[1],color='red');axes[1].set_title('配对 bootstrap 均值分布')
fig.tight_layout();fig.savefig(OUT_DIR/'uncertainty.png');plt.close(fig)
np.savez(OUT_DIR/'paired_errors.npz',baseline=base_errors,improved=new_errors,test_indices=te)
finish({'baseline_accuracy':float(1-base_errors.mean()),'improved_accuracy':float(1-new_errors.mean()),'mean_accuracy_gain':float(per_case_gain.mean()),'paired_95_percent_interval':interval.tolist(),'per_seed_gain':per_seed_gain.tolist(),'mean_training_seconds':np.mean(timings,axis=0).tolist(),'interval_scope':'fixed trained models; iid test examples; not full training uncertainty'})
''')],[
ex('补全：配对方向','核对基线错误减新方案错误是改善方向；所有样本改善0.2时区间必须也是0.2。','''ci,draws=paired_bootstrap(np.full(30,.2),repeats=100)
assert np.allclose(ci,[.2,.2]) and np.allclose(draws,.2)
print(ci)'''),
ex('修改：报告最差种子','计算提升最小的种子与训练成本比，不丢掉差结果。','''worst=int(np.argmin(per_seed_gain));ratio=float(np.mean(timings,axis=0)[1]/np.mean(timings,axis=0)[0])
assert per_seed_gain[worst]==per_seed_gain.min() and ratio>0
print('最弱种子',seeds[worst],'改善',per_seed_gain[worst],'训练耗时比',ratio)'''),
ex('迁移：有时间相关性时按区块抽样','把连续24个损失作为一个区块，先计算每块均值再 bootstrap；不把相邻时刻当完全独立。','''rng=np.random.default_rng(3);daily=rng.normal(.1,.2,20);hourly=np.repeat(daily,24)+rng.normal(0,.02,480)
blocks=hourly.reshape(20,24).mean(1)
ci,draws=paired_bootstrap(blocks,repeats=500)
assert len(blocks)==20 and ci[0]<=ci[1]
print('以天为独立单位的区间',ci)''')],
[('sklearn 交叉验证统计比较','https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_stats.html'),('可靠实验评价论文','https://arxiv.org/html/2108.13264v3')],
'B10、B16；配对比较、重采样与统计口径。','这里的区间条件于训练好的模型；不能把共享测试集的8个结果当8组独立测试数据。统计差异也不等于业务价值。')

RENT_API='''
from fastapi import FastAPI
from pydantic import BaseModel,Field,ConfigDict
class RentRequest(BaseModel):
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False)
    area: float=Field(gt=0,le=500)
    rooms: int=Field(ge=1,le=10)
    floor: int=Field(ge=1,le=80)
def create_app(estimator):
    app=FastAPI(title='教学租金预测 API')
    @app.get('/health')
    def health():return {'status':'ok','model_version':'teaching-v1'}
    @app.post('/predict')
    def predict(payload:RentRequest):
        value=max(0.,float(estimator.predict([[payload.area,payload.rooms,payload.floor]])[0]))
        return {'predicted_rent':round(value,2),'unit':'教学货币/月','model_version':'teaching-v1'}
    return app
'''

make(28,'预测模型本地API','怎样把训练好的模型做成可调用的预测接口？',
'任务：保存、重新加载教学租金模型，用 FastAPI 提供本地接口，并实际发送有效与无效请求。',[
('## 1. 固定特征顺序与目标单位\n租金由面积、房间数、楼层和噪声生成。输入顺序是服务契约的一部分；测试数据不参与训练。这不是实际房价模型。','''
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
rng=np.random.default_rng(28)
X=np.column_stack([rng.uniform(25,180,500),rng.integers(1,6,500),rng.integers(1,25,500)])
y=600+15*X[:,0]+180*X[:,1]+4*X[:,2]+rng.normal(0,70,500)
model=RandomForestRegressor(n_estimators=60,max_depth=8,random_state=28,n_jobs=1).fit(X[:380],y[:380])
mae=float(mean_absolute_error(y[380:],model.predict(X[380:])))
feature_names=['area','rooms','floor']
'''),
('## 2. 保存与重载一致性\njoblib 只能加载可信文件。保存后重载，同一输入的输出必须一致；另存特征顺序、数据性质和目标单位，防止交接时只剩一个无说明的二进制文件。','''
joblib.dump(model,OUT_DIR/'model.joblib');loaded=joblib.load(OUT_DIR/'model.joblib')
assert np.allclose(loaded.predict(X[380:385]),model.predict(X[380:385]))
(OUT_DIR/'model_metadata.json').write_text(json.dumps({'features':feature_names,'model_version':'teaching-v1','data':'synthetic','target_unit':'教学货币/月'},ensure_ascii=False,indent=2),encoding='utf-8')
'''),
('## 3. 输入校验与响应\nPydantic 限制字段、数值范围和额外字段；非法请求返回422，不进入模型。api.py 是同一接口的独立入口，只在127.0.0.1启动。',RENT_API+'''
app=create_app(loaded)
'''),
('## 4. 通过真实路由测试契约\nTestClient 走 FastAPI 路由和验证，不只是调用预测函数。检查健康、正常请求、负面积和额外字段；接口正确不代表模型在真实业务中准确。','''
from fastapi.testclient import TestClient
with TestClient(app) as client:
    health=client.get('/health');valid=client.post('/predict',json={'area':80,'rooms':3,'floor':5})
    invalid=client.post('/predict',json={'area':-1,'rooms':3,'floor':5})
    extra=client.post('/predict',json={'area':80,'rooms':3,'floor':5,'secret':1})
assert health.status_code==valid.status_code==200 and invalid.status_code==extra.status_code==422
assert valid.json()['predicted_rent']>=0
(OUT_DIR/'api_responses.json').write_text(json.dumps({'valid':valid.json(),'invalid_status':invalid.status_code,'extra_field_status':extra.status_code},ensure_ascii=False,indent=2),encoding='utf-8')
barplot(['正常预测','负面积','额外字段'],[valid.status_code,invalid.status_code,extra.status_code],'实际 HTTP 状态码')
finish({'test_mae':mae,'valid_request_status':valid.status_code,'invalid_request_status':invalid.status_code,'serialization_match':True,'sample_prediction':valid.json()})
''')],[
ex('补全：响应契约','调用接口，核对响应含预测值、单位和版本。','''with TestClient(app) as c:r=c.post('/predict',json={'area':60,'rooms':2,'floor':4})
assert r.status_code==200 and set(r.json())=={'predicted_rent','unit','model_version'}
print(r.json())'''),
ex('修改：缺失与类型错误','缺失floor、rooms传abc、面积0，均应返回422。','''bad=[{'area':60,'rooms':2},{'area':60,'rooms':'abc','floor':2},{'area':0,'rooms':2,'floor':2}]
with TestClient(app) as c:
    for body in bad:assert c.post('/predict',json=body).status_code==422
print('非法请求已拒绝')'''),
ex('迁移：批量客户端','客户端顺序发送三条记录，一条输入对应一条输出。','''requests=[{'area':a,'rooms':2,'floor':3} for a in [40,60,80]]
with TestClient(app) as c:responses=[c.post('/predict',json=body) for body in requests]
assert len(responses)==3 and all(r.status_code==200 for r in responses)
print([r.json()['predicted_rent'] for r in responses])''')],
[('FastAPI 测试','https://fastapi.tiangolo.com/tutorial/testing/'),('FastAPI 请求体','https://fastapi.tiangolo.com/tutorial/body/'),('sklearn 模型持久化','https://scikit-learn.org/stable/model_persistence.html')],
'B07、B10；序列化、输入校验与 HTTP。','只用于本地教学；真实服务还需要鉴权、日志与监测。只载入可信序列化文件。',
extra_files={'api.py':"from pathlib import Path\nimport joblib\n"+dedent(RENT_API)+"\nmodel=joblib.load(Path(__file__).resolve().parent/'outputs/model.joblib')\napp=create_app(model)\n",'本地服务说明.md':'先运行 solution.py 生成模型。在本目录用课程 Python 执行 `python -m uvicorn api:app --host 127.0.0.1 --port 8000`，再浏览 http://127.0.0.1:8000/docs。仅本地使用。'})

make(29,'数据漂移与效果监测','新数据变化时，怎样发现模型可能失效？',
'任务：区分数据质量、输入分布变化与标签延迟后的实际效果下降。报警用于调查，不等于自动重训。',[
('## 1. 固定模型与三个监测窗口\n生成正常、输入偏移、关系反转三种情况。关系反转时输入可能不变但模型失效；所有机制与随机种子公开，便于手工验证。','''
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from scipy.stats import ks_2samp
rng=np.random.default_rng(29)
reference=rng.normal(size=(1000,2));reference_y=(reference[:,0]+.6*reference[:,1]+rng.normal(0,.4,1000)>0).astype(int)
model=LogisticRegression().fit(reference[:700],reference_y[:700])
windows={}
for name,shift,flip in [('正常',0,False),('输入偏移',1.5,False),('概念改变',0,True)]:
    xx=rng.normal(size=(400,2));xx[:,0]+=shift
    yy=(xx[:,0]+.6*xx[:,1]+rng.normal(0,.4,400)>0).astype(int)
    if flip:yy=1-yy
    windows[name]=(xx,yy)
baseline_accuracy=float(accuracy_score(reference_y[700:],model.predict(reference[700:])))
'''),
('## 2. 输入质量先于预测\n缺失、无限值和列数错误需要单独报告。明确隔离坏记录，不把所有缺失填成0后宣称数据正常。','''
def quality_mask(X):
    if X.ndim!=2 or X.shape[1]!=2:raise ValueError('要求两列特征')
    return np.isfinite(X).all(axis=1)
broken=windows['正常'][0].copy();broken[0,0]=np.nan;broken[1,1]=np.inf
valid=quality_mask(broken)
assert valid.sum()==len(broken)-2
'''),
('## 3. 标签前后使用不同指标\nKS统计量反映输入差异，阈值0.2只是教学假设。无标签时 accuracy 必须未知；标签到达后才能判断效果。输入漂移既不是性能下降的充分条件，也不是必要条件。','''
def monitor(X,labels=None):
    mask=quality_mask(X);clean=X[mask]
    if len(clean)==0:raise ValueError('没有可用记录')
    ks=[float(ks_2samp(reference[:,j],clean[:,j]).statistic) for j in range(2)]
    accuracy=None if labels is None else float(accuracy_score(np.asarray(labels)[mask],model.predict(clean)))
    return {'valid_rows':int(mask.sum()),'invalid_rows':int((~mask).sum()),'ks_statistics':ks,'input_drift':max(ks)>.2,'accuracy':accuracy,'performance_alarm':accuracy is not None and accuracy<baseline_accuracy-.1}
unlabeled={n:monitor(x) for n,(x,y) in windows.items()}
labeled={n:monitor(x,y) for n,(x,y) in windows.items()}
assert all(r['accuracy'] is None for r in unlabeled.values())
'''),
('## 4. 从报警到行动\n输入异常先查管道与人群组成，效果下降再评估回退或新模型。候选必须在后续留出数据上验收，不能用同一报警窗口训练并验收。','''
report={'reference_accuracy':baseline_accuracy,'before_labels':unlabeled,'after_labels':labeled,'quality_example':monitor(broken)}
(OUT_DIR/'monitoring_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
fig,axes=plt.subplots(1,2,figsize=(10,4));names=list(windows)
axes[0].bar(names,[max(labeled[n]['ks_statistics']) for n in names]);axes[0].axhline(.2,color='red');axes[0].set_title('输入漂移 KS')
axes[1].bar(names,[labeled[n]['accuracy'] for n in names]);axes[1].axhline(baseline_accuracy,color='red');axes[1].set_title('标签到达后的准确率')
fig.tight_layout();fig.savefig(OUT_DIR/'monitoring.png');plt.close(fig)
finish({'reference_accuracy':baseline_accuracy,'window_results':labeled,'invalid_rows_detected':int((~valid).sum()),'thresholds':'teaching assumptions; calibrate for real deployment'})
''')],[
ex('补全：无标签口径','隐藏正常窗口标签，检查准确率未知而输入统计仍可用。','''r=monitor(windows['正常'][0])
assert r['accuracy'] is None and len(r['ks_statistics'])==2
print(r)'''),
ex('修改：错误数据结构','一列数据必须报错，坏记录必须检出。','''try:monitor(np.zeros((10,1)))
except ValueError:pass
else:raise AssertionError('schema错误未拒绝')
assert monitor(broken)['invalid_rows']==2
print('通过')'''),
ex('迁移：同输入不同关系','保持完全相同的X，反转标签；输入统计相同，两种准确率之和为1。','''xx,yy=windows['正常'];a=monitor(xx,yy);b=monitor(xx,1-yy)
assert a['ks_statistics']==b['ks_statistics']
assert np.isclose(a['accuracy']+b['accuracy'],1)
print(a['accuracy'],b['accuracy'])''')],
[('SciPy KS 检验','https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html'),('sklearn 常见陷阱','https://scikit-learn.org/stable/common_pitfalls.html')],
'B09、B27—B28；质量检查、分布统计与延迟标签。','报警不等于失效证明；真实阈值须结合样本量、错误代价和多重检验确定。')

BIKE_API='''
from fastapi import FastAPI
from pydantic import BaseModel,Field,ConfigDict
class BikeRequest(BaseModel):
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False)
    hour:int=Field(ge=0,le=23)
    weekday:int=Field(ge=0,le=6)
    month:int=Field(ge=1,le=12)
    year_index:int=Field(ge=0,le=20)
    holiday:int=Field(ge=0,le=1)
    workingday:int=Field(ge=0,le=1)
    lag24:float=Field(ge=0)
    lag168:float=Field(ge=0)
def create_bike_app(bundle):
    app=FastAPI(title='小时单车需求教学 API')
    @app.get('/health')
    def health():return {'status':'ok','model':bundle['selected']}
    @app.post('/predict')
    def predict(payload:BikeRequest):
        row=pd.DataFrame([payload.model_dump()])[bundle['features']]
        prediction=max(0.,float(bundle['model'].predict(row)[0]))
        return {'predicted_rentals':round(prediction,2),'unit':'次/小时','horizon':'one-hour-ahead rolling'}
    return app
'''

make(30,'单车需求完整项目','怎样独立完成可复现、可交接的预测项目？',
'任务：用真实 UCI 小时单车记录预测下一小时需求，完成数据契约、时序评价、模型交接与本地接口。先阅读独立任务书，再对照参考实现。',[
('## 1. 明确预测时点并读取真实数据\n在目标小时t开始前预测cnt；只使用日历信息与t−24、t−168小时已观测需求。实际天气、casual和registered均禁止作输入。原数据有缺失小时，先补齐时间索引，再shift，避免把24条记录误作24小时。','''
import hashlib
import pandas as pd
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error
data_path=COURSE_ROOT/'assets/data/bike_hour.csv'
if not data_path.is_file():raise FileNotFoundError(f'缺少离线 UCI 数据：{data_path}')
raw=pd.read_csv(data_path)
required={'dteday','hr','cnt','holiday','workingday'}
assert required.issubset(raw.columns)
raw['timestamp']=pd.to_datetime(raw['dteday'])+pd.to_timedelta(raw['hr'],unit='h')
assert not raw['timestamp'].duplicated().any()
hourly=raw.set_index('timestamp').sort_index().reindex(pd.date_range(raw['timestamp'].min(),raw['timestamp'].max(),freq='h'))
print('原记录',len(raw),'缺失小时',int(hourly['cnt'].isna().sum()))
'''),
('## 2. 只构造当时可用特征\nshift(24)先移位再取值，绝不能使用未来信息补齐历史。缺少目标或历史滞后值的行删除并报告。按时间70%/15%/15%分割；测试阶段是逐小时滚动预测，可使用当时已发生的早期测试需求，不是固定起点预测整季。','''
frame=pd.DataFrame(index=hourly.index)
frame['hour']=frame.index.hour;frame['weekday']=frame.index.dayofweek;frame['month']=frame.index.month
frame['year_index']=frame.index.year-2011
frame['holiday']=hourly['holiday'];frame['workingday']=hourly['workingday']
frame['lag24']=hourly['cnt'].shift(24);frame['lag168']=hourly['cnt'].shift(168);frame['target']=hourly['cnt']
frame=frame.dropna()
features=['hour','weekday','month','year_index','holiday','workingday','lag24','lag168']
forbidden={'cnt','casual','registered','instant','temp','atemp','hum','windspeed','weathersit'}
assert not set(features)&forbidden
n=len(frame);a=int(.7*n);b=int(.85*n)
train,validation,test=frame.iloc[:a],frame.iloc[a:b],frame.iloc[b:]
assert train.index.max()<validation.index.min()<test.index.min()
assert len(test)>100
'''),
('## 3. 验证集选择简单模型与实用模型\n基线是昨天同一小时需求；另比较 Ridge 和 HistGradientBoosting。固定候选与预算，验证集按MAE选方案，再在训练+验证上重拟合一次。最终测试不参与选择。','''
candidates={'ridge':make_pipeline(StandardScaler(),Ridge(alpha=10)),
            'hist_gradient_boosting':HistGradientBoostingRegressor(max_iter=110,max_leaf_nodes=15,l2_regularization=2,early_stopping=False,random_state=30)}
validation_scores={}
for name,candidate in candidates.items():
    candidate.fit(train[features],train['target'])
    validation_scores[name]=float(mean_absolute_error(validation['target'],np.maximum(0,candidate.predict(validation[features]))))
selected=min(validation_scores,key=validation_scores.get)
model=candidates[selected];development=frame.iloc[:b];model.fit(development[features],development['target'])
predicted=np.maximum(0,model.predict(test[features]));baseline=test['lag24'].to_numpy()
model_mae=float(mean_absolute_error(test['target'],predicted));baseline_mae=float(mean_absolute_error(test['target'],baseline))
rmse=float(np.sqrt(mean_squared_error(test['target'],predicted)))
'''),
('## 4. 保存可交接结果、错误与模型卡\n记录完整数据哈希、日期范围、输入列和预测口径。图展示测试开头168个小时，误差表覆盖全部测试。改善必须来自实际计算；疫情、城市与用户变化不在此历史数据保证内。','''
result=test[['target','lag24']].copy();result['prediction']=predicted;result['absolute_error']=np.abs(result['target']-predicted)
result.to_csv(OUT_DIR/'predictions.csv',encoding='utf-8-sig',index_label='timestamp')
bundle={'model':model,'features':features,'selected':selected};joblib.dump(bundle,OUT_DIR/'model.joblib')
metadata={'dataset':'UCI Bike Sharing hourly','sha256':hashlib.sha256(data_path.read_bytes()).hexdigest(),'features':features,'excluded':sorted(forbidden),
          'train_end':str(train.index.max()),'validation_end':str(validation.index.max()),'test_start':str(test.index.min()),'test_end':str(test.index.max()),
          'forecast_protocol':'one-hour-ahead rolling; lag24 and lag168 already observed; no observed future weather','selected':selected,'validation_mae':validation_scores}
(OUT_DIR/'model_card.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
fig,ax=plt.subplots(figsize=(10,4));sample=result.iloc[:168];ax.plot(sample.index,sample['target'],label='真实');ax.plot(sample.index,sample['prediction'],label='预测');ax.legend();ax.set_title('独立测试最初168个可评价小时');fig.autofmt_xdate();fig.tight_layout();fig.savefig(OUT_DIR/'forecast.png');plt.close(fig)
print('最差的五个小时',result.nlargest(5,'absolute_error')[['target','prediction','absolute_error']])
'''),
('## 5. 接口验证与完整项目验收\nAPI 接收已在外部准备好的日历与历史滞后值，拒绝负需求和额外字段。服务不偷偷回看未来CSV。用测试中一条记录核对批预测和API输出一致。',BIKE_API+'''
from fastapi.testclient import TestClient
app=create_bike_app(bundle)
payload=test[features].iloc[0].to_dict()
for key in ['hour','weekday','month','year_index','holiday','workingday']:payload[key]=int(payload[key])
with TestClient(app) as client:
    response=client.post('/predict',json=payload)
    invalid=client.post('/predict',json={**payload,'lag24':-1})
assert response.status_code==200 and invalid.status_code==422
assert abs(response.json()['predicted_rentals']-predicted[0])<.011
finish({'raw_rows':len(raw),'usable_rows':len(frame),'test_rows':len(test),'selected_model':selected,'validation_mae':validation_scores,'test_mae':model_mae,'yesterday_baseline_mae':baseline_mae,'test_rmse':rmse,'relative_mae_change':(baseline_mae-model_mae)/baseline_mae,'api_status':response.status_code,'invalid_api_status':invalid.status_code,'leakage_columns_excluded':True})
''')],[
ex('补全：检查时间与泄漏','核对禁止列不在输入中，随机挑一个测试时间检查lag24确实来自前24小时。','''assert not set(features)&forbidden
t=test.index[20]
assert np.isclose(test.loc[t,'lag24'],hourly.loc[t-pd.Timedelta(hours=24),'cnt'])
assert train.index.max()<validation.index.min() and validation.index.max()<test.index.min()
print('时间与特征检查通过')'''),
ex('修改：按天定位错误','将全部测试误差按日期聚合，打印误差最大的五天；不能只挑好看的168小时展示。','''daily=result['absolute_error'].groupby(result.index.date).mean().sort_values(ascending=False)
assert len(daily)>5 and np.isclose(result['absolute_error'].mean(),model_mae)
print(daily.head())'''),
ex('迁移：把预测变成人员安排','假设一名调度员可处理80次需求，按预测向上取整配置；与按训练平均需求配置对照，报告工时和未覆盖需求。成本与能力是假设。','''capacity=80
staff=np.maximum(1,np.ceil(predicted/capacity)).astype(int)
fixed=int(max(1,np.ceil(train['target'].mean()/capacity)))
shortage=np.maximum(0,test['target'].to_numpy()-staff*capacity)
fixed_shortage=np.maximum(0,test['target'].to_numpy()-fixed*capacity)
assert len(staff)==len(test) and np.all(staff>=1) and np.all(shortage>=0)
print({'forecast_staff_hours':int(staff.sum()),'fixed_staff_hours':fixed*len(test),'forecast_uncovered':float(shortage.sum()),'fixed_uncovered':float(fixed_shortage.sum())})''')],
[('UCI Bike Sharing 数据集','https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset'),('sklearn 时序滞后特征案例','https://scikit-learn.org/stable/auto_examples/applications/plot_time_series_lagged_features.html'),('FastAPI 测试','https://fastapi.tiangolo.com/tutorial/testing/')],
'B01—B29；重点 B15、B27—B29。','没有未来天气预报数据，因此不使用实际天气字段。测试采用滚动单步协议；不能把该结果描述为固定起点的长期预测。真实部署需重新校准监测阈值。',
'真实 UCI Bike Sharing 小时数据，原作者 Hadi Fanaee-T 和 Joao Gama，2011—2012 Capital Bikeshare。assets/data/bike_hour.csv 由课程预先下载；原始来源、许可、哈希见assets与第三方清单。cnt=casual+registered，二者禁止作为输入。',
extra_files={'api.py':"from pathlib import Path\nimport pandas as pd\nimport joblib\n"+dedent(BIKE_API)+"\nbundle=joblib.load(Path(__file__).resolve().parent/'outputs/model.joblib')\napp=create_bike_app(bundle)\n",
'独立任务书.md':'# 单车需求项目：先做再看答案\n\n你的委托是提前一个小时估计租借需求，供调度团队配置能力。请提交：数据字段与可用时点表；按时间划分的训练、验证、测试；昨天同小时基线及一个可运行模型；全部测试逐时误差；可本地调用且校验输入的API；一页建议与更新触发条件。\n\n禁止把casual、registered、当小时真实天气作为已知未来输入。必须补齐小时索引后创建滞后特征。允许使用测试期内已经发生的历史需求，但须声明滚动单步预测协议。\n\n验收：无数据泄漏、API与批预测一致、非法请求拒绝、模型来源和数据哈希可追溯。模型没有胜过基线也要解释，不能改测试集隐藏失败。',
'本地服务说明.md':'先运行 solution.py。使用课程 Python 在本目录执行 `python -m uvicorn api:app --host 127.0.0.1 --port 8001`，访问 http://127.0.0.1:8001/docs。输入约定见 model_card.json。服务只接收当时已知的特征，不自动读取未来数据。'})

print('已生成 B16—B30；运行验证器后才能声称案例通过。')

