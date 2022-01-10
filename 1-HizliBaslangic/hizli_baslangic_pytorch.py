# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 08:36:32 2022

@author: 104863
"""

'''
Veriler İLe Çalışma:   
Pytorch verilerle çalışmak için iki temel ögeye sahiptir.
    torch.utils.data.DataLoader
    torch.utils.data.Dataset. Dataset
'''
# İlgili kütüphanelerin yüklenmesi:
import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor,Lambda,Compose
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Pytorch veri setlerini TorchText , TorchVision ve TorchAudio gibi kitaplıklarda tutar. Kullanıcıya bunlar üzerinden sunar.
# Bu hızlı eğitim için TorchVision date seti içerisindeki FashionMNIST dateseti kullanacağız.
# TorchVision'a ait iki adet parametre vardır.transform ve target_transform.

# Eğitim Veri Kümesinin Yüklenmesi:
training_data = datasets.FashionMNIST(
                        root='data',
                        train=True,
                        download=True,
                        transform=ToTensor(),
                        )
# Test Veri Kümesinin Yüklenmesi:
test_data = datasets.FashionMNIST(
                    root='data',
                    train=False,
                    download=True,
                    transform=ToTensor(),
                        )
print('Egitim ve Test verileri yüklendi.')

# DataLoader kullanarak veri kümelerimizi 64'lük yeni yinelenebilir bir veri kümesine çeviriyoruz. Bu yapı oldukça esnektir.
# DataLoader'in oluşturulması:
batch_size = 64
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)
# dataloader'ın tiplerine hızlıca göz atalım.
'''
for X,y in test_dataloader:
    print(f"X'in şekli: {X.shape} ")
    print(f"y'nin şekli/tipi: {y.shape , y.dtype} ")
'''

# Model Oluşturma:
'''
Pytorch'da bir sinir ağı tanımlamak için nn.Module'dan biris alınan bir sınıf oluşturulur.__init__ yapıcılar eklenir.
Ağdaki işlemleri hızlandırmak için sinir ağını GPU ya taşıyoruz.
'''
# Eğitim için CPU ya da GPU kontrolü.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Eğitim için {device} kullanılacaktır.")

# Modelin Tanımlanması:
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
# modelimizden nesne oluşturduk.
model = NeuralNetwork().to(device)
print(model)

# Model parametlerini optimize edelim.
'''
Bütün YSA(Yapay Sinir Ağı) da geçerli olmak kaydıyla bir modeli eğitmek için kayıp fonksiyonuna
ve modeli optimize ediciye ihtiyacımız vardır.
'''
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3) # lr olarak tanımlanan, learning rate'tir. yani modelin ne kadar hızlı öğreneceğidir.
        
# Eğitim sınıfının tanımlanması:
def train(dataloader,model,loss_fn,optimizer):
    size = len(dataloader.dataset)
    model.train() # model eğitimi burada başlar.
    for batch, (X,y) in enumerate(dataloader):# dataloader enumerate veri yapısına çevrilir. X eğitim kümesi ve y sonuç kümesi gpu üzerine verilir.
        X,y = X.to(device),y.to(device)
        # Tahmin hatasının hesaplanması.
        pred = model(X) # X verileri kullanılarak pred yani tahmin verileri elde edilir.
        loss =  loss_fn(pred,y) # tahmin verileri ile eğitimde ki gerçek sonuçlar(y) loss_fn ye verilir. ve loss değeri elde edilir.
        # Geri Besleme(modeli iyileştirme):
        optimizer.zero_grad() # geri besleme değerleri en başta 0 olarak kabul edilir.
        loss.backward() # loss değeri ysa ya ceza niteliğinde geri besleme olarak verilir. ve kendini günceller, iyileştirir.
        optimizer.step()
        
        if batch % 100 == 0: #artık verilecek veri kümesi kalmadıysa, değerler hesaplanır.
            loss,current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# Test sınıfının tanımlanması:            
# Modelin performansını test veri kümesi ile karşılaştırıyoruz.
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0,0 # test kaybı ve modelin doğruluğu en başta 0 kabul edilir.
    with torch.no_grad(): # random değeri olan requires_grade parametresi false'dır.Gradyan hesaplamasını devre dışı bırakmaya yarar.true olursa hesaplamayı devre dışı bırakır ve bellek tüketimini azaltır.
        for X,y in dataloader:
            X,y = X.to(device),y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred,y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Sonucunuz: \n Accuracy: {(100*correct):>0.1f}%, Ortalama Kayıp: {test_loss:>8f} \n")
    
# Eğitimin bütün veri setlerini kullanarak kaç defa tekrar edileceği parametresine epochs denir.
# Her epochs da modelin doğruluğunun arttığını ve kaybın azaldığını görmek isteriz.
epochs = 20
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model,loss_fn,optimizer)
    test(test_dataloader, model, loss_fn)
print(f"{epochs} Epoch Tamamlandı!")

# Modeli kaydetme:
torch.save(model.state_dict(), "model.pth")
print("Modeliniz başarılı bir şekilde kaydedildi.")

'''
# Modeli yükleme:
model = NeuralNetwork()
model.load_state_dict(torch.load("model.pth"))
'''

'''
Modelimiz tahmin için hazır.
Örnek:
    
classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Tahmin ettiğim sınıf: "{predicted}", Gerçek sınıfım: "{actual}"')
    
'''

'''
NOT:
    epoch değeri 20 iken model de elde ettiğimiz sonuç:
    Accuracy: 78.4%, Ortalama Kayıp: 0.616277
    şeklindedir. Epoch de learning rate değerleri değiştirilerek model tekrardan sınanabilir.
    Bu durumda accuracy ve loss değerleri gözlemlenebilir.
    Ya da model.pth yukarıda bahsedildiği üzere, eğitim için kaynak değeri tüketilmeden direkt olarak 
    tahmin işlemlerinde kullanılabilir.
'''


















