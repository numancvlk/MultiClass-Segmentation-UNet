#LIBRARIES
import torch
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.datasets import VOCSegmentation
from tqdm.auto import tqdm

#SCRIPTS
from Model import DEVICE

def saveCheckpoint(model, optimizer, epoch,checkpointFile = "myCheckpoint.pth"):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch
    }
    torch.save(checkpoint,checkpointFile)
    print("MODEL KAYDEDİLDİ")

def loadCheckpoint(checkpointFile,model,optimizer,epoch):
    checkpoint = torch.load(checkpointFile, map_location=DEVICE)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    epoch.load_state_dict(checkpoint["epoch"])
    print("MODEL YÜKLENDİ")

def printTrainTime(start,end,device):
    totalTime = end - start
    print(f"Train time is {totalTime} on the {device}")


def getLoaders(trainDatas,
               validationDatas,
               batchSize,
               numWorkers,
               pinMemory,
               trainTransform,
               validationTransform):
    
    trainDatas = VOCSegmentation(
        root="dataset",
        year="2012",
        image_set="train",
        download=True,
        target_transform=trainTransform
    )

    validationDatas = VOCSegmentation(
        root="dataset",
        year="2012",
        image_set="val",
        download=True,
        target_transform=validationTransform
    )


    trainDataLoader = DataLoader(
        dataset=trainDatas,
        batch_size=batchSize,
        shuffle=True,
        num_workers=numWorkers,
        pin_memory=pinMemory
    )

    validationDataLoader = DataLoader(
        dataset=validationDatas,
        batch_size=batchSize,
        shuffle=False,
        num_workers= numWorkers,
        pin_memory=pinMemory
    )

    return trainDataLoader, validationDataLoader


def trainStep(model: torch.nn.Module,
              dataLoader: torch.utils.data.DataLoader,
              optimizer: torch.optim.Optimizer,
              lossFn,
              scaler,
              device: torch.device = DEVICE):
    
    model.train()
    loop = tqdm(dataLoader)
    for batch, (xTrain,yTrain) in enumerate(loop):
        xTrain, yTrain = xTrain.to(device), yTrain.to(device)

        with torch.autocast(device_type=device):
            trainPred = model(xTrain)
            loss = lossFn(trainPred, yTrain)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loop.set_postfix(loss = loss.item())

def pixelAccuracy(pred,target):
    with torch.no_grad():
        predClasses = torch.argmax(pred,dim=1)
        correct = torch.eq(predClasses,target).float()
        acc = correct.sum() / correct.numel()
    
    return acc.item()

def meanIoU(pred,target,numClasses):
    """
    pred: modelin çıktısı, shape = (N, C, H, W)
    target: ground truth maskeleri, shape = (N, H, W)
    num_classes: toplam sınıf sayısı (Pascal VOC için 21)
    """

    #HANGİ SINIFI TAHMİN ETTİĞİ
    predClasses = torch.argmax(pred,dim=1)

    #Her bir sınıf için IoU değeri saklıyoruz listede
    ious = []

    for cls in range(numClasses):
        # predClasses = [[0,1,2],
        #                [1,2,0]]
        # cls = 1
        # predInds = [[False, True, False],
        #             [True, False, False]]
        predInds = torch.eq(pred,cls)
        
        #predInds ile aynı mantık
        targetInds = torch.eq(target,cls)

        intersection = (predInds & targetInds).sum().float() #yani sadece hem tahmin hem gerçek maskede aynı sınıfta olan pikseller

        union = (predInds | targetInds).sum().float() #yani tahmin veya gerçek maskede olan tüm pikseller

        if union == 0: #bu sınıf o görüntüde hiç yok
            iou = torch.tensor(0.1)
        else: 
            iou = intersection / union #normal IoU formülü
        
        #Bu sınıfın IoU değerini listeye ekle
        ious.append(iou)
    #Tüm sınıfların IoU ortalamasını alıyoruz
    return torch.mean(torch.stack(ious)).item()

def meanDiceScore(pred,target,numClasses):
    """
    pred: modelin çıktısı, shape = (N, C, H, W)
    target: ground truth maskeleri, shape = (N, H, W)
    numClasses: toplam sınıf sayısı (Pascal VOC için 21)
    """

    #Her piksel için tahmin edilen sınıfı buluyoruz
    predClasses = torch.argmax(pred,dim=1)

    #Her sınıfın Dice Score değerlerini saklamak için liste
    dices = []

    for cls in range(numClasses):
        #MEAN IoU ile aynı mantık
        predInds = torch.eq(predClasses,cls)
        targetInds = torch.eq(target, cls)


        intersection = (predInds & targetInds).sum().float() #hem tahmin hem ground truth'ta olan pikseller
        total = predInds.sum() + targetInds.sum() #tahmin ve ground truth maskesindeki toplam pikseller

        if total == 0: #Eğer o sınıfa ait hiç piksel yoksa
             dice = torch.tensor(1.0)
        else:
            dice = (2.0 * intersection) / total #Dice Score formülü
        
        dices.append(dice)
    #Tüm sınıfların Dice Score ortalamasını alıyoruz
    return torch.mean(torch.stack(dices)).item()

def savePredictions(model, dataLoader, epoch, device, saveDir="predictions", numSamples=4):
    """
    model: eğitilmiş model
    dataLoader: DataLoader (validation set)
    epoch: hangi epoch
    device: "cuda" veya "cpu"
    saveDir: tahminlerin kaydedileceği klasör
    numSamples: her epoch'ta kaç örnek kaydedilecek
    """

    model.eval()
    os.makedirs(saveDir, exist_ok=True)

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(dataLoader):
            images = images.to(device)
            masks = masks.to(device)

            # Tahmin
            outputs = model(images)
            predClasses = torch.argmax(outputs, dim=1)

            # Sadece ilk numSamples örneğini kaydet
            for i in range(min(numSamples, images.size(0))):
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                
                # Ground truth
                axes[0].imshow(masks[i].cpu(), cmap="jet", vmin=0, vmax=21)
                axes[0].set_title("Ground Truth")
                axes[0].axis("off")
                
                # Prediction
                axes[1].imshow(predClasses[i].cpu(), cmap="jet", vmin=0, vmax=21)
                axes[1].set_title("Prediction")
                axes[1].axis("off")

                # Kaydet
                save_path = os.path.join(saveDir, f"epoch{epoch}_sample{batch_idx*numSamples + i}.png")
                plt.savefig(save_path)
                plt.close(fig)

            # Sadece ilk batch’i kullanmak yeterli
            break

    print(f"[INFO] Epoch {epoch}: {numSamples} predictions saved to {saveDir}")