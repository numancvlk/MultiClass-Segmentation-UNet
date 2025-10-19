#LIBRARIES
import torch
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import OxfordIIITPet
from tqdm.auto import tqdm
from torchvision import transforms
from skimage import measure
import numpy as np

#SCRIPTS
from model import DEVICE


def saveCheckpoint(model, optimizer, epoch,checkpointFile = "myCheckpoint.pth"):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch
    }
    torch.save(checkpoint,checkpointFile)
    print("MODEL KAYDEDİLDİ")

def loadCheckpoint(checkpointFile,model,optimizer):
    checkpoint = torch.load(checkpointFile, map_location=DEVICE)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    epoch = checkpoint["epoch"]
    print("MODEL YÜKLENDİ")
    return epoch + 1

def printTrainTime(start,end,device):
    totalTime = end - start
    print(f"Train time is {totalTime} on the {device}")


def getLoaders(batchSize,
               numWorkers,
               pinMemory):
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    target_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.PILToTensor()
    ])

    fullDataset = OxfordIIITPet(
        root="dataset",
        split="trainval",
        target_types="segmentation",
        transform=transform,
        target_transform=target_transform,
        download=True
    )

    train_size = int(0.8 * len(fullDataset))
    val_size = len(fullDataset) - train_size
    trainDatas, validationDatas = random_split(fullDataset, [train_size, val_size])
    
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
        xTrain, yTrain = xTrain.to(device), yTrain.to(device).squeeze(1).long()
        yTrain = torch.clamp(yTrain, 0, 3-1)

        with torch.autocast(device_type="cuda", enabled=True):
            trainPred = model(xTrain)
            loss = lossFn(trainPred, yTrain)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loop.set_postfix(loss = loss.item())


def validationStep(model, dataLoader, device, numClasses=3, savePred=False):
    model.eval()
    valAcc, valDice, valIoU = 0.0, 0.0, 0.0

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(dataLoader):
            images, masks = images.to(device), masks.to(device).squeeze(1).long()
            masks = torch.clamp(masks, 0, 3-1)
            outputs = model(images)

            valAcc += pixelAccuracy(outputs, masks)
            valDice += meanDiceScore(outputs, masks, numClasses)
            valIoU += meanIoU(outputs, masks, numClasses)

    valAcc /= len(dataLoader)
    valDice /= len(dataLoader)
    valIoU /= len(dataLoader)

    print(f"Validation - Acc: {valAcc:.4f} | Dice: {valDice:.4f} | IoU: {valIoU:.4f}")
    return valDice


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
        predInds = torch.eq(predClasses,cls)
        
        #predInds ile aynı mantık
        targetInds = torch.eq(target,cls)

        intersection = (predInds & targetInds).sum().float() #yani sadece hem tahmin hem gerçek maskede aynı sınıfta olan pikseller

        union = (predInds | targetInds).sum().float() #yani tahmin veya gerçek maskede olan tüm pikseller

        if union == 0: #bu sınıf o görüntüde hiç yok
            iou = torch.tensor(0.1,device=DEVICE)
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
             dice = torch.tensor(1.0, device=DEVICE)
        else:
            dice = (2.0 * intersection) / total #Dice Score formülü
        
        dices.append(dice)
    #Tüm sınıfların Dice Score ortalamasını alıyoruz
    return torch.mean(torch.stack(dices)).item()

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
def savePredictions(model, dataLoader, device, saveDir="predictions", numSamples=4):
    model.eval()
    os.makedirs(saveDir, exist_ok=True)

    # 1. ÖZEL RENK HARİTASI TANIMLAMA (KESİN İSTENEN EŞLEŞME)
    # Maske Değeri: 0 (Pet)    -> Renk: Yeşil (#00FF00)
    # Maske Değeri: 1 (BG)     -> Renk: Kırmızı (#FF0000)
    # Maske Değeri: 2 (Border) -> Renk: Sarı (#FFFF00)
    colors = ['#00FF00', '#FF0000', '#FFFF00'] 
    cmap = mcolors.ListedColormap(colors)
    
    # Sınıfların sınırları
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    
    # Renk Skalası Etiketleri
    class_labels = ['Pet (0) - Yeşil', 'Background (1) - Kırmızı', 'Border (2) - Sarı']


    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(dataLoader):
            
            images = images.to(device)
            # Maske şekli (N, 1, H, W) -> (N, H, W) ve 0-indeksli yap
            masks = masks.to(device).squeeze(1).long()
            masks = torch.clamp(masks, 0, 3 - 1) 

            outputs = model(images)
            predClasses = torch.argmax(outputs, dim=1)

            current_saved = 0
            for i in range(images.size(0)):
                if current_saved >= numSamples:
                    break
                
                # 💥 DÜZELTİLMİŞ KISIM: Görüntü Normalizasyonunu Geri Alma
                image = images[i].cpu()
                # Önce STD ile çarp, sonra MEAN ekle (Doğru tersine çevirme)
                image = image * STD + MEAN 
                image_np = image.permute(1, 2, 0).numpy()
                image_np = np.clip(image_np, 0, 1) # [0, 1] aralığına sıkıştır

                gt_mask = masks[i].cpu().numpy()
                pred_mask = predClasses[i].cpu().numpy()

                # Görselleştirme
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                
                # 1. Orijinal Görüntü
                axes[0].imshow(image_np)
                axes[0].set_title("Orijinal Görüntü")
                axes[0].axis("off")
                
                # 2. Gerçek Maske (GT)
                im2 = axes[1].imshow(gt_mask, cmap=cmap, norm=norm)
                axes[1].set_title("Gerçek Maske (GT)")
                axes[1].axis("off")
                
                # Kontur ekle (Daha iyi görünürlük için Pet sınırına)
                contours_pet = measure.find_contours(gt_mask == 0, 0.5)
                for contour in contours_pet:
                    axes[1].plot(contour[:, 1], contour[:, 0], color='white', linewidth=1)


                # 3. Model Tahmini
                im3 = axes[2].imshow(pred_mask, cmap=cmap, norm=norm)
                axes[2].set_title("Model Tahmini")
                axes[2].axis("off")
                
                # Kontur ekle
                contours_pred = measure.find_contours(pred_mask == 0, 0.5)
                for contour in contours_pred:
                    axes[2].plot(contour[:, 1], contour[:, 0], color='white', linewidth=1)


                # Renk skalası ekleme
                cbar_ax = fig.add_axes([0.92, 0.15, 0.01, 0.7]) 
                cbar = fig.colorbar(im2, cax=cbar_ax, ticks=[0, 1, 2])
                cbar.ax.set_yticklabels(class_labels)
                cbar.ax.set_title("Sınıflar", fontsize=10)
                
                
                save_path = os.path.join(saveDir, f"sample_{batch_idx}_{i}.png")
                plt.savefig(save_path)
                plt.close(fig)
                
                current_saved += 1

            if current_saved >= numSamples:
                break
                
    print(f"[INFO] {current_saved} predictions saved to {saveDir}")