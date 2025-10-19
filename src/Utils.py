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
from torch.utils.data import Subset
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
        # yTrain = torch.clamp(yTrain, 0, 3-1)
        yTrain = yTrain - 1
        yTrain = torch.clamp(yTrain, 0, 2)

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
            # masks = torch.clamp(masks, 0, 3-1)
            masks = masks - 1
            masks = torch.clamp(masks, 0, 2)
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
            continue
        else: 
            iou = intersection / union #normal IoU formülü
            ious.append(iou)
        #Bu sınıfın IoU değerini listeye ekle
        
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
             continue
        else:
            dice = (2.0 * intersection) / total #Dice Score formülü
            dices.append(dice)
        
    #Tüm sınıfların Dice Score ortalamasını alıyoruz
    return torch.mean(torch.stack(dices)).item()

def savePredictions(model, dataLoader, device, saveDir="predictions", numSamples=4):
    model.eval()
    os.makedirs(saveDir, exist_ok=True)

    class_colors = {
        0: [0, 0, 1], # Mavi: Pet
        1: [0, 1, 0], # Yeşil: Background
        2: [1, 0, 0]  # Kırmızı: Border
    } 
    
    # YENİ: Toplam kaç resim kaydedildiğini takip eden sayaç
    saved_count = 0 

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(dataLoader):
            images = images.to(device)
            masks = masks.to(device).squeeze(1).long()
            
            # Etiket kaydırma işlemi
            masks = masks - 1
            masks = torch.clamp(masks, 0, 2) 

            outputs = model(images)
            predClasses = torch.argmax(outputs, dim=1)

            # Batç içindeki her bir resmi kontrol et
            for i in range(images.size(0)):
                
                # Toplam numSamples limitine ulaşıldıysa, dış döngüyü durdur
                if saved_count >= numSamples:
                    break # İç döngüyü kır
                
                # --- KAYDETME KODU BAŞLANGICI (Figür oluşturma, çizme, kaydetme) ---
                
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))

                # 1) Orijinal Resim (Aynı)
                img = images[i].cpu().permute(1,2,0).numpy()
                img = (img - img.min()) / (img.max() - img.min()) 
                axes[0].imshow(img)
                axes[0].set_title("Original Image")
                axes[0].axis("off")

                # 2) Ground Truth (Aynı)
                gt_mask = masks[i].cpu().numpy().squeeze()
                gt_color = np.zeros((*gt_mask.shape, 3))
                for cls, color in class_colors.items():
                    gt_color[gt_mask == cls] = color
                axes[1].imshow(gt_color)
                axes[1].set_title("Ground Truth")
                axes[1].axis("off")
                for cls in [0, 1, 2]:
                    contours = measure.find_contours(gt_mask == cls, 0.5) 
                    for contour in contours:
                        axes[1].plot(contour[:, 1], contour[:, 0], color='white', linewidth=1)

                # 3) Prediction (Aynı)
                pred_mask = predClasses[i].cpu().numpy().squeeze()
                pred_color = np.zeros((*pred_mask.shape, 3))
                for cls, color in class_colors.items():
                    pred_color[pred_mask == cls] = color
                axes[2].imshow(pred_color)
                axes[2].set_title("Prediction")
                axes[2].axis("off")
                for cls in [0, 1, 2]:
                    contours = measure.find_contours(pred_mask == cls, 0.5)
                    for contour in contours:
                        axes[2].plot(contour[:, 1], contour[:, 0], color='white', linewidth=1)

                save_path = os.path.join(saveDir, f"sample_{saved_count}.png") # İsimlendirmeyi sayaçla yap
                plt.savefig(save_path, bbox_inches='tight')
                plt.close(fig)

                # YENİ: Başarıyla kaydedildi, sayacı artır
                saved_count += 1 
            
            # İç döngü bittiğinde, toplam limite ulaşıldıysa ana döngüyü de kır
            if saved_count >= numSamples:
                break 

    print(f"[INFO] {saved_count} predictions saved to {saveDir}")