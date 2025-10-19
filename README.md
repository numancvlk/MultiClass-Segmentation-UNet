# MultiClass-Segmentation-UNet
# [TR]
## Proje Hakkında
Bu projeyi, 3 sınıflı (multi-class) semantik segmentasyon problemini çözmek üzere tasarladım. Eğitim sürecinde UNet mimarisini kullandım. Modeli veri seti üzerinde eğittim ve elde edilen sonuçlarla tahminler gerçekleştirdim.

## ⚡ Kullanılan Teknolojiler
- PyTorch
- Python
- UNet

## 🖼️ Veri Seti
- Veri seti olarak **OxfordIIITPet** veri setini kullandım.
- Veri seti [buradan](https://www.robots.ox.ac.uk/~vgg/data/pets/) indirilebilir.

## 📈 MODEL PERFORMANSI
Model, 3 segmentasyon sınıfı (Pet, Arka Plan ve Sınır) üzerinde genel olarak başarılı bir performans sergilemiştir.
- Eğitim Süresi = 1s 15dk
- Dice Score = 0.7180
- IoU Score = 0.5874
- Accuracy = %81.58

## 📊 MODEL SONUÇLARI
<div align="center">
  <img width="80%" height="452" alt="sample_184" src="https://github.com/user-attachments/assets/746548ef-e3a4-4894-90ad-1d81bb7201d8" />
</div>

<div align="center">
  <img width="80%" height="452" alt="sample_154" src="https://github.com/user-attachments/assets/2dcf7715-9436-4f3f-b530-39b69012594e" />
</div>

<div align="center">
  <img width="80%" height="452" alt="sample_141" src="https://github.com/user-attachments/assets/0046a167-5909-4879-8039-89037fe8a8bb" />
</div>

## Bu proje, sadece portföy amacıyla ve ticari bir amaç gütmeden paylaşılmaktadır.

# [EN]
## Project Overview
I designed this project to solve the 3-class (multi-class) semantic segmentation problem. I used the UNet architecture during the training process. I trained the model on the dataset and performed predictions with the obtained results.

## ⚡ Technologies Used
- PyTorch
- Python
- UNet

## 🖼️ Dataset
- I used the **OxfordIIITPet** dataset as the dataset.
- The dataset can be downloaded [here](https://www.robots.ox.ac.uk/~vgg/data/pets/).

## 📈 MODEL PERFORMANCE
The model demonstrated generally successful performance across the three segmentation classes (Pet, Background, and Border).

- Training Duration = 1h 15m
- Dice Score = 0.7180
- IoU Score = 0.5874
- Accuracy = 81.58%

## 📊 MODEL RESULTS
<div align="center">
  <img width="80%" height="452" alt="sample_184" src="https://github.com/user-attachments/assets/746548ef-e3a4-4894-90ad-1d81bb7201d8" />
</div>

<div align="center">
  <img width="80%" height="452" alt="sample_154" src="https://github.com/user-attachments/assets/2dcf7715-9436-4f3f-b530-39b69012594e" />
</div>

<div align="center">
  <img width="80%" height="452" alt="sample_141" src="https://github.com/user-attachments/assets/0046a167-5909-4879-8039-89037fe8a8bb" />
</div>

## This project is shared solely for portfolio purposes and without any commercial intent.
