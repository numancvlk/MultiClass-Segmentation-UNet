#LIBRARIES
import torch
from torch.nn import CrossEntropyLoss
from timeit import default_timer
from torch.optim.lr_scheduler import ReduceLROnPlateau

#SCRIPTS
from model import unetModel, DEVICE
from utils import loadCheckpoint, saveCheckpoint, printTrainTime, getLoaders,trainStep, validationStep, savePredictions


#HYPERPARAMETERS
EPOCHS = 100
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False

if __name__ == "__main__":
    torch.manual_seed(42)
    patience = 25
    patienceCounter = 0
    bestDice = 0.0

    trainDataLoader, validationDataLoader = getLoaders(batchSize=BATCH_SIZE,
                                                           numWorkers=NUM_WORKERS,
                                                           pinMemory=PIN_MEMORY)
        
    #MODEL / OPTIMIZER / LOSS FN / SCALER / PLATEU
    model = unetModel.to(DEVICE)
    optimizer = torch.optim.AdamW(params=model.parameters(),
                                    lr=LEARNING_RATE)
    lossFn = CrossEntropyLoss()
    scaler = torch.amp.GradScaler()
    scheduler = ReduceLROnPlateau(optimizer=optimizer,
                                  mode="max",
                                  factor=0.5,
                                  patience=10)

    if LOAD_MODEL == True:
        startEpoch = loadCheckpoint(checkpointFile="myCheckpoint.pth",
                    model=model,
                    optimizer=optimizer)
    else:
        startEpoch = 0

    trainStartTimer = default_timer()
    for epoch in range(startEpoch,EPOCHS):

        print(f"-----EPOCH = {epoch}-----")

        trainStep(model=model,
                  dataLoader=trainDataLoader,
                  optimizer=optimizer,
                  lossFn=lossFn,
                  scaler=scaler,
                  device=DEVICE)

        diceScore = validationStep(model=model,
                                   dataLoader=validationDataLoader,
                                   device=DEVICE,
                                   numClasses=3,
                                   savePred=True)
        
        scheduler.step(diceScore)

        for paramsGroup in optimizer.param_groups:
            currentLR = paramsGroup["lr"]
            print(f"CURRENT LR = {currentLR}")

        if diceScore > bestDice:
            bestDice = diceScore
            patienceCounter = 0
            saveCheckpoint(model=model,
                           optimizer=optimizer,
                           epoch=epoch,
                           checkpointFile="myCheckpoint.pth")
            savePredictions(model=model,
                            dataLoader=validationDataLoader,
                            device=DEVICE,
                            numSamples=30)
        else:
            patienceCounter += 1
            print(f"{patienceCounter} epoch'tur gelişme yok.")

            if patienceCounter == patience:
                print("EARLY STOPPING TRIGGERED")
                break
    
    endTrainTimer = default_timer()

    printTrainTime(start=trainStartTimer,
                   end=endTrainTimer,
                   device=DEVICE)