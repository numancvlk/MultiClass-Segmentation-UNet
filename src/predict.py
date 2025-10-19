#LIBRARIES
import torch

#SCRIPTS
from utils import getLoaders, savePredictions
from model import DEVICE, unetModel

BATCH_SIZE = 32
NUM_WORKERS = 2
PIN_MEMORY = True

if __name__ == "__main__":
    trainDataLoader, validationDataLoader = getLoaders(batchSize=BATCH_SIZE,
                                                    numWorkers=NUM_WORKERS,
                                                    pinMemory=PIN_MEMORY)

    model = unetModel.to(DEVICE)
    checkpoint = torch.load("myCheckpoint.pth", map_location=DEVICE)
    model.load_state_dict(checkpoint["model"])

    model.eval() 

    savePredictions(model=model,
                    dataLoader=validationDataLoader,
                    device=DEVICE,
                    numSamples=200)