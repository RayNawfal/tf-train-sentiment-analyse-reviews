import training_dataset
from scramjet.streams import Stream

async def run(context, input):
    loss_acc = training_dataset.train(input)
    # Retrun the loss and accuracy of the trained model
    return Stream.read_from(loss_acc)
