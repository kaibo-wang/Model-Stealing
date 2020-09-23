from basic_model import *
from dataset_configuration import *
from dataset_loader import *
from victim_model import *
from substitute_model import *


if __name__ == '__main__':
    for i in range(1):
        interval = 5
        steps = 50
        data_expansion = DataExpansion(interval, steps)
        model = CIFAR10SubstituteModel(version=i)
        model.clean_file()

        trainer = Trainer(gpus=1,
                          max_epochs=interval * steps,
                          check_val_every_n_epoch=1,
                          logger=model.logger,
                          reload_dataloaders_every_epoch=True,
                          callbacks=[data_expansion])

        trainer.fit(model)
        trainer.test()
        trainer.save_checkpoint(model.ckpt)
