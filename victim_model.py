from basic_model import *
from dataset_configuration import *
from dataset_loader import *


class MNISTVictimModel(BasicModel):

    def __init__(self, *args, **kwargs):

        # init dataset
        data_config = DataSetConfiguration('mnist')
        data_loader = DatasetLoader(data_config)
        super().__init__(data_loader, *args, **kwargs)

        # rewrite layers
        channels = [10, 20]
        hid_nums = 320
        self.conv = torch.nn.Sequential(torch.nn.Conv2d(self.channel, channels[0], kernel_size=3, stride=1, padding=1),
                                        torch.nn.ReLU(),
                                        torch.nn.Conv2d(channels[0], channels[1], kernel_size=3, stride=1, padding=1),
                                        torch.nn.ReLU(),
                                        torch.nn.MaxPool2d(stride=2, kernel_size=2))  # //2
        self.dense = torch.nn.Sequential(torch.nn.Linear(channels[-1] * self.height * self.width // 4, hid_nums),
                                         torch.nn.ReLU(),
                                         torch.nn.Dropout(p=0.5),
                                         torch.nn.Linear(hid_nums, self.num_classes))


class CIFAR10VictimModel(BasicModel):

    def __init__(self, *args, **kwargs):

        # init dataset
        data_config = DataSetConfiguration('cifar10', 3, 32, 32)
        data_loader = DatasetLoader(data_config)
        super().__init__(data_loader, lr=0.002, lr_step=5, lr_gamma=0.4, *args, **kwargs)

        # rewrite layers
        channels = [16, 32, 64, 128, 256]
        hid_nums = 150
        self.conv = torch.nn.Sequential(torch.nn.Conv2d(self.channel, channels[0], kernel_size=3, stride=1, padding=1),
                                        torch.nn.ReLU(),
                                        torch.nn.Conv2d(channels[0], channels[1], kernel_size=3, stride=1, padding=1),
                                        torch.nn.ReLU(),
                                        torch.nn.MaxPool2d(stride=2, kernel_size=2),  # //2
                                        torch.nn.Conv2d(channels[1], channels[2], kernel_size=3, stride=1, padding=1),
                                        torch.nn.ReLU(),
                                        torch.nn.Conv2d(channels[2], channels[3], kernel_size=3, stride=1, padding=1),
                                        torch.nn.ReLU(),
                                        torch.nn.MaxPool2d(stride=2, kernel_size=2),  # //2
                                        torch.nn.Conv2d(channels[3], channels[4], kernel_size=3, stride=1, padding=1),
                                        torch.nn.ReLU(),
                                        torch.nn.MaxPool2d(stride=2, kernel_size=2),  # //2
                                        )
        self.dense = torch.nn.Sequential(torch.nn.Linear(channels[-1] * self.height * self.width // (8 ** 2), hid_nums),
                                         torch.nn.ReLU(),
                                         torch.nn.Dropout(p=0.4),
                                         torch.nn.Linear(hid_nums, self.num_classes))


if __name__ == '__main__':
    model = CIFAR10VictimModel()
    model.clean_file()
    trainer = Trainer(gpus=1,
                      min_epochs=50,
                      max_epochs=100,
                      check_val_every_n_epoch=1,
                      early_stop_callback=pl.callbacks.early_stopping.EarlyStopping(monitor='Accuracy', patience=3),
                      logger=model.logger)
    trainer.fit(model)
    trainer.test()
    trainer.save_checkpoint(model.ckpt)
