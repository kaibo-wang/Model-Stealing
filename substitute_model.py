from basic_model import *
from dataset_configuration import *
from dataset_loader import *
from victim_model import *


class SubstituteModel(BasicModel):

    def __init__(self, victim_model: BasicModel, init_dataset='', init_labeled_dataset='', lam=0.1, *args, **kwargs):

        dataset_config = victim_model.dataset_config
        init_dataset = init_dataset if init_dataset else 'fake_' + dataset_config.name
        if not init_labeled_dataset:
            dataset_loader = DatasetLoader(dataset_config, train_image_name=init_dataset)
        else:
            dataset_loader = DatasetLoader(dataset_config, dataset_name=init_dataset)
        name = kwargs['name'] if 'name' in kwargs.keys() else 'substitute_' + dataset_config.name

        super().__init__(dataset_loader, name=name, loss_func_type='KL', *args, **kwargs)
        self.lam = lam
        self.victim_model = victim_model
        if not init_labeled_dataset:
            self.dataset_loader.prepare_data()
            self.dataset_loader.train_label = self.label_data(self.dataset_loader.train_image)
            self.dataset_loader.test_label = (self.label_data(self.dataset_loader.test_image), self.dataset_loader.test_label)
        else:
            dataset = torch.load(self.dataset / (init_labeled_dataset + '.pt'))
            self.dataset_loader.__dict__.update(dataset)
            self.dataset_loader.__dict__['val_image'] = None
            self.dataset_loader.__dict__['val_label'] = None

    def label_data(self, data):
        data = data.detach()
        batch_size = 1000
        labels = None
        if data.size()[0] <= batch_size:
            return torch.softmax(self.victim_model(data), dim=-1)
        else:
            loader = DataLoader(TensorDataset(data), batch_size=batch_size)
            for i, [x] in enumerate(loader):
                label = torch.softmax(self.victim_model(x), dim=-1)
                labels = label if i == 0 else torch.cat([labels, label])
            return labels

    def adversarial_test(self, image, label, output, predict):
        methods = self.adversarial_methods
        label = label.max(1)[1]
        correct = predict.eq(label).squeeze(0)
        res_dic = {i.upper() + '_adversarial_' + j: torch.tensor([0]).type_as(label)
                   for i, j in product(methods, ('transfer_success', 'success', 'total'))}
        if int(correct):
            for method in methods:
                perturbed_image = self.adversarial_attack(image, label, method)
                perturbed_output = self(perturbed_image)
                perturbed_predict = perturbed_output.max(1)[1]
                res_dic[method.upper() + '_adversarial_success'] = perturbed_predict.ne(label).type_as(label)
                res_dic[method.upper() + '_adversarial_total'] = torch.tensor([1]).type_as(label)
                if int(perturbed_predict.ne(label).squeeze(0)):
                    perturbed_victim_output = self.victim_model(perturbed_image)
                    perturbed_victim_predict = perturbed_victim_output.max(1)[1]
                    res_dic[method.upper() + '_adversarial_transfer_success'] = \
                        perturbed_victim_predict.ne(label).type_as(label)
                    if int(perturbed_victim_predict.ne(label).squeeze(0)):
                        if int(label) not in self.__dict__[method + '_images'].keys():
                            self.__dict__[method + '_images'][int(perturbed_predict)] = perturbed_image.squeeze(0)
        return res_dic

    @staticmethod
    def adversarial_test_end(outputs):
        methods = (i.rstrip('_adversarial_success') for i in outputs[0].keys() if i.endswith('_adversarial_success'))
        res_dic = {}
        for method in methods:
            success = torch.stack([x[method + '_adversarial_success'] for x in outputs]).sum() * 1.0
            transfer = torch.stack([x[method + '_adversarial_transfer_success'] for x in outputs]).sum() * 1.0
            total = torch.stack([x[method + '_adversarial_total'] for x in outputs]).sum()
            rate = round(float(success / total), 3)
            transfer_rate = round(float(transfer / success), 3)
            victim_rate = round(float(transfer / total), 3)
            res_dic[method + '_Substitute_Adversarial_Rate'] = rate
            res_dic[method + '_Transfer_Rate'] = transfer_rate
            res_dic[method + '_Victim_Adversarial_Rate'] = victim_rate
        return res_dic

    def dataset_expansion(self, images, index=0):
        new_images = None
        for i, image in enumerate(images):
            new_image = self.create_new_image(image.unsqueeze(0))
            new_images = new_image if i == 0 else torch.cat([new_images, new_image])
        dic = {'image': new_images, 'label': self.label_data(new_images)}
        if index == 0:
            torch.save(dic, self.cache_path / 'cache.pt')
        else:
            torch.save(dic, self.cache_path / ('cache_' + str(index) + '.pt'))

    def create_new_image(self, image):
        with torch.enable_grad():
            image.requires_grad = True
            output = self(image)
            loss = torch.max(torch.softmax(output, dim=-1), dim=-1)[0]
            self.zero_grad()
            loss.backward()
            image_grad = image.grad.data
            image = torch.clamp(image + self.lam * image_grad.sign(), 0, 1)
        return image

    def train_dataloader(self):
        loader = DataLoader(self.dataset_loader.get_dataset('train'), batch_size=self.batch_size, shuffle=True)
        return loader


class DataExpansion(pl.Callback):

    def __init__(self, interval=5, steps=10):
        super().__init__()
        self.interval = interval
        self.total = interval * steps

    def on_epoch_end(self, trainer, model):
        if model.epoch and model.epoch != self.total and model.epoch % self.interval == 0:
            shuffle = torch.randperm(model.train_image.size()[0])
            data = model.train_image[shuffle[:1000]].type_as(model.type_tensor)
            index = model.epoch // self.interval
            model.dataset_expansion(data, index=index)
            model.train_data_append(torch.load(model.cache_path / ('cache_' + str(index) + '.pt')))


class MNISTSubstituteModel(SubstituteModel):

    def __init__(self, *args, **kwargs):
        ckpt_path = './data/Model/Mnist/mnist/version_0/mnist.ckpt'
        victim_model = MNISTVictimModel.load_from_checkpoint(ckpt_path)
        super().__init__(victim_model, *args, **kwargs)

        channels = [15, 20, 30]
        hid_nums = 100
        self.conv = torch.nn.Sequential(torch.nn.Conv2d(self.channel, channels[0], kernel_size=3, stride=1, padding=1),
                                        torch.nn.ReLU(),
                                        torch.nn.Conv2d(channels[0], channels[1], kernel_size=3, stride=1, padding=1),
                                        torch.nn.ReLU(),
                                        torch.nn.MaxPool2d(stride=2, kernel_size=2),  # //2
                                        torch.nn.Conv2d(channels[1], channels[2], kernel_size=3, stride=1, padding=1),
                                        torch.nn.ReLU(),
                                        torch.nn.MaxPool2d(stride=2, kernel_size=2))  # //2
        self.dense = torch.nn.Sequential(torch.nn.Linear(channels[-1] * self.height * self.width // 16, hid_nums),
                                         torch.nn.ReLU(),
                                         torch.nn.Dropout(p=0.5),
                                         torch.nn.Linear(hid_nums, self.num_classes))


class CIFAR10SubstituteModel(SubstituteModel):

    def __init__(self, *args, **kwargs):
        ckpt_path = './data/Model/Cifar10/cifar10/version_0/cifar10.ckpt'
        victim_model = CIFAR10VictimModel.load_from_checkpoint(ckpt_path)
        init_labeled_dataset = 'substitute_cifar10'
        super().__init__(victim_model, init_labeled_dataset=init_labeled_dataset, lr=0.01, lr_step=5, lr_gamma=0.7, *args, **kwargs)

        channels = [32, 64, 128, 256]
        hid_nums = 200

        self.conv = torch.nn.Sequential(torch.nn.Conv2d(self.channel, channels[0], kernel_size=3, stride=1, padding=1),
                                        torch.nn.ReLU(),
                                        torch.nn.Conv2d(channels[0], channels[1], kernel_size=3, stride=1, padding=1),
                                        torch.nn.ReLU(),
                                        torch.nn.MaxPool2d(stride=2, kernel_size=2),  # //2
                                        torch.nn.Conv2d(channels[1], channels[2], kernel_size=3, stride=1, padding=1),
                                        torch.nn.ReLU(),
                                        torch.nn.Conv2d(channels[2], channels[3], kernel_size=3, stride=1, padding=1),
                                        torch.nn.ReLU(),
                                        torch.nn.MaxPool2d(stride=2, kernel_size=2))  # //2
        self.dense = torch.nn.Sequential(torch.nn.Linear(channels[-1] * self.height * self.width // (4 ** 2), hid_nums),
                                         torch.nn.ReLU(),
                                         torch.nn.Dropout(p=0.4),
                                         torch.nn.Linear(hid_nums, self.num_classes))
