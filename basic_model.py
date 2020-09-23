from torch.nn import functional as F
from torchvision import transforms
from torchvision.utils import make_grid
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from dataset_configuration import *
from dataset_loader import *
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


class BasicModel(pl.LightningModule):

    def __init__(self, dataset_loader: DatasetLoader,
                 name='',
                 batch_size=100,
                 loss_func_type='CE',
                 version=0,
                 lr=0.01,
                 lr_step=10,
                 lr_gamma=0.1,
                 adversarial_methods=('i_fgsm', 'mi_fgsm', 'fgsm'),
                 adversarial_epsilon=0.1,
                 adversarial_mu=1.0):
        super().__init__()

        # init para
        arg_dict = inspect.getargvalues(inspect.currentframe()).locals
        del arg_dict['self']
        self.__dict__.update(arg_dict)

        # init dataset and name
        self.dataset_config = dataset_loader.dataset_config
        self.name = name if name else self.dataset_config.name

        # init loss function
        self.CE_loss = torch.nn.CrossEntropyLoss()
        self.KL_loss = torch.nn.KLDivLoss()

        # init logger
        self.logger = TensorBoardLogger(save_dir=self.dataset_config.log, name=self.name, version=self.version)
        self.correct_classification = {}
        self.mis_classification = {}
        for method in self.adversarial_methods:
            self.__dict__[method + '_images'] = {}
        self.train_log = pd.DataFrame()
        self.val_log = pd.DataFrame()
        self.test_log = pd.DataFrame()

        # init path
        version_name = 'version_' + str(self.version)
        self.log_path = self.log / self.name / version_name
        self.model_path = self.model / self.name / version_name
        self.cache_path = self.cache / self.name / version_name
        self.ckpt = self.model_path / (self.name + '.ckpt')
        for path in (self.log_path, self.model_path, self.cache_path):
            if not path.exists():
                path.mkdir(parents=True)

        # init model
        self.conv = None
        self.dense = None

    def __getattr__(self, item: str):
        if not item.startswith('_') and item in self.dataset_config.__dict__.keys():
            return self.dataset_config.__dict__[item]
        if not item.startswith('_') and item in self.dataset_loader.__dict__.keys():
            return self.dataset_loader.__dict__[item]
        return super().__getattr__(item)

    def clean_file(self):
        for path in (self.log_path, self.model_path, self.cache_path):
            for file in path.glob('*'):
                try:
                    file.unlink()
                except BaseException:
                    pass

    def prepare_data(self):
        self.dataset_loader.prepare_data()

    def train_dataloader(self):
        return DataLoader(self.dataset_loader.get_dataset('train'), batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dataset_loader.get_dataset('test'), batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.dataset_loader.get_dataset('test'), batch_size=1)

    def train_data_append(self, train_data: dict):
        self.dataset_loader.train_image = torch.cat([self.dataset_loader.train_image,
                                                     train_data['image'].type_as(self.dataset_loader.train_image)])
        self.dataset_loader.train_label = torch.cat([self.dataset_loader.train_label,
                                                     train_data['label'].type_as(self.dataset_loader.test_image)])

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.lr_step, gamma=self.lr_gamma)
        return [optimizer], [scheduler]

    def forward(self, x):
        if 'type_tensor' not in self.__dict__.keys():
            self.__dict__['type_tensor'] = torch.zeros_like(x)
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)
        x = self.dense(x)
        return x

    def loss_func(self, outputs, labels):
        if self.loss_func_type == 'CE':
            return self.CE_loss(outputs, labels)
        elif self.loss_func_type == 'KL':
            return self.KL_loss(torch.log_softmax(outputs, dim=-1), labels)

    def training_step(self, batch, batch_nb):
        images, labels = batch
        outputs = self(images)
        loss = self.loss_func(outputs, labels)
        logs = {'Train_Loss': float(loss)}
        self.train_log = self.train_log.append(logs, ignore_index=True)
        return {
            'loss': loss,
            'log': logs
        }

    @staticmethod
    def update_logs(logs, dic):
        if dic:
            assert isinstance(dic, dict)
            logs.update(dic)
        return logs

    @staticmethod
    def merge_dicts(*args):
        res = {}
        for dic in args:
            res.update(dic)
        return res

    def validation_step(self, batch, batch_nb):
        true_label = None
        if len(batch) == 2:
            images, labels = batch
        else:
            images, labels, true_label = batch
        size = len(batch[0])
        outputs = self(images)
        loss = self.loss_func(outputs, labels) * size
        predict = outputs.max(1)[1]
        logs = {'val_loss': loss}
        if len(batch) != 2:
            logs.update(self.judge_correct(predict, true_label))
        # update more
        logs = self.update_logs(logs, self.validation_eval(images, labels, size, outputs, predict))
        return logs

    @staticmethod
    def judge_correct(predict: torch.Tensor, labels: torch.Tensor):
        if labels.ndimension() == 1:
            correct = predict.eq(labels).sum().squeeze(0) \
                if labels.size()[0] == 1 else predict.eq(labels).sum()
            return {'correct': correct}
        else:
            same = predict.eq(labels.max(1)[1]).sum().squeeze(0) \
                if labels.size()[0] == 1 else predict.eq(labels.max(1)[1]).sum()
            return {'same': same}

    def validation_eval(self, images, labels, size, outputs, predict):
        return self.judge_correct(predict, labels)

    @property
    def epoch(self):
        return len(self.val_log.index)

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).sum() / len(self.val_dataloader().dataset)
        logs = {'Val_Loss': float(avg_loss)}
        # update more
        logs = self.update_logs(logs, self.validation_end_eval(outputs))
        self.val_log = self.val_log.append(logs, ignore_index=True)
        res_logs = {'log': logs}
        res_logs.update(logs)
        return res_logs

    def validation_end_eval(self, outputs):
        res = {}
        if 'correct' in outputs[0].keys():
            accuracy = torch.stack([x['correct'] for x in outputs]).sum() * 1.0 / len(self.val_dataloader().dataset)
            res['Accuracy'] = float(accuracy)
        if 'same' in outputs[0].keys():
            similarity = torch.stack([x['same'] for x in outputs]).sum() * 1.0 / len(self.val_dataloader().dataset)
            res['Similarity'] = float(similarity)
        return res

    def test_step(self, batch, batch_nb):
        true_label = None
        if len(batch) == 2:
            image, label = batch
        else:
            image, label, true_label = batch
        output = self(image)
        loss = self.loss_func(output, label)
        predict = output.max(1)[1]
        logs = {'test_loss': loss}
        if len(batch) != 2:
            logs.update(self.judge_correct(predict, true_label))
        # update more
        logs = self.update_logs(logs, self.test_eval(image, label, output, predict))

        true_label = true_label if len(batch) != 2 else label
        if int(logs['correct']):
            if int(true_label) not in self.correct_classification.keys():
                self.correct_classification[int(true_label)] = image.squeeze(0)
        else:
            if int(true_label) not in self.mis_classification.keys():
                self.mis_classification[int(true_label)] = image.squeeze(0)
        return logs

    def test_eval(self, image, label, output, predict):
        return self.merge_dicts(
            self.judge_correct(predict, label),
            self.adversarial_test(image, label, output, predict),
        )

    def test_epoch_end(self, outputs):
        # evaluate
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).sum() / len(self.test_dataloader().dataset)

        # collect images
        self.log_images(self.correct_classification, 'Correct_Images')
        self.log_images(self.mis_classification, 'Wrong_Images')
        for method in self.adversarial_methods:
            self.log_images(self.__dict__[method + '_images'], method.upper() + '_Adversarial_Images')

        # write logs
        logs = {'Test_Loss': round(float(avg_loss), 3)}
        # update more
        logs = self.update_logs(logs, self.test_end_eval(outputs))
        self.test_log = self.test_log.append(logs, ignore_index=True)
        res_logs = {'progress_bar': logs}
        res_logs.update(logs)
        # save
        self.train_log.to_csv(self.log_path / 'train_log.csv')
        self.val_log.to_csv(self.log_path / 'val_log.csv')
        self.test_log.to_csv(self.log_path / 'test_log.csv')
        return res_logs

    def test_end_eval(self, outputs):
        res = {}
        if 'correct' in outputs[0].keys():
            accuracy = torch.stack([x['correct'] for x in outputs]).sum() * 1.0 / len(self.val_dataloader().dataset)
            res['Test_Accuracy'] = round(float(accuracy), 3)
        if 'same' in outputs[0].keys():
            similarity = torch.stack([x['same'] for x in outputs]).sum() * 1.0 / len(self.val_dataloader().dataset)
            res['Test_Similarity'] = round(float(similarity), 3)
        return self.merge_dicts(
            res,
            self.adversarial_test_end(outputs)
        )

    def log_images(self, dic, name):
        if dic:
            images = torch.stack([dic[i] for i in sorted(dic)])
            grid = make_grid(images, nrow=10)
            self.logger.experiment.add_image(name, grid)

    def adversarial_attack(self, image, label, method):
        epsilon = self.adversarial_epsilon / self.channel
        mu = self.adversarial_mu
        if method == 'fgsm':
            with torch.enable_grad():
                image_grad = self.get_image_grad(image, label)
            perturbed_image = self.fgsm(image, image_grad, epsilon)
            return perturbed_image
        elif method == 'mi_fgsm':
            num_iters = int(100 * epsilon)
            alpha = epsilon / num_iters
            acc = torch.zeros_like(image)
            perturb = torch.zeros_like(image)
            for _ in range(num_iters):
                with torch.enable_grad():
                    image_grad = self.get_image_grad(image + perturb, label)
                    acc = mu * acc + image_grad / image_grad.abs()
                    perturb = perturb + alpha * acc.sign()
                    perturb = torch.clamp(perturb, -epsilon, epsilon)
            perturbed_image = torch.clamp(perturb + image, 0, 1)
            return perturbed_image
        elif method == 'i_fgsm':
            num_iters = int(100 * epsilon)
            alpha = epsilon / num_iters
            perturb = torch.zeros_like(image)
            for _ in range(num_iters):
                with torch.enable_grad():
                    image_grad = self.get_image_grad(image + perturb, label)
                    perturb = perturb + alpha * image_grad.sign()
                    perturb = torch.clamp(perturb, -epsilon, epsilon)
            perturbed_image = torch.clamp(perturb + image, 0, 1)
            return perturbed_image

    @staticmethod
    def fgsm(image, image_grad, epsilon):
        sign_image_grad = image_grad.sign()
        perturbed_image = image + epsilon * sign_image_grad
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        return perturbed_image

    def get_image_grad(self, image, label):
        image.requires_grad = True
        output = self(image)
        loss = self.CE_loss(output, label)
        self.zero_grad()
        loss.backward()
        image_grad = image.grad.data

        return image_grad

    def adversarial_test(self, image, label, output, predict):
        methods = self.adversarial_methods
        correct = predict.eq(label).squeeze(0)
        res_dic = {i.upper() + '_adversarial_' + j: torch.tensor([0]).type_as(label)
                   for i, j in product(methods, ('success', 'total'))}
        if int(correct):
            for method in methods:
                perturbed_image = self.adversarial_attack(image, label, method)
                perturbed_output = self(perturbed_image)
                perturbed_predict = perturbed_output.max(1)[1]
                res_dic[method.upper() + '_adversarial_success'] = perturbed_predict.ne(label).type_as(label)
                res_dic[method.upper() + '_adversarial_total'] = torch.tensor([1]).type_as(label)
                if int(perturbed_predict.ne(label).squeeze(0)):
                    if int(label) not in self.__dict__[method + '_images'].keys():
                        self.__dict__[method + '_images'][int(perturbed_predict)] = perturbed_image.squeeze(0)
        return res_dic

    @staticmethod
    def adversarial_test_end(outputs):
        methods = (i.rstrip('_adversarial_success') for i in outputs[0].keys() if i.endswith('_adversarial_success'))
        res_dic = {}
        for method in methods:
            success = torch.stack([x[method + '_adversarial_success'] for x in outputs]).sum() * 1.0
            total = torch.stack([x[method + '_adversarial_total'] for x in outputs]).sum()
            rate = round(float(success / total), 3)
            res_dic[method + '_Adversarial_Rate'] = rate
        return res_dic

