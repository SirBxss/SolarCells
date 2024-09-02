import numpy as np
import torch as t
from sklearn.metrics import f1_score
from tqdm.autonotebook import tqdm

class Trainer:
    def __init__(self, model, crit, optim=None, train_dl=None, val_test_dl=None, cuda=True, early_stopping_patience=-1, scheduler=None):
        self._model = model
        self._crit = crit
        self._optim = optim
        self._scheduler = scheduler
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._cuda = cuda
        self._early_stopping_patience = early_stopping_patience

        if cuda:
            self._model = model.cuda()
            self._crit = crit.cuda()

    def save_checkpoint(self, epoch):
        t.save({'state_dict': self._model.state_dict()}, 'checkpoints/checkpoint.ckp')
        self.save_onnx('checkpoints/model.onnx')

    def restore_checkpoint(self, epoch_n, path='checkpoints'):
        ckp = t.load(f'{path}/checkpoint.ckp', 'cuda' if self._cuda else None)
        self._model.load_state_dict(ckp['state_dict'])

    def save_onnx(self, fn):
        m = self._model.cpu()
        m.eval()
        x = t.randn(1, 3, 300, 300, requires_grad=True)
        t.onnx.export(m, x, fn, export_params=True, opset_version=10, do_constant_folding=True,
                      input_names=['input'], output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
        if self._cuda:
            m = self._model.cuda()

    def train_step(self, x, y):
        self._optim.zero_grad()
        out = self._model(x)
        loss = self._crit(out, y.float())
        loss.backward()
        self._optim.step()
        if self._scheduler is not None:
            self._scheduler.step()
        return loss.item()

    def val_test_step(self, x, y):
        out = self._model(x)
        loss = self._crit(out, y.float())
        out = out.detach().cpu().numpy()
        pred_0 = (out[:, 0] > 0.5).astype(int)
        pred_1 = (out[:, 1] > 0.5).astype(int)
        pred = np.stack([pred_0, pred_1], axis=1)
        return loss.item(), pred

    def train_epoch(self):
        self._model.train()
        avg_loss = 0
        for x, y in self._train_dl:
            if self._cuda:
                x = x.cuda()
                y = y.cuda()
            loss = self.train_step(x, y)
            avg_loss += loss / len(self._train_dl)
        return avg_loss

    def val_test(self):
        self._model.eval()
        with t.no_grad():
            avg_loss = 0
            preds = []
            labels = []
            for x, y in self._val_test_dl:
                if self._cuda:
                    x = x.cuda()
                    y = y.cuda()
                loss, pred = self.val_test_step(x, y)
                avg_loss += loss / len(self._val_test_dl)
                if self._cuda:
                    y = y.cpu()
                preds.extend(pred)
                labels.extend(y.numpy())
            preds, labels = np.array(preds), np.array(labels)
            score = f1_score(labels, preds, average='micro')
        return avg_loss, score

    def fit(self, epochs=-1):
        assert self._early_stopping_patience > 0 or epochs > 0
        train_losses = []
        val_losses = []
        val_metrics = []
        epoch_n = 0

        while True:
            if epoch_n == epochs:
                break
            print(f'Epoch: {epoch_n + 1}')
            train_loss = self.train_epoch()
            val_loss, val_metric = self.val_test()

            if len(val_losses) != 0 and val_loss < min(val_losses):
                self.save_checkpoint(epoch_n)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_metrics.append(val_metric)

            if self._early_stopping_patience > 0:
                if len(val_losses) > self._early_stopping_patience:
                    if val_losses[-1] > val_losses[-self._early_stopping_patience - 1]:
                        break
            epoch_n += 1
            print(f'\tTrain Loss: {train_loss:.4f}\tVal Loss: {val_loss:.4f}\tVal Metric: {val_metric:.4f}')
        return train_losses, val_losses, val_metrics
