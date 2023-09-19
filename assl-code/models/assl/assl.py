import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.cuda.amp import autocast, GradScaler

import os
import contextlib
from train_utils import AverageMeter

from .assl_utils import consistency_loss, Get_Scalar
from train_utils import ce_loss


class ASSL:
    def __init__(self, net_builder, num_classes, ema_m, T, p_cutoff, lambda_u, \
                 hard_label=True, t_fn=None, p_fn=None, it=0, num_eval_iter=1000, tb_log=None, logger=None):

        super(ASSL, self).__init__()

        # momentum update param
        self.loader = {}
        self.num_classes = num_classes
        self.ema_m = ema_m

        # create the encoders
        # network is builded only by num_classes,
        # other configs are covered in main.py

        self.train_model = net_builder(num_classes=num_classes)
        self.eval_model = net_builder(num_classes=num_classes)
        self.num_eval_iter = num_eval_iter
        self.t_fn = Get_Scalar(T)  # temperature params function
        self.p_fn = Get_Scalar(p_cutoff)  # confidence cutoff function
        self.lambda_u = lambda_u
        self.tb_log = tb_log
        self.use_hard_label = hard_label

        self.optimizer = None
        self.scheduler = None

        self.best_eval_acc = 0.0

        self.logger = logger
        self.print_fn = print if logger is None else logger.info

        for param_q, param_k in zip(self.train_model.parameters(), self.eval_model.parameters()):
            param_k.data.copy_(param_q.detach().data)
            param_k.requires_grad = False

        self.eval_model.eval()

    @torch.no_grad()
    def _eval_model_update(self):
        """
        Momentum update of evaluation model (exponential moving average)
        """

        for param_train, param_eval in zip(self.train_model.parameters(), self.eval_model.parameters()):
            param_eval.copy_(param_eval * self.ema_m + param_train.detach() * (1 - self.ema_m))

        for buffer_train, buffer_eval in zip(self.train_model.buffers(), self.eval_model.buffers()):
            buffer_eval.copy_(buffer_train)

    def set_data_loader(self, loader_dict):
        self.loader_dict = loader_dict
        self.print_fn(f'[!] data loader keys: {self.loader_dict.keys()}')

    def set_optimizer(self, optimizer, scheduler=None):
        self.optimizer = optimizer
        self.scheduler = scheduler

    def first_train(self, args, logger=None):
        """
        Train function of ASSL.
        From data_loader, it inference training data, computes losses, and update the networks.
        """
        ngpus_per_node = torch.cuda.device_count()

        self.train_model.train()

        # for gpu profiling
        start_batch = torch.cuda.Event(enable_timing=True)
        end_batch = torch.cuda.Event(enable_timing=True)
        start_run = torch.cuda.Event(enable_timing=True)
        end_run = torch.cuda.Event(enable_timing=True)

        start_batch.record()

        self.it = 1
        best_it = 1

        scaler = GradScaler()
        amp_cm = autocast if args.amp else contextlib.suppress

        for x_lb, x_lb_s, x_lb_s2, x_lb_s3, x_lb_s4, x_lb_s5, x_lb_s6, x_lb_s7, x_lb_s8, x_lb_s9, x_lb_s10, y_lb in \
        self.loader_dict['train_lb']:  # loading x 、augment_x and label

            # prevent the training iterations exceed args.num_train_iter
            if self.it > args.every_num_train_iter:
                break

            end_batch.record()
            torch.cuda.synchronize()
            start_run.record()

            x_lb, x_lb_s, x_lb_s2, x_lb_s3, x_lb_s4, x_lb_s5, x_lb_s6, x_lb_s7, x_lb_s8, x_lb_s9, x_lb_s10 = x_lb.cuda(
                args.gpu), x_lb_s.cuda(args.gpu), x_lb_s2.cuda(args.gpu), x_lb_s3.cuda(args.gpu), x_lb_s4.cuda(
                args.gpu), x_lb_s5.cuda(args.gpu), x_lb_s6.cuda(args.gpu), x_lb_s7.cuda(args.gpu), x_lb_s8.cuda(
                args.gpu), x_lb_s9.cuda(args.gpu), x_lb_s10.cuda(args.gpu)

            y_lb = y_lb.cuda(args.gpu)

            inputs = torch.cat(
                (x_lb, x_lb_s, x_lb_s2, x_lb_s3, x_lb_s4, x_lb_s5, x_lb_s6, x_lb_s7, x_lb_s8, x_lb_s9, x_lb_s10))
            y_lb = torch.cat((y_lb, y_lb, y_lb, y_lb, y_lb, y_lb, y_lb, y_lb, y_lb, y_lb, y_lb))

            # inference and calculate sup/unsup losses
            with amp_cm():  #
                logits = self.train_model(inputs)

                # hyper-params for update
                T = self.t_fn(self.it)
                p_cutoff = self.p_fn(self.it)
                y_lb = y_lb.long()
                sup_loss = ce_loss(logits, y_lb, reduction='mean')

                total_loss = sup_loss

            # parameter updates
            if args.amp:
                scaler.scale(total_loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
            else:
                total_loss.backward()
                self.optimizer.step()

            self.scheduler.step()
            self.train_model.zero_grad()

            with torch.no_grad():
                self._eval_model_update()

            end_run.record()
            torch.cuda.synchronize()

            # tensorboard_dict update
            tb_dict = {}
            tb_dict['train/sup_loss'] = sup_loss.detach()
            tb_dict['train/total_loss'] = total_loss.detach()
            tb_dict['lr'] = self.optimizer.param_groups[0]['lr']
            tb_dict['train/prefecth_time'] = start_batch.elapsed_time(end_batch) / 1000.
            tb_dict['train/run_time'] = start_run.elapsed_time(end_run) / 1000.

            if self.it % self.num_eval_iter == 0:
                eval_dict = self.evaluate(args=args, cifar10_evl=False)
                tb_dict.update(eval_dict)

                save_path = os.path.join(args.save_dir, args.save_name)

                print(self.it, "：", "Best model accuracy", self.best_eval_acc, "Accuracy of the current eval_model on the validation set", tb_dict['eval/top-1-acc'])

                if tb_dict['eval/top-1-acc'] > self.best_eval_acc:
                    self.best_eval_acc = tb_dict['eval/top-1-acc']
                    best_it = self.it

                self.print_fn(f"{self.it} iteration, USE_EMA: {hasattr(self, 'eval_model')}, {tb_dict}, BEST_EVAL_ACC: {self.best_eval_acc}, at {best_it} iters")

            if not args.multiprocessing_distributed or \
                    (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):

                if self.it == best_it:
                    save_path = os.path.join(args.save_dir, args.save_name)
                    self.save_model('model_best.pth', save_path)

                if not self.tb_log is None:
                    self.tb_log.update(tb_dict, self.it)

            self.it += 1
            del tb_dict
            start_batch.record()

        save_path = os.path.join(args.save_dir, args.save_name)
        self.save_last_model('model_last_iter.pth', save_path)

        cifar10_evl = self.evaluate(args=args)
        print("Ask about the results of 1w cifar10 tests：", cifar10_evl)
        return cifar10_evl

    def train(self, args, logger=None):
        """
        Train function of ASSL.
        From data_loader, it inference training data, computes losses, and update the networks.
        """
        ngpus_per_node = torch.cuda.device_count()

        # loading model_last_iter
        self.train_model = self.load_last_iter_model(load_path='/saved_models/cifar10-40/model_last_iter.pth')
        self.train_model.train()

        # for gpu profiling
        start_batch = torch.cuda.Event(enable_timing=True)
        end_batch = torch.cuda.Event(enable_timing=True)
        start_run = torch.cuda.Event(enable_timing=True)
        end_run = torch.cuda.Event(enable_timing=True)

        start_batch.record()

        self.it = 1
        best_it = 1

        scaler = GradScaler()
        amp_cm = autocast if args.amp else contextlib.suppress

        for (x_lb, y_lb), (x_ulb_w, x_ulb_s, _) in zip(self.loader_dict['train_lb'], self.loader_dict['train_ulb']):

            # prevent the training iterations exceed args.num_train_iter
            if self.it > args.every_num_train_iter:
                break

            end_batch.record()
            torch.cuda.synchronize()
            start_run.record()

            num_lb = x_lb.shape[0]
            num_ulb = x_ulb_w.shape[0]
            assert num_ulb == x_ulb_s.shape[0]

            x_lb, x_ulb_w, x_ulb_s = x_lb.cuda(args.gpu), x_ulb_w.cuda(args.gpu), x_ulb_s.cuda(args.gpu)
            y_lb = y_lb.cuda(args.gpu)

            inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))

            # inference and calculate sup/unsup losses
            with amp_cm():
                logits = self.train_model(inputs)
                logits_x_lb = logits[:num_lb]
                logits_x_ulb_w, logits_x_ulb_s = logits[num_lb:].chunk(2)
                del logits
                T = self.t_fn(self.it)
                p_cutoff = self.p_fn(self.it)
                y_lb = y_lb.long()
                sup_loss = ce_loss(logits_x_lb, y_lb, reduction='mean')
                unsup_loss, mask = consistency_loss(logits_x_ulb_w,
                                                    logits_x_ulb_s,
                                                    'ce', T, p_cutoff,
                                                    use_hard_labels=args.hard_label)

                total_loss = sup_loss + self.lambda_u * unsup_loss

            if args.amp:
                scaler.scale(total_loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
            else:
                total_loss.backward()
                self.optimizer.step()

            self.scheduler.step()
            self.train_model.zero_grad()

            with torch.no_grad():
                self._eval_model_update()

            end_run.record()
            torch.cuda.synchronize()

            # tensorboard_dict update
            tb_dict = {}
            tb_dict['train/sup_loss'] = sup_loss.detach()
            tb_dict['train/unsup_loss'] = unsup_loss.detach()
            tb_dict['train/total_loss'] = total_loss.detach()
            tb_dict['train/mask_ratio'] = 1.0 - mask.detach()
            tb_dict['lr'] = self.optimizer.param_groups[0]['lr']
            tb_dict['train/prefecth_time'] = start_batch.elapsed_time(end_batch) / 1000.
            tb_dict['train/run_time'] = start_run.elapsed_time(end_run) / 1000.

            if self.it % self.num_eval_iter == 0:
                eval_dict = self.evaluate(args=args, cifar10_evl=False)
                tb_dict.update(eval_dict)

                save_path = os.path.join(args.save_dir, args.save_name)

                print(self.it, "：", "Best model accuracy", self.best_eval_acc, "Accuracy of the current eval_model on the validation set", tb_dict['eval/top-1-acc'],
                      "Accuracy of the current train_model on the validation set", tb_dict['train/top-1-acc'])

                if tb_dict['eval/top-1-acc'] > self.best_eval_acc:
                    self.best_eval_acc = tb_dict['eval/top-1-acc']
                    best_it = self.it

                self.print_fn(
                    f"{self.it} iteration, USE_EMA: {hasattr(self, 'eval_model')}, {tb_dict}, BEST_EVAL_ACC: {self.best_eval_acc}, at {best_it} iters")

            if not args.multiprocessing_distributed or \
                    (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):

                if self.it == best_it:
                    save_path = os.path.join(args.save_dir, args.save_name)
                    self.save_model('model_best.pth', save_path)

                if not self.tb_log is None:
                    self.tb_log.update(tb_dict, self.it)

            self.it += 1
            del tb_dict
            start_batch.record()

        save_path = os.path.join(args.save_dir, args.save_name)
        self.save_last_model('model_last_iter.pth', save_path)

        cifar10_evl = self.evaluate(args=args)
        print("Ask about the results of 1w cifar10 tests：", cifar10_evl)
        return cifar10_evl

    @torch.no_grad()
    def evaluate(self, eval_loader=None, args=None, cifar10_evl=True):
        use_ema = hasattr(self, 'eval_model')

        eval_model = self.eval_model if use_ema else self.train_model
        eval_model.eval()
        if cifar10_evl:
            if eval_loader is None:
                eval_loader = self.loader_dict['eval_cifar10']
        else:
            if eval_loader is None:
                eval_loader = self.loader_dict['eval']

        total_loss = 0.0
        total_acc = 0.0
        total_num = 0.0
        total_acc_train = 0.0
        total_loss_train = 0.0
        for x, y in eval_loader:
            x, y = x.cuda(args.gpu), y.cuda(args.gpu)
            num_batch = x.shape[0]
            total_num += num_batch
            logits = eval_model(x)
            logits_train = self.train_model(x)
            loss = F.cross_entropy(logits, y.long(), reduction='mean')
            loss_train = F.cross_entropy(logits_train, y.long(), reduction='mean')
            acc = torch.sum(torch.max(logits, dim=-1)[1] == y)
            acc_train = torch.sum(torch.max(logits_train, dim=-1)[1] == y)

            total_loss += loss.detach() * num_batch
            total_acc += acc.detach()
            total_loss_train += loss_train.detach() * num_batch
            total_acc_train += acc_train.detach()

        if not use_ema:
            eval_model.train()

        return {'eval/top-1-acc': total_acc / total_num, 'train/top-1-acc': total_acc_train / total_num,
                'eval/loss': total_loss / total_num, 'train/loss': total_loss_train / total_num}

    def save_model(self, save_name, save_path):
        save_filename = os.path.join(save_path, save_name)
        train_model = self.train_model.module if hasattr(self.train_model, 'module') else self.train_model
        eval_model = self.eval_model.module if hasattr(self.eval_model, 'module') else self.eval_model
        torch.save({'train_model': train_model.state_dict(),
                    'eval_model': eval_model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                    'it': self.it}, save_filename)

        self.print_fn(f"model saved: {save_filename}")

    def save_last_model(self, save_name, save_path):
        save_filename = os.path.join(save_path, save_name)
        torch.save({'train_model': self.train_model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                    'it': self.it}, save_filename)

        torch.save(self.train_model, './saved_models/cifar10-40/model_last_iter_only_model.pth')

        self.print_fn(f"last_model saved: {save_filename}")

    def load_model(self, load_path):
        checkpoint = torch.load(load_path)

        train_model = self.train_model.module if hasattr(self.train_model, 'module') else self.train_model
        eval_model = self.eval_model.module if hasattr(self.eval_model, 'module') else self.eval_model

        for key in checkpoint.keys():
            if hasattr(self, key) and getattr(self, key) is not None:
                if 'train_model' in key:
                    train_model.load_state_dict(checkpoint[key])
                elif 'eval_model' in key:
                    eval_model.load_state_dict(checkpoint[key])
                elif key == 'it':
                    self.it = checkpoint[key]
                elif key == 'scheduler':
                    self.scheduler.load_state_dict(checkpoint[key])
                elif key == 'optimizer':
                    self.optimizer.load_state_dict(checkpoint[key])
                else:
                    getattr(self, key).load_state_dict(checkpoint[key])
                self.print_fn(f"Check Point Loading: {key} is LOADED")
            else:
                self.print_fn(f"Check Point Loading: {key} is **NOT** LOADED")

    def load_last_iter_model(self, load_path):
        checkpoint = torch.load(load_path)

        train_model = self.train_model.module if hasattr(self.train_model, 'module') else self.train_model

        for key in checkpoint.keys():
            if hasattr(self, key) and getattr(self, key) is not None:
                if 'train_model' in key:
                    train_model.load_state_dict(checkpoint[key])
                elif key == 'it':
                    self.it = checkpoint[key]
                elif key == 'scheduler':
                    self.scheduler.load_state_dict(checkpoint[key])
                elif key == 'optimizer':
                    self.optimizer.load_state_dict(checkpoint[key])
                else:
                    getattr(self, key).load_state_dict(checkpoint[key])
                self.print_fn(f"Check Point Loading: {key} is LOADED")
            else:
                self.print_fn(f"Check Point Loading: {key} is **NOT** LOADED")
        return train_model


if __name__ == "__main__":
    pass
