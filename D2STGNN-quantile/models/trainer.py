import numpy as np
import torch
import torch.optim as optim
from torchinfo.torchinfo import summary
import pickle
from utils.train import data_reshaper, save_model
from .losses import masked_mae, masked_rmse, masked_mape, metric, quantile_loss


def masked_mape_np(y_true, y_pred, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            mask = np.not_equal(y_true, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mape = np.abs(np.divide(np.subtract(y_pred, y_true).astype('float32'), y_true))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape) * 100


class trainer():
    def __init__(self, scaler, model, **optim_args):
        self.model  = model         # init model
        self.scaler = scaler        # data scaler
        self.output_seq_len = optim_args['output_seq_len']  # output sequence length
        self.print_model_structure = optim_args['print_model']

        # training strategy parametes
        ## adam optimizer
        self.lrate  =  optim_args['lrate']
        self.wdecay = optim_args['wdecay']
        self.eps    = optim_args['eps']
        ## learning rate scheduler
        self.if_lr_scheduler    = optim_args['lr_schedule']
        self.lr_sche_steps      = optim_args['lr_sche_steps']
        self.lr_decay_ratio     = optim_args['lr_decay_ratio']
        ## curriculum learning
        self.if_cl          = optim_args['if_cl']
        self.cl_steps       = optim_args['cl_steps']
        self.cl_len = 0 if self.if_cl else self.output_seq_len
        ## warmup
        self.warm_steps     = optim_args['warm_steps']
        self.batch_size     = optim_args['batch_size']

        # Adam optimizer
        self.optimizer      = optim.Adam(self.model.parameters(), lr=self.lrate, weight_decay=self.wdecay, eps=self.eps)
        # learning rate scheduler
        self.lr_scheduler   = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.lr_sche_steps, gamma=self.lr_decay_ratio) if self.if_lr_scheduler else None

        # loss
        self.loss   = quantile_loss
        self.clip   = 5             # gradient clip
    
    def end_of_cl(self):
        return self.cl_len == self.output_seq_len

    def set_resume_lr_and_cl(self, epoch_num, batch_num):
        if batch_num == 0:
            return
        else:
            for _ in range(batch_num):
                # curriculum learning
                if _ < self.warm_steps:   # warmupping
                    self.cl_len = self.output_seq_len
                elif _ == self.warm_steps:
                    # init curriculum learning
                    self.cl_len = 1
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.lrate
                else:
                    # begin curriculum learning
                    if (_ - self.warm_steps) % self.cl_steps == 0 and self.cl_len < self.output_seq_len:
                        self.cl_len += int(self.if_cl)
            print("resume training from epoch{0}, where learn_rate={1} and curriculum learning length={2}".format(epoch_num, self.lrate, self.cl_len))

    def print_model(self, **kwargs):
        if self.print_model_structure and int(kwargs['batch_num'])==0:
            summary(self.model, input_data=input)
            parameter_num = 0
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    print(name, param.shape)
                tmp = 1
                for _ in param.shape:
                    tmp = tmp*_
                parameter_num += tmp
            print("Parameter size: {0}".format(parameter_num))

    def train(self, input, real_val, **kwargs):
        self.model.train()
        self.optimizer.zero_grad()

        self.print_model(**kwargs)

        output  = self.model(input)
        output  = output.transpose(1,2)

        # curriculum learning
        if kwargs['batch_num'] < self.warm_steps:   # warmupping
            self.cl_len = self.output_seq_len
        elif kwargs['batch_num'] == self.warm_steps:
            # init curriculum learning
            self.cl_len = 1
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.lrate
            print("======== Start curriculum learning... reset the learning rate to {0}. ========".format(self.lrate))
        else:
            # begin curriculum learning
            if (kwargs['batch_num'] - self.warm_steps) % self.cl_steps == 0 and self.cl_len <= self.output_seq_len:
                self.cl_len += int(self.if_cl)
        # scale data and calculate loss
        ## inverse transform for both predict and real value.
        predict     = self.scaler.inverse_transform(output)
        real_val    = self.scaler.inverse_transform(real_val[:,:,:,0])
        
        last_dim = predict.shape[2]
        predict = predict.view(self.batch_size, self.output_seq_len, 3, last_dim)
        loss    = self.loss(predict[:, :self.cl_len, :, :], real_val[:, :self.cl_len, :], 0)
        loss.backward()

        # gradient clip and optimization
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()

        # metrics
        predict = torch.unbind(predict,2)[1]
        mae = masked_mae(predict,real_val,0.0)
        mape = masked_mape(predict,real_val,0.0)
        rmse = masked_rmse(predict,real_val,0.0)
        return loss.item(), mae.item(), mape.item(), rmse.item()

    def eval(self, device, dataloader, model_name, **kwargs):
        # val a epoch
        valid_loss = []
        valid_mae  = []
        valid_mape = []
        valid_rmse = []
        self.model.eval()
        for itera, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            testx   = data_reshaper(x, device)
            testy   = data_reshaper(y, device)
            # for dstgnn
            output  = self.model(testx)
            output  = output.transpose(1,2)

            # scale data
            
            predict = self.scaler.inverse_transform(output)
            last_dim = predict.shape[2]
            predict = predict.view(self.batch_size, self.output_seq_len, 3, last_dim)
            real_val= self.scaler.inverse_transform(testy[:,:,:,0])

            # metrics
            loss = self.loss(predict, real_val, 0.0).item()
            predict = torch.unbind(predict,2)[1]
            mae = masked_mae(predict, real_val, 0.0).item()
            mape = masked_mape(predict,real_val,0.0).item()
            rmse = masked_rmse(predict,real_val,0.0).item()

            #print(f"test loss: {loss}, MAE: {mae}", end = '\r')

            valid_loss.append(loss)
            valid_mae.append(mae)
            valid_mape.append(mape)
            valid_rmse.append(rmse)

        mvalid_loss = np.mean(valid_loss)
        valid_mae   = np.mean(valid_mae)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)

        return mvalid_loss, valid_mae, mvalid_mape, mvalid_rmse

    @staticmethod
    def test(model, save_path_resume, device, dataloader, scaler, loss, save=True, **kwargs):
        # test
        model.eval()
        outputs = []
        realy   = torch.Tensor(dataloader['y_test']).to(device)
        realy   = realy.transpose(1, 2)
        y_list  = []
        for itera, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
            testx   = data_reshaper(x, device)
            testy   = data_reshaper(y, device).transpose(1, 2)

            with torch.no_grad():
                preds   = model(testx)
                preds  = preds.transpose(1,2)
                first_dim = preds.shape[0]
                last_dim = preds.shape[2]
                preds = preds.view(first_dim, 12, 3, last_dim)
                preds = torch.unbind(preds,2)[1]
                preds  = preds.transpose(1,2)

            outputs.append(preds)
            y_list.append(testy)
        yhat    = torch.cat(outputs,dim=0)[:realy.size(0),...]
        y_list  = torch.cat(y_list, dim=0)[:realy.size(0),...]

        assert torch.where(y_list == realy)

        realy   = scaler.inverse_transform(realy)[:, :, :, 0]
        yhat    = scaler.inverse_transform(yhat)

        # summarize the results.
        amae    = []
        amape   = []
        armse   = []

        for i in range(12):
            # For horizon i, only calculate the metrics **at that time** slice here.
            pred    = yhat[:,:,i]
            real    = realy[:,:,i]
            metrics = metric(pred,real)
            log     = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'
            print(log.format(i+1, metrics[0], metrics[2], metrics[1]))
            amae.append(metrics[0])     # mae
            amape.append(metrics[1])    # mape
            armse.append(metrics[2])    # rmse

        y12 = realy[:,:,11].cpu().detach().numpy()
        yhat12 = yhat[:,:,11].cpu().detach().numpy()

        y9 = realy[:,:,8].cpu().detach().numpy()
        yhat9 = yhat[:,:,8].cpu().detach().numpy()

        y6 = realy[:,:,5].cpu().detach().numpy()
        yhat6 = yhat[:,:,5].cpu().detach().numpy()

        y3 = realy[:,:,2].cpu().detach().numpy()
        yhat3 = yhat[:,:,2].cpu().detach().numpy()

        data = {'real12':y12, 'pred12': yhat12, 'real9':y9, 'pred9': yhat9, 'real6': y6, 'pred6': yhat6, 'real3': y3, 'pred3': yhat3}
        pickle.dump(data, open(f"result/D2STGNN-{loss}.pkl", "wb"))

        log = '(On average over 12 horizons) | Test MAE: {:.2f} | Test RMSE: {:.2f} | Test MAPE: {:.2f}% |'
        print(log.format(np.mean(amae), np.mean(armse),np.mean(amape) * 100))

        if save:
            save_model(model, save_path_resume)
