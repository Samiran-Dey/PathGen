import numpy as np
import torch
import pickle 
from utils.utils import *
import os
from collections import OrderedDict, Counter
from tqdm import tqdm
import pandas as pd
from argparse import Namespace
from sksurv.metrics import concordance_index_censored
from sklearn.metrics import cohen_kappa_score, roc_auc_score, roc_curve, auc


def train_loop_survival_coattn(args, cur, epoch, model, loader, optimizer, n_classes, writer, loss_fn_surv, loss_fn_grad, loss_alpha=0.5, gc=16):   
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    model.train()
    train_loss_surv, train_loss_grad, train_loss = 0., 0., 0.
    pred_grad, true_grad = [], []
    print('\n')

    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))
    
    for batch_idx, (slide_id, level, data_WSI, data_omic1, data_omic2, data_omic3, data_omic4, data_omic5, data_omic6, label_g, label_s, event_time, c) in tqdm(enumerate(loader)):

        data_WSI = data_WSI.to(device)
        data_omic1 = data_omic1.type(torch.FloatTensor).to(device)
        data_omic2 = data_omic2.type(torch.FloatTensor).to(device)
        data_omic3 = data_omic3.type(torch.FloatTensor).to(device)
        data_omic4 = data_omic4.type(torch.FloatTensor).to(device)
        data_omic5 = data_omic5.type(torch.FloatTensor).to(device)
        data_omic6 = data_omic6.type(torch.FloatTensor).to(device)
        label_g = label_g.type(torch.FloatTensor).to(device)
        label_s = label_s.type(torch.LongTensor).to(device)
        c = c.type(torch.FloatTensor).to(device)


        grad, surv, hazards, S, Y_hat, A  = model(x_path=data_WSI, x_omic1=data_omic1, x_omic2=data_omic2, x_omic3=data_omic3, x_omic4=data_omic4, x_omic5=data_omic5, x_omic6=data_omic6)
        
        grad_loss = loss_fn_grad(grad, label_g)
        surv_loss = loss_fn_surv(hazards=hazards, S=S, Y=label_s, c=c)

        loss = (loss_alpha * surv_loss + (1-loss_alpha) * grad_loss) / gc
        loss.backward()

        if (batch_idx + 1) % gc == 0: 
            optimizer.step()
            optimizer.zero_grad()

        risk = -torch.sum(S, dim=1).detach().cpu().numpy()
        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = c.item()
        all_event_times[batch_idx] = event_time

        pred_grad.append(int(torch.argmax(torch.softmax(grad, dim=-1), dim=-1)[0]))
        true_grad.append(int(torch.argmax(label_g, dim=-1)[0]))

        train_loss_surv += float(surv_loss)
        train_loss_grad += float(grad_loss)
        train_loss += float(loss)
                        

        if (batch_idx + 1) % 50 == 0:
            #save and print result
            message = f"{batch_idx}/{len(loader)} -- Train Loss Survival: {float(surv_loss)} -- Train Loss Gradation: {float(grad_loss)} -- Train Loss: {float(loss)}"
            #print(message)
            with open(writer, "a") as log_file:
                log_file.write('%s\n' % message)  # save the message
        

    # calculate loss and error for epoch
    train_loss_surv /= len(loader)
    train_loss_grad /= len(loader)
    train_loss /= len(loader)

    kappa=cohen_kappa_score(true_grad, pred_grad, weights='quadratic')

    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]

    if epoch % args.save_epoch == 0 or epoch == args.max_epochs-1:
        torch.save(model.state_dict(), args.results_dir+'/'+str(cur)+"_w"+str(epoch)+'.pt')

    #save and print result
    message = f"{epoch}/{args.max_epochs} -- Train Loss Survival: {train_loss_surv} -- Train Loss Gradation: {train_loss_grad} -- C Index: {c_index} -- Kappa: {kappa}"
    print(message)
    with open(writer, "a") as log_file:
        log_file.write('%s\n' % message)  # save the message



def validate_survival_coattn(args, cur, epoch, model, loader, n_classes, early_stopping=None, writer=None, loss_fn_surv=None, loss_fn_grad=None, results_dir=None):
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    val_loss_surv, val_loss_grad = 0., 0.
    pred_grad, true_grad, pred_prob, true_prob = [], [], [], []
    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))

    for batch_idx, (slide_id, level, data_WSI, data_omic1, data_omic2, data_omic3, data_omic4, data_omic5, data_omic6, label_g, label_s, event_time, c) in tqdm(enumerate(loader)):

        data_WSI = data_WSI.to(device)
        data_omic1 = data_omic1.type(torch.FloatTensor).to(device)
        data_omic2 = data_omic2.type(torch.FloatTensor).to(device)
        data_omic3 = data_omic3.type(torch.FloatTensor).to(device)
        data_omic4 = data_omic4.type(torch.FloatTensor).to(device)
        data_omic5 = data_omic5.type(torch.FloatTensor).to(device)
        data_omic6 = data_omic6.type(torch.FloatTensor).to(device)
        label_g = label_g.type(torch.FloatTensor).to(device)
        label_s = label_s.type(torch.LongTensor).to(device)
        c = c.type(torch.FloatTensor).to(device)

        with torch.no_grad():
            grad, surv, hazards, S, Y_hat, A = model(x_path=data_WSI, x_omic1=data_omic1, x_omic2=data_omic2, x_omic3=data_omic3, x_omic4=data_omic4, x_omic5=data_omic5, x_omic6=data_omic6) # return hazards, S, Y_hat, A_raw, results_dict

        grad_loss = loss_fn_grad(grad, label_g)
        surv_loss = loss_fn_surv(hazards=hazards, S=S, Y=label_s, c=c, alpha=0)

        risk = -torch.sum(S, dim=1).cpu().numpy()
        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = c.cpu().numpy()
        all_event_times[batch_idx] = event_time

        pred_grad.append(int(torch.argmax(torch.softmax(grad, dim=-1), dim=-1)[0]))
        true_grad.append(int(torch.argmax(label_g, dim=-1)[0]))
        pred_prob.append(torch.softmax(grad, dim=-1)[0].cpu().numpy())
        true_prob.append(label_g.cpu().numpy())

        val_loss_surv += float(surv_loss)
        val_loss_grad += float(grad_loss)


    val_loss_surv /= len(loader)
    val_loss_grad /= len(loader)
    kappa=cohen_kappa_score(true_grad, pred_grad, weights='quadratic')
    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    fpr, tpr, _ = roc_curve(np.array(true_prob).ravel(), np.array(pred_prob).ravel(), pos_label=1)
    roc_auc = float(auc(fpr, tpr)) 

    #save and print result
    message = f"{epoch}/{args.max_epochs} -- Validation Loss Survival: {val_loss_surv} -- Validation Loss Gradation: {val_loss_grad} -- C Index: {c_index} -- AUC: {roc_auc} -- Kappa: {kappa}"
    print(message)
    with open(writer, "a") as log_file:
        log_file.write('%s\n' % message) 

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss_surv, model, ckpt_name=os.path.join(results_dir, "s_{}_minloss_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True, c_index

    return False, c_index, kappa, roc_auc
