import numpy as np
import torch
import pickle 
from utils.utils import *
import os
from collections import OrderedDict
from tqdm import tqdm
import pandas as pd
from collections import Counter

from argparse import Namespace
from lifelines.utils import concordance_index
from sksurv.metrics import concordance_index_censored
from sksurv.metrics import integrated_brier_score
from sklearn.metrics import cohen_kappa_score, roc_auc_score, balanced_accuracy_score, f1_score, roc_curve, auc, confusion_matrix 
import matplotlib.pyplot as plt
import seaborn as sns

def get_risk(hazards):
    S = torch.cumprod(1 - hazards, dim=1)
    risk = -torch.sum(S, dim=1).cpu().numpy()
    return risk[0]

def calibrate(args, model, loader, n_classes, writer, alpha=0.1):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model.eval()

    n = len (loader)

    if n == 0:
        return 1.0
    
    
    # 1: get conformal scores

    cal_grad, cal_surv, cal_risk = [], [], []

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
            
            
            grad, surv, hazards, S, Y_hat, A = model(
                x_path=data_WSI, 
                x_omic1=data_omic1, 
                x_omic2=data_omic2, 
                x_omic3=data_omic3, 
                x_omic4=data_omic4, 
                x_omic5=data_omic5, 
                x_omic6=data_omic6
            )

            s_grad = torch.softmax(grad, dim=-1)[0].detach().cpu().numpy()
            s_surv = torch.softmax(surv, dim=-1).detach().cpu()
        
        
        label_surv = torch.zeros_like(s_surv)
        if c == 1: #if alive
            label_surv[0][label_s-1]=1
        else:
            label_surv[0][label_s-2]=1
        s_surv = s_surv[0].numpy()
        risk = get_risk(hazards)
        gt_risk = get_risk(label_surv)-1

        cal_grad.append(1-s_grad[int(torch.argmax(label_g, dim=-1))])
        cal_surv.append(1-s_surv[int(torch.argmax(label_surv, dim=-1))]) #calbrate through bin
        cal_risk.append(abs(abs(gt_risk)-abs(risk))) #calibrate using risk

    # 2: get quantile 
    q_level = np.ceil((n+1)*(1-alpha))/n
    if q_level>1:
        q_level=1
    if q_level<0:
        q_level=0

    # 3: compute qhat
    qhat_grad = np.quantile(cal_grad, q_level, method='higher')
    qhat_surv = np.quantile(cal_surv, q_level, method='higher')
    qhat_risk = np.quantile(cal_risk, q_level, method='higher')

    #display
    msg = 'q_level = %.5f \nqhat_grad = %.5f \nqhat_surv = %.5f \nqhat_risk = %.5f \n'%(q_level,qhat_grad,qhat_surv,qhat_risk)
    print(msg)
    with open(writer, "a") as log_file:
        log_file.write('%s\n' % msg)  # save the message

    return qhat_grad, qhat_surv, qhat_risk


def test_calibrate(args, model, loader, n_classes, writer, category='overall'):
    
    msg = '\nResult %s ->'%category
    print(msg)
    with open(writer, "a") as log_file:
        log_file.write('%s\n' % msg)  # save the message
    
    if len(loader[1]) == 0:
        return 
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model.eval()

    qhat_grad, qhat_surv, qhat_risk = calibrate(args, model, loader[0], n_classes, writer)
    
    ac_grad, ac_surv, ac_risk = 0, 0, 0
    cal_grad, un_grad, prd_grad, w_grad = [], [], [], []
    cal_surv, un_surv, prd_surv, w_surv = [], [], [], []
    lb_risk, ub_risk, cal_risk, un_risk, prd_risk, w_risk = [], [], [], [], [], []
    sid = []
    
    all_risk_scores = np.zeros((len(loader[1])))
    all_censorships = np.zeros((len(loader[1])))
    all_event_times = np.zeros((len(loader[1])))

    results = []

    for batch_idx, (slide_id, level, data_WSI, data_omic1, data_omic2, data_omic3, data_omic4, data_omic5, data_omic6, label_g, label_s, event_time, c) in tqdm(enumerate(loader[1])):
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
            

            grad, surv, hazards, S, Y_hat, A = model(
                x_path=data_WSI, 
                x_omic1=data_omic1, 
                x_omic2=data_omic2, 
                x_omic3=data_omic3, 
                x_omic4=data_omic4, 
                x_omic5=data_omic5, 
                x_omic6=data_omic6
            )

            risk = get_risk(hazards)
            
        

        s_grad = torch.softmax(grad, dim=-1)[0].detach().cpu().numpy()
        s_surv = torch.softmax(surv, dim=-1)

        label_surv = torch.zeros_like(s_surv)
        if c == 1: #if alive
            label_surv[0][label_s-1]=1
        else:
            label_surv[0][label_s-2]=1  

        
        gt_risk = get_risk(label_surv)-1
        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = c.cpu().numpy()
        all_event_times[batch_idx] = event_time
        ub_risk.append(-((-risk)-qhat_risk)) 
        if ub_risk[-1]>-1:
            ub_risk[-1]=-1
        lb_risk.append(-((-risk)+qhat_risk))
        if lb_risk[-1]<-5:
            lb_risk[-1]=-5
        c_risk = np.arange(np.floor(-ub_risk[-1]), np.ceil(-lb_risk[-1]))
        cal_risk.append(c_risk)

        s_surv = s_surv[0].detach().cpu().numpy()
        p_grad = int(torch.argmax(torch.softmax(grad, dim=-1), dim=-1)[0])

        #calibrate by thresholding using computed qhat
        c_grad = np.arange(args.n_grade)[(s_grad >= (1-qhat_grad))]
        if len(c_grad) == 0:
            c_grad = np.arange(args.n_grade)
        c_surv = np.arange(args.n_timebin)[(s_surv >= (1-qhat_surv))]
        if len(c_surv) == 0:
            c_surv = np.arange(args.n_timebin)
        
        cal_grad.append(c_grad)
        cal_surv.append(c_surv)
        #compute uncertainty
        un_grad.append((len(c_grad)/s_grad.shape[-1])*((max(c_grad)-min(c_grad))/(s_grad.shape[-1]-1)))
        un_surv.append((len(c_surv)/args.n_timebin)*((max(c_surv)-min(c_surv))/(args.n_timebin)))
        un_risk.append((len(c_risk)/args.n_timebin)*(ub_risk[-1]-lb_risk[-1])/(-1 - -(args.n_timebin+1)))

        #test coverage
        if int(torch.argmax(label_g, dim=-1)[0]) in c_grad:
            ac_grad+=1
            prd_grad.append(int(torch.argmax(label_g, dim=-1)[0]))
        else:
            w_grad.append(slide_id)
            prd_grad.append(s_grad.shape[-1]+1)

        if any(c <= int(torch.argmax(label_surv, dim=-1)[0]) for c in c_surv):
            ac_surv+=1
            prd_surv.append(int(torch.argmax(label_surv, dim=-1)[0]))
            
            all_risk_scores[batch_idx] = max(-(min(c_surv)), risk)

        else:
            w_surv.append(slide_id)
            prd_surv.append(label_surv.shape[-1]+1)

        if lb_risk[-1]<=gt_risk<=ub_risk[-1]:
            ac_risk+=1
            prd_risk.append(gt_risk)
            
            all_risk_scores[batch_idx] = ub_risk[-1]

        else:
            w_risk.append(slide_id)
            prd_risk.append(0)

        results.append({
            'slide_id': slide_id,
            'level': level,
            'label_g': label_g,
            'qhat_grad': qhat_grad,
            'pred_grad': s_grad,
            'cal_grad': c_grad,
            'grade': p_grad,
            'unc_grad': un_grad[-1],


            'label_s': label_surv,
            'time': event_time,
            'censorship': c,
            'pred_surv': s_surv,
            'risk': risk,
            'gt_risk': gt_risk,
            'qhat_risk': qhat_risk,
            'cal_risk_set': c_risk,
            'lb_risk': lb_risk[-1],
            'ub_risk': ub_risk[-1],
            'unc_risk': un_risk[-1]
            
        })

    # Save results to Excel
    df_results = pd.DataFrame(results)
    df_results.to_excel(args.results_dir + '/'+args.data_type+'_'+category+'_calibration.xlsx', index=False)
    
    #get average uncertainty
    unc_grad = sum(un_grad)/len(un_grad)
    unc_surv = sum(un_surv)/len(un_surv)
    unc_risk = sum(un_risk)/len(un_risk)

    #dg -> highest cardinality of calibration set
    dg_grad = len(max(cal_grad, key=len))
    dg_surv = len(max(cal_surv, key=len))
    dg_risk = len(max(cal_risk, key=len))

    #display results
    m1='\n Correct Predictions -> grad = %d \t surv = %d \t risk = %d '%(ac_grad, ac_surv, ac_risk)
    m2='\n Wrong Predictions -> grad = %d \t surv = %d \t risk = %d '%(len(w_grad), len(w_surv), len(w_risk))
    acc_grad = ac_grad/len(cal_grad)
    acc_surv = ac_surv/len(cal_surv)
    acc_risk = ac_risk/len(cal_risk)

    msg_unc = '\n unc_grad=%.5f \t unc_surv=%.5f \t unc_risk=%.5f'%(unc_grad, unc_surv, unc_risk)
    msg_dg = '\n dg_grad=%d \t dg_surv=%d \t dg_risk=%d '%(dg_grad, dg_surv, dg_risk)
    msg_acc = '\n coverage_grad=%.5f \t coverage_surv=%.5f \t coverage_risk=%.5f'%(acc_grad, acc_surv, acc_risk)
    msg=category+'->\n'+m1+m2+msg_unc+msg_dg+msg_acc

    print(msg)
    with open(writer, "a") as log_file:
        log_file.write('%s\n' % msg)  # save the message

    dct = { 'category': category,
            'correct_grad': int(ac_grad),
            'wrong_grad': len(w_grad),
            'unc_grad': unc_grad,
            'cov_grad': acc_grad,
            'correct_risk': int(ac_risk),
            'wrong_risk': len(w_risk),
            'unc_risk': unc_risk,
            'cov_risk': acc_risk
            }

    return dct





