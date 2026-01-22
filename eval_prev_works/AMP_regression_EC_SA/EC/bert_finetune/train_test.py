import argparse
import torch
from torch import nn
import pandas as pd
import torch.optim as optim
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr, kendalltau
from EC.bert_finetune.model_def import REG
from EC.bert_finetune.seq_dataloader import _get_train_data_loader, _get_test_data_loader, freeze
from datetime import datetime
import os

def train(seed, batch_size, train_fract, train_data, valid_data, *,
          epochs, frozen_layers, lr, weight_decay, early_stopping):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    train_loader = _get_train_data_loader(batch_size, train_data, train_fract)
    valid_loader = _get_test_data_loader(500, valid_data)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # set the seed for generating random numbers


    model = REG()
    freeze(model, frozen_layers)
    model = model.to(device)

    optimizer = optim.AdamW(
        filter(lambda x: x.requires_grad is not False,
        model.parameters()),
        lr=lr,
        weight_decay=weight_decay)

    #loss_fn = ConcordanceCorCoeff()
    loss_fn = nn.MSELoss()
    #loss_fn = MeanSquaredLogError().to(device)
    mse_list = []
    r2_list = []
    pcc_list = []
    ktc_list = []

    test_mse_list = []
    test_r2_list = []
    test_pcc_list = []
    test_ktc_list = []
    test_min_mse = 0.5
    test_min_mse_epoch = 1
    test_min_mse_pcc = 0.5
    test_min_mse_r2 = 0.0
    if os.path.exists('prot_bert_finetune_amazon.pkl'):
        os.remove("prot_bert_finetune_amazon.pkl")
    for epoch in range(epochs): # args.epochs
        model.train()
        train_predict_list = []
        train_target_list = []
        start = datetime.now()
        for step, batch in enumerate(train_loader):
            b_input_ids = batch['input_ids'].to(device)
            b_input_mask = batch['attention_mask'].to(device)
            b_labels = batch['targets'].to(device)

            predict_MIC, _ = model(b_input_ids, attention_mask=b_input_mask)
            loss = loss_fn(predict_MIC.view(-1), b_labels.float().view(-1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()
            optimizer.zero_grad()

            #scheduler.step()
            train_predict_list.append(predict_MIC.cpu().data.numpy())
            train_target_list.append(b_labels.cpu())
        train_predict_list = np.concatenate(train_predict_list).flatten()
        train_target_list = np.concatenate(train_target_list).flatten()

        mse_epoch = mean_squared_error(train_predict_list, train_target_list)
        r2_epoch = r2_score(train_predict_list, train_target_list)
        pcc_epoch = pearsonr(train_predict_list, train_target_list)[0]
        ktc_epoch = kendalltau(train_predict_list, train_target_list)[0]

        print(f'[Epoch {epoch + 1} TRAIN] - MSE {mse_epoch:.2f}, R2 {r2_epoch:.2f}, PCC {pcc_epoch:.2f}, ktau {ktc_epoch:.2f}')
        mse_list.append(mse_epoch)
        r2_list.append(r2_epoch)
        pcc_list.append(pcc_epoch)
        ktc_list.append(ktc_epoch)

        test_predict_list = []
        test_target_list = []

        model.eval()
        with torch.no_grad():
            for batch in valid_loader:
                b_input_ids = batch['input_ids'].to(device)
                b_input_mask = batch['attention_mask'].to(device)
                b_labels = batch['targets'].to(device)
                predict_MIC, _ = model(b_input_ids, attention_mask=b_input_mask)

                test_predict_list.append(predict_MIC.cpu().numpy())
                test_target_list.append(b_labels.cpu())
        test_predict_list = np.concatenate(test_predict_list).flatten()
        test_target_list = np.concatenate(test_target_list).flatten()

        test_mse = mean_squared_error(test_predict_list, test_target_list)
        test_r2 = r2_score(test_predict_list, test_target_list)
        test_pcc = pearsonr(test_predict_list, test_target_list)[0]
        test_ktc = kendalltau(test_predict_list, test_target_list)[0]
        print(f'[TEST {datetime.now() - start}] - MSE {test_mse:.2f}, R2 {test_r2:.2f}, PCC {test_pcc:.2f}, ktau {test_ktc:.2f}\n')

        test_mse_list.append(test_mse)
        test_r2_list.append(test_r2)
        test_pcc_list.append(test_pcc)
        test_ktc_list.append(test_ktc)
        if test_mse < test_min_mse and early_stopping:
            test_min_mse = test_mse
            test_min_mse_pcc = test_pcc
            test_min_mse_epoch = epoch
            test_min_mse_r2 = test_r2
            torch.save(model.state_dict(), "prot_bert_finetune_amazon.pkl")

        result_dict = {"seed": seed, "train_mse": mse_list, "pcc": r2_list,
                       "test_mse": test_mse_list, "test_pcc": test_pcc_list, "test_r2":test_r2_list,
                       "best_mse": test_min_mse, "best_pcc": test_min_mse_pcc}
        result_df = pd.DataFrame(result_dict)
        #result_df.to_csv(args.result_dir)
    if early_stopping:
        model.load_state_dict(torch.load("prot_bert_finetune_amazon.pkl"))

    return test_predict_list, model, test_min_mse_epoch, test_min_mse, test_min_mse_pcc, test_min_mse_r2

if __name__ == "__main__":
    max_num_split = 1
    max_num_seed = 1
    for i in range(max_num_split):
        train_path = f"data/train-EC.csv"
        test_path = f"data/test-EC.csv"
        best_path = f".cache/{i}/mse_loss_best_result-EC.csv"
        best_seed_list, best_mse_epoch_list, best_mse_list, best_mse_pcc_list, best_mse_r2_list = [], [], [], [], []
        for my_seed in range(max_num_seed):
            result_path = f".cache/{i}/mse_loss_result-EC-{my_seed}.csv"

            parser = argparse.ArgumentParser()
            parser.add_argument("--seed", type=int, default=my_seed, metavar="S", help="random seed (default: 43)")
            parser.add_argument("--train_dir", type=str, default=train_path)
            parser.add_argument("--train_frac", type=float, default=1.0)
            parser.add_argument("--test_dir", type=str, default=test_path)
            parser.add_argument("--result_dir", type=str, default=result_path)
            parser.add_argument(
                "--batch-size", type=int, default=12, metavar="N", help="input batch size for training (default: 1)"
            )
            # parser.add_argument("--train_dir", type=str, default= "/home/jianxiu/Documents/EC/data/0/train-EC_ad.csv")
            # parser.add_argument("--test_dir", type=str, default="/home/jianxiu/Documents/EC/data/0/test-EC_ad.csv")
            parser.add_argument("--frozen_layers", type=int, default=0, metavar="NL",
                                help="number of frozen layers(default: 0)")
            parser.add_argument("--lr", type=float, default=1e-5, metavar="LR", help="learning rate (default: 0.3e-5)")
            parser.add_argument("--weight_decay", type=float, default=3e-3, metavar="M",
                                help="weight_decay (default: 0.01)")
            parser.add_argument("--epochs", type=int, default=100, metavar="N",
                                help="number of epochs to train (default: 100)")

            args = parser.parse_args()
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)
            test_predict_list, model, best_epoch, best_test_mse, best_test_pcc, best_test_r2 = train(args.seed, args.batch_size,
                                                                                      args.train_frac, train_data,
                                                                                      test_data,
                                                                                      epochs=args.epochs,
                                                                                      frozen_layers=args.frozen_layers,
                                                                                      lr=args.lr,
                                                                                      weight_decay=args.weight_decay
                                                                                      )
            best_mse_epoch_list.append(best_epoch)
            best_mse_list.append(best_test_mse)
            best_mse_pcc_list.append(best_test_pcc)
            best_mse_r2_list.append(best_test_r2)

        best_result_dict = {"best_epoch" : best_mse_epoch_list,
                            "best_mse": best_mse_list, "best_mse_pcc": best_mse_pcc_list, "best_mse_r2": best_mse_r2_list}
        best_result_df = pd.DataFrame(best_result_dict)
        #best_result_df.to_csv(best_path)
    print("smart")
