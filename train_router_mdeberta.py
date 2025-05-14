
import argparse
import json
import os
import random

import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, DebertaV2Model
from utils.meters import AverageMeter
import numpy as np


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class RouterDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_path,
                 source_max_token_len: int = 512,
                 target_max_token_len: int = 512,
                 size: int = None,
                 data_type: str = "multi_attempt",
                 dataset_id=0):
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        if size:
            while len(self.data) < size:
                self.data.extend(self.data)
            self.data = self.data[:size]
        self.router_node = list(self.data[0]['scores'].keys())
        self.tokenizer = None
        self.source_max_token_len = source_max_token_len
        self.target_max_token_len = target_max_token_len
        self.data_type = data_type
        self.dataset_id = dataset_id

    def __getitem__(self, index):
        data_point = self.data[index]
        scores = torch.tensor(list(data_point['scores'].values()))
        question = data_point['question']
        encoded = self.tokenizer(
            question,
            max_length=self.target_max_token_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        encoded['input_ids'] = encoded.input_ids.flatten()
        encoded['attention_mask'] = encoded.attention_mask.flatten()
        cluster_id = data_point.get('cluster_id', 0)
        return encoded, scores, self.dataset_id, cluster_id

    def __len__(self):
        return len(self.data)

    def register_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer


class RouterModule(nn.Module):
    def __init__(self, backbone, hidden_state_dim=768, node_size=3, similarity_function="cos"):
        super(RouterModule, self).__init__()
        self.backbone = backbone
        self.hidden_state_dim = hidden_state_dim
        self.node_size = node_size
        self.embeddings = nn.Embedding(node_size, hidden_state_dim)
        with torch.no_grad():
            nn.init.normal_(self.embeddings.weight, mean=0, std=0.78)
        self.similarity_function = similarity_function

    def compute_similarity(self, input1, input2):
        if self.similarity_function == "cos":
            return (input1 @ input2.T) / (
                torch.norm(input1, dim=1).unsqueeze(1) * torch.norm(input2, dim=1).unsqueeze(0)
            )
        else:
            return input1 @ input2.T

    def forward(self, t=1, **input_kwargs):
        x = self.backbone(**input_kwargs)
        hidden_state = x['last_hidden_state'][:, 0, :]
        sim = self.compute_similarity(hidden_state, self.embeddings.weight)
        return sim / t, hidden_state

    def compute_sample_llm_loss(self, x, index_true, top_k, last_k):
        loss = 0
        top_true, top_idx = index_true.sort(dim=-1, descending=True)
        last_true, neg_idx = index_true.topk(k=last_k, largest=False, dim=-1)
        for i in range(top_k):
            pos_idx = top_idx[:, i].view(-1, 1)
            mask = (top_true[:, i].view(-1, 1) > 0).float()
            top_x = torch.gather(x, 1, pos_idx)
            neg_x = torch.gather(x, 1, neg_idx)
            neg_x = torch.where(last_true > 0.5, float('-inf'), neg_x)
            concat = torch.cat([top_x, neg_x], dim=-1)
            logp = torch.log_softmax(concat, dim=-1)[:, 0]
            loss += torch.mean(-logp * mask)
        return loss

    def compute_cluster_loss(self, hidden_state, cluster_ids, t, H=3):
        sim_score = self.compute_similarity(hidden_state, hidden_state)
        all_idx = []
        for cid in cluster_ids:
            pos = torch.nonzero(cluster_ids == cid)
            neg = torch.nonzero(cluster_ids != cid)
            if len(neg) < H:
                continue
            pos_sel = random.choice(pos).view(-1)
            neg_sel = neg[torch.randperm(len(neg))[:H]].view(-1)
            all_idx.append(torch.cat([pos_sel.unsqueeze(0), neg_sel]))
        if not all_idx:
            return torch.tensor(0.0, device=hidden_state.device)
        idx_stack = torch.stack(all_idx)
        sel = torch.gather(sim_score, 1, idx_stack)
        logp = torch.log_softmax(sel, dim=-1)
        return torch.mean(-logp[:, 0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_paths', nargs='+', required=True)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--training_steps', type=int, default=1000)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--save_path', type=str, default='./logs/router_no_eval/')
    parser.add_argument('--top_k', type=int, default=3)
    parser.add_argument('--last_k', type=int, default=3)
    parser.add_argument('--tempreture', type=float, default=1)
    parser.add_argument('--sample_loss_weight', type=float, default=0)
    parser.add_argument('--cluster_loss_weight', type=float, default=1)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)
    setup_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained("microsoft/mdeberta-v3-base")
    encoder = DebertaV2Model.from_pretrained("microsoft/mdeberta-v3-base")

    datasets = []
    for idx, path in enumerate(args.data_paths):
        ds = RouterDataset(path, size=None, data_type=None, dataset_id=idx)
        ds.register_tokenizer(tokenizer)
        datasets.append(ds)
    train_data = ConcatDataset(datasets)

    model = RouterModule(encoder, hidden_state_dim=768, node_size=len(datasets[0].router_node), similarity_function=args.similarity_function)
    model.cuda()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    meter = AverageMeter('Loss', ':3.2f')

    step = 0
    pbar = tqdm(total=args.training_steps)
    while step < args.training_steps:
        for batch in loader:
            optimizer.zero_grad()
            inputs, scores, _, cluster_ids = batch
            inputs = {k: v.cuda() for k, v in inputs.items()}
            scores = scores.cuda()
            cluster_ids = cluster_ids.cuda()

            sim, hidden = model.forward(t=args.tempreture, **inputs)
            loss = model.compute_sample_llm_loss(sim, scores, args.top_k, args.last_k)
            if args.cluster_loss_weight:
                loss += args.cluster_loss_weight * model.compute_cluster_loss(hidden, cluster_ids, args.tempreture)
            loss.backward()
            optimizer.step()

            meter.update(loss.item(), inputs['input_ids'].size(0))
            pbar.set_postfix({'loss': meter.avg})
            pbar.update(1)
            step += 1
            if step >= args.training_steps:
                break
        loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    pbar.close()

    torch.save(model.state_dict(), os.path.join(args.save_path, 'final_model.pth'))
    print(f"Training complete. Model saved to {args.save_path}/final_model.pth")

if __name__ == '__main__':
    main()
