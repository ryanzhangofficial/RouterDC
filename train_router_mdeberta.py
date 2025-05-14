import argparse
import json
import os
import random

import wandb
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
    def __init__(self, data_path, size=None, dataset_id=0):
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        if size:
            while len(self.data) < size:
                self.data.extend(self.data)
            self.data = self.data[:size]
        self.router_node = list(self.data[0]['scores'].keys())
        self.tokenizer = None
        self.dataset_id = dataset_id

    def __getitem__(self, index):
        dp = self.data[index]
        scores = torch.tensor(list(dp['scores'].values()), dtype=torch.float)
        enc = self.tokenizer(
            dp['question'],
            max_length=512,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        enc['input_ids'] = enc.input_ids.flatten()
        enc['attention_mask'] = enc.attention_mask.flatten()
        cluster_id = dp.get('cluster_id', 0)
        return enc, scores, self.dataset_id, cluster_id

    def __len__(self):
        return len(self.data)

    def register_tokenizer(self, tok):
        self.tokenizer = tok

class RouterModule(nn.Module):
    def __init__(self, backbone, node_size, similarity_function="cos"):
        super().__init__()
        self.backbone = backbone
        hidden_dim = backbone.config.hidden_size
        self.embeddings = nn.Embedding(node_size, hidden_dim)
        with torch.no_grad():
            nn.init.normal_(self.embeddings.weight, mean=0, std=0.78)
        self.sim_fn = similarity_function

    def compute_similarity(self, h, e):
        if self.sim_fn == "cos":
            return (h @ e.T) / (h.norm(dim=1).unsqueeze(1) * e.norm(dim=1).unsqueeze(0))
        return h @ e.T

    def forward(self, t=1, **inputs):
        out = self.backbone(**inputs)
        h = out.last_hidden_state[:,0,:]
        sim = self.compute_similarity(h, self.embeddings.weight) / t
        return sim, h

    def compute_sample_llm_loss(self, sim, scores, top_k, last_k):
        loss = 0.0
        top_vals, top_idx = scores.sort(dim=-1, descending=True)
        _, neg_idx = scores.topk(k=last_k, largest=False, dim=-1)
        for i in range(top_k):
            pos = top_idx[:,i].unsqueeze(1)
            mask = (top_vals[:,i].unsqueeze(1) > 0).float()
            pos_score = sim.gather(1, pos)
            neg_score = sim.gather(1, neg_idx)
            neg_score = torch.where(scores.topk(k=last_k, largest=False)[0] > 0, float("-inf"), neg_score)
            cat = torch.cat([pos_score, neg_score], dim=1)
            logp = torch.log_softmax(cat, dim=1)[:,0]
            loss = loss + (-logp * mask.squeeze(1)).mean()
        return loss

    def compute_cluster_loss(self, h, cluster_ids, t, H=3):
        sim_mat = self.compute_similarity(h, h)
        idxs = []
        for cid in cluster_ids:
            pos = torch.nonzero(cluster_ids==cid).view(-1)
            neg = torch.nonzero(cluster_ids!=cid).view(-1)
            if len(neg) < H or len(pos) < 1:
                continue
            p = pos[torch.randint(len(pos),(1,))]
            n = neg[torch.randperm(len(neg))[:H]]
            idxs.append(torch.cat([p.unsqueeze(0), n]))
        if not idxs:
            return torch.tensor(0.0, device=h.device)
        idxs = torch.stack(idxs)
        sel = sim_mat.gather(1, idxs)
        logp = torch.log_softmax(sel, dim=1)
        return -logp[:,0].mean()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_paths', nargs='+', required=True)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--training_steps', type=int, default=1000)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--top_k', type=int, default=3)
    parser.add_argument('--last_k', type=int, default=3)
    parser.add_argument('--tempreture', type=float, default=1)
    parser.add_argument('--sample_loss_weight', type=float, default=0)
    parser.add_argument('--cluster_loss_weight', type=float, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--similarity_function', type=str, default='cos')
    parser.add_argument('--wandb_entity', type=str, default=None)
    parser.add_argument('--project_name', type=str, default=None)
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)
    setup_seed(args.seed)

    if args.project_name:
        wandb.init(entity=args.wandb_entity,
                   project=args.project_name,
                   config=vars(args),
                   save_code=True)

    tok = AutoTokenizer.from_pretrained("microsoft/mdeberta-v3-base")
    enc_mod = DebertaV2Model.from_pretrained("microsoft/mdeberta-v3-base")
    datasets = []
    for i, p in enumerate(args.data_paths):
        ds = RouterDataset(p, dataset_id=i)
        ds.register_tokenizer(tok)
        datasets.append(ds)
    train_ds = ConcatDataset(datasets)

    model = RouterModule(enc_mod, node_size=len(datasets[0].router_node),
                         similarity_function=args.similarity_function).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    meter = AverageMeter('Loss', ':3.2f')

    step = 0
    pbar = tqdm(total=args.training_steps)
    while step < args.training_steps:
        for batch in loader:
            opt.zero_grad()
            inputs, scores, _, cluster_ids = batch
            inputs = {k: v.cuda() for k,v in inputs.items()}
            scores = scores.cuda()
            cluster_ids = cluster_ids.cuda()

            sim, hidden = model.forward(t=args.tempreture, **inputs)
            loss = model.compute_sample_llm_loss(sim, scores, args.top_k, args.last_k)
            if args.cluster_loss_weight:
                loss = loss + args.cluster_loss_weight * model.compute_cluster_loss(hidden, cluster_ids, args.tempreture)
            loss.backward()
            opt.step()

            meter.update(loss.item(), inputs['input_ids'].size(0))
            pbar.set_postfix({'loss': meter.avg})
            pbar.update(1)
            step += 1

            if args.project_name:
                wandb.log({'loss': loss.item(), 'step': step})
            if step >= args.training_steps:
                break
        loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    pbar.close()

    torch.save(model.state_dict(), os.path.join(args.save_path, 'final_model.pth'))
    print(f"Training complete. Model saved to {args.save_path}/final_model.pth")

    if args.project_name:
        wandb.finish()

if __name__ == '__main__':
    main()
