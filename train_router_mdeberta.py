import argparse
import json
import os
import random

import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from tqdm import tqdm
from transformers import AutoTokenizer, DebertaV2Model
from utils.meters import AverageMeter
import numpy as np
import wandb  # Weights & Biases

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class RouterDataset(Dataset):
    def __init__(self,
                 data_path,
                 source_max_token_len: int = 512,
                 target_max_token_len: int = 512,
                 size: int = None,
                 data_type: str = "multi_attempt",
                 dataset_id = 0):
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
        question_id = self.tokenizer(
            question,
            max_length=self.target_max_token_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        question_id['input_ids'] = question_id.input_ids.flatten()
        question_id['attention_mask'] = question_id.attention_mask.flatten()
        cluster_id = data_point.get('cluster_id', 0)
        return question_id, scores, self.dataset_id, cluster_id

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
        top_vals, top_idx = index_true.sort(dim=-1, descending=True)
        last_vals, neg_idx = index_true.topk(k=last_k, largest=False, dim=-1)
        for i in range(top_k):
            pos_idx = top_idx[:, i].view(-1, 1)
            mask = (top_vals[:, i].view(-1, 1) > 0).float()
            pos_score = torch.gather(x, 1, pos_idx)
            neg_score = torch.gather(x, 1, neg_idx)
            neg_score = torch.where(last_vals > 0.5, float('-inf'), neg_score)
            cat = torch.cat([pos_score, neg_score], dim=-1)
            logp = torch.log_softmax(cat, dim=-1)[:, 0]
            loss += torch.mean(-logp * mask)
        return loss

    def compute_sample_sample_loss_with_task_tag(self, hidden_state, dataset_ids, t, H=3):
        similar_score = self.compute_similarity(hidden_state, hidden_state)
        all_index = []
        for dataset_id in dataset_ids:
            pos_idxs = torch.nonzero(dataset_ids == dataset_id)
            neg_idxs = torch.nonzero(dataset_ids != dataset_id)
            if len(neg_idxs) < H:
                continue
            pos_sel = random.choice(pos_idxs).view(-1)
            neg_sel = neg_idxs[torch.randperm(len(neg_idxs))[:H]].view(-1)
            all_index.append(torch.cat([pos_sel.unsqueeze(0), neg_sel]))
        if not all_index:
            return torch.tensor(0.0, device=hidden_state.device)
        idx_stack = torch.stack(all_index)
        sel = torch.gather(similar_score, 1, idx_stack)
        logp = torch.log_softmax(sel, dim=-1)
        return torch.mean(-logp[:, 0])

    def compute_cluster_loss(self, hidden_state, cluster_ids, t, H=3):
        sim_score = self.compute_similarity(hidden_state, hidden_state)
        all_index = []
        for cid in cluster_ids:
            pos_idxs = torch.nonzero(cluster_ids == cid)
            neg_idxs = torch.nonzero(cluster_ids != cid)
            if len(neg_idxs) < H:
                continue
            pos_sel = random.choice(pos_idxs).view(-1)
            neg_sel = neg_idxs[torch.randperm(len(neg_idxs))[:H]].view(-1)
            all_index.append(torch.cat([pos_sel.unsqueeze(0), neg_sel]))
        if not all_index:
            return torch.tensor(0.0, device=hidden_state.device)
        idx_stack = torch.stack(all_index)
        sel = torch.gather(sim_score, 1, idx_stack)
        logp = torch.log_softmax(sel, dim=-1)
        return torch.mean(-logp[:, 0])

def evaluation(router_model, dataset_paths, dataset_types, tokenizer, batch_size, device):
    result = {}
    with torch.no_grad():
        assert len(dataset_paths) == len(dataset_types)
        for idx, data_path in enumerate(dataset_paths):
            test_ds = RouterDataset(data_path)
            test_ds.register_tokenizer(tokenizer)
            dl = DataLoader(test_ds, batch_size=batch_size, shuffle=True)
            correct = 0
            correct_pred = 0
            for inputs, scores, _, _ in dl:
                inputs = inputs.to(device)
                scores = scores.to(device)
                logits, _ = router_model.forward(**inputs)
                preds = torch.softmax(logits, dim=1).argmax(dim=1)
                targets = scores.argmax(dim=1)
                correct += (preds == targets).sum().item()
                if dataset_types[idx] in ("probability", "multi_attempt"):
                    mask = torch.zeros_like(scores)
                    mask.scatter_(1, preds.unsqueeze(1), 1)
                    correct_pred += (scores * mask).sum().item()
            result[data_path] = [
                correct / len(test_ds),
                correct_pred / len(test_ds)
            ]
    return result

if __name__ == '__main__':
    device = "cuda"
    parser = argparse.ArgumentParser(description="RouterDC training with W&B")
    # dataset args
    parser.add_argument('--data_paths', nargs='+', required=True)
    parser.add_argument('--test_data_paths', nargs='+', default=[])
    parser.add_argument('--test_data_type', nargs='+', default=[])
    parser.add_argument('--final_eval_data_paths', nargs='+', default=[])
    parser.add_argument('--final_eval_data_type', nargs='+', default=[])

    # training args
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--training_steps', type=int, default=1000)
    parser.add_argument('--eval_steps', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--gradient_accumulation', type=int, default=1)
    parser.add_argument('--save_path', type=str, default='./logs/router_debug/')
    parser.add_argument('--top_k', type=int, default=3)
    parser.add_argument('--last_k', type=int, default=3)
    parser.add_argument('--tempreture', type=float, default=1)
    parser.add_argument('--similarity_function', type=str, default='cos')
    parser.add_argument('--sample_loss_weight', type=float, default=0)
    parser.add_argument('--cluster_loss_weight', type=float, default=0)
    parser.add_argument('--H', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--training_samples_per_dataset', type=int, default=1000)

    # W&B args
    parser.add_argument('--wandb_entity', type=str, default=None)
    parser.add_argument('--project_name', type=str, default=None)

    # final eval flag
    parser.add_argument('--final_eval', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)
    setup_seed(args.seed)

    # initialize W&B
    if args.project_name:
        wandb.init(
            entity=args.wandb_entity,
            project=args.project_name,
            config=vars(args),
            save_code=True
        )

    # tokenizer and model setup
    tokenizer = AutoTokenizer.from_pretrained("microsoft/mdeberta-v3-base", truncation_side='left', padding=True)
    encoder_model = DebertaV2Model.from_pretrained("microsoft/mdeberta-v3-base")

    # prepare datasets
    router_datasets = []
    for i, path in enumerate(args.data_paths):
        ds = RouterDataset(path, size=args.training_samples_per_dataset, data_type=None, dataset_id=i)
        ds.register_tokenizer(tokenizer)
        router_datasets.append(ds)
    train_ds = ConcatDataset(router_datasets)

    # model & optimizer
    router_model = RouterModule(
        encoder_model,
        hidden_state_dim=768,
        node_size=len(router_datasets[0].router_node),
        similarity_function=args.similarity_function
    ).to(device)
    optimizer = torch.optim.AdamW(router_model.parameters(), lr=args.learning_rate)

    # training loop
    print("Training start!!!")
    step = 0
    pbar = tqdm(total=args.training_steps)
    max_avg = 0

    while step < args.training_steps:
        losses = AverageMeter('Loss', ':3.2f')
        dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        for inputs, scores, dataset_ids, cluster_ids in dl:
            optimizer.zero_grad()
            inputs = inputs.to(device)
            scores = scores.to(device)
            cluster_ids = cluster_ids.to(device)

            logits, hidden = router_model.forward(t=args.tempreture, **inputs)
            loss = router_model.compute_sample_llm_loss(logits, scores, args.top_k, args.last_k)
            if args.cluster_loss_weight:
                loss += args.cluster_loss_weight * router_model.compute_cluster_loss(hidden, cluster_ids, args.tempreture)
            loss.backward()
            if step % args.gradient_accumulation == 0:
                optimizer.step()

            losses.update(loss.item(), inputs['input_ids'].size(0))
            step += 1
            pbar.update(1)
            pbar.set_postfix({'step': step, 'loss': losses.avg})

            # log to W&B
            if args.project_name:
                wandb.log({'train/loss': losses.avg, 'train/step': step})

            if step >= args.training_steps:
                break

            # Commented out evaluation section:
            # if step % args.eval_steps == 0 and args.test_data_paths:
            #     print("Validation start")
            #     val_res = evaluation(router_model, args.data_paths, args.test_data_type, tokenizer, args.batch_size, device)
            #     print("Test start")
            #     test_res = evaluation(router_model, args.test_data_paths, args.test_data_type, tokenizer, args.batch_size, device)
            #     avg_test = sum(v[1] for v in test_res.values()) / len(test_res)
            #     print("avg test", avg_test)
            #     if args.project_name:
            #         wandb.log({'eval/avg_test_acc': avg_test, 'eval/step': step})
            #     if avg_test > max_avg:
            #         torch.save(router_model.state_dict(), os.path.join(args.save_path, "best_model.pth"))
            #         max_avg = avg_test

        if step >= args.training_steps:
            break

    pbar.close()

    # Commented out final evaluation
    # if args.final_eval:
    #     print("Final evaluation")
    #     state = torch.load(os.path.join(args.save_path, "best_model.pth"))
    #     router_model.load_state_dict(state)
    #     final_res = evaluation(router_model, args.final_eval_data_paths, args.final_eval_data_type, tokenizer, 32, device)
    #     print(final_res)

    # save the final model
    torch.save(router_model.state_dict(), os.path.join(args.save_path, 'final_model.pth'))
    print(f"Training complete. Model saved to {args.save_path}/final_model.pth")

    # save config and log
    with open(os.path.join(args.save_path, "config.txt"), 'w') as f:
        f.write(str(args))

    if args.project_name:
        wandb.finish()
