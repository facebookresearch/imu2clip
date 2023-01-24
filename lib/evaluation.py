# Copyright (c) Meta Platforms, Inc. and affiliates.
# LICENSE file in the root directory of this source tree.

import torch
import json
from tqdm import tqdm
from torchmetrics import RetrievalRecall, RetrievalMRR


def evaluate(
    test_set,
    collate_fn,
    model,
    source_modality,
    target_modalities,
    result_path,
    configs=None,
):
    dataloader = torch.utils.data.DataLoader(
        test_set,
        batch_size=32,
        num_workers=20,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True,
    )
    device = torch.device("cuda:0")
    target_modalities.sort()
    list_modalities = [source_modality] + target_modalities
    if "imu" in list_modalities:
        imu_encoder = model.imu_encoder
        imu_encoder.to(device)
        imu_encoder.eval()
    if "text" in list_modalities:
        text_encoder = model.text_encoder
        text_encoder.to(device)
        text_encoder.eval()
    if "video" in list_modalities:
        video_encoder = model.video_encoder
        video_encoder.to(device)
        video_encoder.eval()
    if "audio" in list_modalities:
        audio_encoder = model.audio_encoder
        audio_encoder.to(device)
        audio_encoder.eval()

    out = {"imu": [], "text": [], "video": [], "audio": []}
    with torch.no_grad():
        for batch in tqdm(dataloader):
            if "imu" in list_modalities:
                x_imu = batch["imu"].to(device)
                y_imu = imu_encoder(x_imu)
                out["imu"].append(y_imu)

            if "text" in list_modalities:
                x_narration = batch["narration"]
                y_narration = text_encoder.get_text_embeddings(x_narration, device)
                out["text"].append(y_narration)

            if "video" in list_modalities:
                x_video = batch["video"].to(device)
                y_video = video_encoder.get_video_embeddings(x_video)
                out["video"].append(y_video)

            if "audio" in list_modalities:
                x_audio = batch["audio"].to(device)
                y_audio = audio_encoder.get_audio_embeddings(x_audio)
                out["audio"].append(y_audio)

    y_query_modality = torch.cat(out[source_modality], dim=0)

    if "text" in target_modalities:
        y_key_modality = torch.cat(out["text"], dim=0)

    elif "video" in target_modalities:
        y_key_modality = torch.cat(out["video"], dim=0)

    elif "audio" in target_modalities:
        y_key_modality = torch.cat(out["audio"], dim=0)

    s_t_metrics, t_s_metrics = compute_metrics(y_query_modality, y_key_modality)

    # Save metrics
    num_candidates = y_query_modality.shape[0]
    metrics = {
        "s_t_metrics": s_t_metrics,
        "t_s_metrics": t_s_metrics,
        "num_candidates": num_candidates,
    }
    result_path += f"_candi_num_{num_candidates}.json"
    with open(result_path, "w") as f:
        json.dump({"metrics": metrics, "configs": configs}, f, indent=4)

    return metrics


def compute_metrics(source_embeddings, target_embeddings):
    """
    input:
    - source_embeddings: (n, m)
    - target_embeddings: (n, m)
    output:
    - Recall@1
    - Recall@10
    - Recall@50
    - MRR
    """
    # prepare metrics
    compute_mrr = RetrievalMRR()
    compute_r1 = RetrievalRecall(k=1)
    compute_r10 = RetrievalRecall(k=10)
    compute_r50 = RetrievalRecall(k=50)
    s_t_metrics = {"MRR": 0, "R@1": 0, "R@10": 0, "R@50": 0}
    t_s_metrics = {"MRR": 0, "R@1": 0, "R@10": 0, "R@50": 0}
    n = source_embeddings.shape[0]
    print(f"the number of queries & candidates = {n}")
    target = torch.eye(n).view(-1)
    indexes = torch.arange(n).repeat(n, 1).transpose(0, 1)
    indexes = indexes.reshape(-1)
    #  Compute similarity
    s = torch.nn.functional.normalize(source_embeddings, dim=1)
    t = torch.nn.functional.normalize(target_embeddings, dim=1)
    tt = t.transpose(0, 1)
    st = s.transpose(0, 1)
    # Do query batch by batch to avoid OOM issue.
    bsz = 32
    batch_num = n // bsz
    print("Start batch retrieval:")
    # s -> t
    s_t_batch_results = []
    for i in tqdm(range(batch_num)):
        start = i * bsz
        end = min((i + 1) * bsz, n)
        query_batch = torch.mm(s[start:end], tt)  # (bsz, m) (m, n) -> (bsz, n)
        s_t_batch_results.append(query_batch)
    s_t_batch_results = torch.cat(s_t_batch_results, dim=0).view(-1)  # (n,n)
    mrr = compute_mrr(s_t_batch_results, target, indexes=indexes).item()
    r1 = compute_r1(s_t_batch_results, target, indexes=indexes).item()
    r10 = compute_r10(s_t_batch_results, target, indexes=indexes).item()
    r50 = compute_r50(s_t_batch_results, target, indexes=indexes).item()
    s_t_metrics = {"MRR": mrr, "R@1": r1, "R@10": r10, "R@50": r50}

    # t -> s
    t_s_batch_results = []
    for i in tqdm(range(batch_num)):
        start = i * bsz
        end = min((i + 1) * bsz, n)
        query_batch = torch.mm(t[start:end], st)  # (bsz, m) (m, n) -> (bsz, n)
        t_s_batch_results.append(query_batch)
    t_s_batch_results = torch.cat(t_s_batch_results, dim=0).view(-1)  # (n,n)
    mrr = compute_mrr(t_s_batch_results, target, indexes=indexes).item()
    r1 = compute_r1(t_s_batch_results, target, indexes=indexes).item()
    r10 = compute_r10(t_s_batch_results, target, indexes=indexes).item()
    r50 = compute_r50(t_s_batch_results, target, indexes=indexes).item()
    t_s_metrics = {"MRR": mrr, "R@1": r1, "R@10": r10, "R@50": r50}

    return s_t_metrics, t_s_metrics