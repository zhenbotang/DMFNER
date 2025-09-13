import torch
from scipy.optimize import linear_sum_assignment
from .lap import auction_lap
from torch import nn

class HungarianMatcher(nn.Module):

    def __init__(self, cost_class: float = 1, cost_span: float = 1, match_boundary_type = 'f1', solver = "hungarian"):
        super().__init__()
        self.cost_class = cost_class
        self.cost_span = cost_span
        self.match_boundary_type = match_boundary_type
        self.solver = solver

    @torch.no_grad()
    def forward(self, outputs, targets):
        if self.solver == "order":
            sizes = targets["sizes"]
            indices = [(list(range(size)),list(range(size))) for size in sizes]
        else:
            bs, num_queries = outputs["pred_logits"].shape[:2]

            # We flatten to compute the cost matrices in a batch
            out_prob = outputs["pred_logits"].flatten(0, 1).softmax(dim=-1) # [batch_size * num_queries, 8]

            entity_left = outputs["pred_left"].flatten(0, 1)
            entity_right = outputs["pred_right"].flatten(0, 1) # [batch_size * num_queries]


            gt_ids = targets["labels"]
            gt_left = targets["gt_left"]
            gt_right = targets["gt_right"]
            
            # import pdb;pdb.set_trace()
            cost_class = -out_prob[:, gt_ids]            

            C = None

            # Final cost matrix
            if self.match_boundary_type == "f1":
                entity_left_idx = entity_left.argmax(dim=-1)  # [batch_size * num_queries]
                entity_right_idx = entity_right.argmax(dim=-1)  # [batch_size * num_queries]
                cost_dis = torch.abs(entity_left_idx.unsqueeze(-1) - gt_left.unsqueeze(0)) + torch.abs(entity_right_idx.unsqueeze(-1) - gt_right.unsqueeze(0))
                C = self.cost_span * cost_dis + self.cost_class * cost_class
            
            if self.match_boundary_type == "logp":
                cost_span = -(entity_left[:, gt_left] + entity_right[:, gt_right])
                C = self.cost_span * cost_span + self.cost_class * cost_class

            C = C.view(bs, num_queries, -1)

            sizes = targets["sizes"]
            indices = None
            
            if self.solver == "hungarian":
                C = C.cpu()
                indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
            if self.solver == "auction":
                indices = [auction_lap(c[i])[:2] for i, c in enumerate(C.split(sizes, -1))]

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]