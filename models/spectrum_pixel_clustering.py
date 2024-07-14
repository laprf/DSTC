import torch
import torch.nn as nn
from einops import rearrange
from models.clustering.pair_wise_distance import PairwiseDistFunction
import torch.nn.functional as F


@torch.no_grad()
def get_abs_indices(init_label_map, num_spixels_width):
    """Get absolute indices for the initial label map."""
    b, n_pixel = init_label_map.shape
    device = init_label_map.device
    r = torch.arange(-1, 2.0, device=device)
    relative_spix_indices = torch.cat([r - num_spixels_width, r, r + num_spixels_width], 0)

    abs_pix_indices = torch.arange(n_pixel, device=device)[None, None].repeat(b, 9, 1).reshape(-1).long()
    abs_spix_indices = (init_label_map[:, None] + relative_spix_indices[None, :, None]).reshape(-1).long()
    abs_batch_indices = torch.arange(b, device=device)[:, None, None].repeat(1, 9, n_pixel).reshape(-1).long()

    return torch.stack([abs_batch_indices, abs_spix_indices, abs_pix_indices], 0)


def calc_init_centroid(images, num_spixels_width, num_spixels_height):
    """Calculate initial superpixels and return centroids and initial label map."""
    batchsize, channels, height, width = images.shape
    device = images.device

    centroids = F.adaptive_avg_pool2d(images, (num_spixels_height, num_spixels_width))

    with torch.no_grad():
        num_spixels = num_spixels_width * num_spixels_height
        labels = torch.arange(num_spixels, device=device).reshape(1, 1, *centroids.shape[-2:]).type_as(centroids)
        init_label_map = F.interpolate(labels, size=(height, width), mode="nearest")
        init_label_map = init_label_map.repeat(batchsize, 1, 1, 1)

    init_label_map = init_label_map.reshape(batchsize, -1)
    centroids = centroids.reshape(batchsize, channels, -1)

    return centroids, init_label_map


def ssn_iter(pixel_features, deep_feature, sdf_feature, proposals=None, n_iter=2):
    if proposals is None:
        proposals = [2, 2]
    num_spixels_height, num_spixels_width = proposals
    num_spixels = num_spixels_height * num_spixels_width

    spixel_features, init_label_map = calc_init_centroid(pixel_features, num_spixels_width, num_spixels_height)
    abs_indices = get_abs_indices(init_label_map, num_spixels_width)

    pixel_features = pixel_features.reshape(*pixel_features.shape[:2], -1)
    permuted_pixel_features = pixel_features.permute(0, 2, 1).contiguous()

    deep_feature = deep_feature.reshape(*deep_feature.shape[:2], -1)
    sdf_feature = sdf_feature.reshape(*sdf_feature.shape[:2], -1)

    with torch.no_grad():
        for k in range(n_iter):
            if k < n_iter - 1:
                dist_matrix = PairwiseDistFunction.apply(
                    pixel_features, deep_feature, sdf_feature, spixel_features, init_label_map,
                    num_spixels_width, num_spixels_height
                )

                affinity_matrix = (-dist_matrix).softmax(1)
                reshaped_affinity_matrix = affinity_matrix.reshape(-1)

                mask = (abs_indices[1] >= 0) * (abs_indices[1] < num_spixels)
                sparse_abs_affinity = torch.sparse_coo_tensor(abs_indices[:, mask], reshaped_affinity_matrix[mask])

                abs_affinity = sparse_abs_affinity.to_dense().contiguous()
                spixel_features = torch.bmm(abs_affinity, permuted_pixel_features) \
                                  / (abs_affinity.sum(2, keepdim=True) + 1e-16)

                spixel_features = spixel_features.permute(0, 2, 1).contiguous()
            else:
                dist_matrix = PairwiseDistFunction.apply(
                    pixel_features, deep_feature, sdf_feature, spixel_features, init_label_map,
                    num_spixels_width, num_spixels_height
                )

                affinity_matrix = (-dist_matrix).softmax(1)
                reshaped_affinity_matrix = affinity_matrix.reshape(-1)

                mask = (abs_indices[1] >= 0) * (abs_indices[1] < num_spixels)
                sparse_abs_affinity = torch.sparse_coo_tensor(abs_indices[:, mask], reshaped_affinity_matrix[mask])

                abs_affinity = sparse_abs_affinity.to_dense().contiguous()

    return abs_affinity, num_spixels


class Cluster(nn.Module):
    def __init__(self, dim, num_classes, proposal=2, hidden_dim=96, fold=8):
        super().__init__()
        self.proposal = proposal
        self.fold = fold
        self.classes = num_classes

        self.f = nn.Conv2d(dim, hidden_dim, kernel_size=1)
        self.f_sdf1 = nn.Conv2d(dim - 1, hidden_dim, kernel_size=1)

        self.centers_proposal = nn.AdaptiveAvgPool2d((proposal, proposal))

        self.init_weights()

    def cluster_forward(self, x, feat, sdf_data):
        W, H = x.shape[-2:]

        hsi_in = self.f(x)
        value = feat

        b0, c0, w0, h0 = hsi_in.shape
        assert w0 % self.fold == 0 and h0 % self.fold == 0, \
            f"Ensure the feature map size ({w0}*{h0}) can be divided by fold {self.fold}*{self.fold}"
        hsi_in = rearrange(hsi_in, "b c (f1 w) (f2 h) -> (b f1 f2) c w h", f1=self.fold, f2=self.fold)
        value = rearrange(value, "b c (f1 w) (f2 h) -> (b f1 f2) c w h", f1=self.fold, f2=self.fold)
        sdf_data = self.f_sdf1(sdf_data)
        sdf_data = rearrange(sdf_data, "b c (f1 w) (f2 h) -> (b f1 f2) c w h", f1=self.fold, f2=self.fold)

        value_centers = self.centers_proposal(value)

        sim, _ = ssn_iter(value, hsi_in, sdf_data,
                          proposals=[self.proposal, self.proposal], n_iter=2)
        # [B,M,N] 每一个中心点与每一个像素点的相似度，M为中心点个数，N为像素点个数

        # we use mask to sololy assign each point to one center
        _, sim_max_idx = sim.max(dim=1, keepdim=True)  # sim_max_idx: [B,1,N] 找到每一个像素点对应的最大相似度的中心点的索引
        mask = torch.zeros_like(sim)  # binary #[B,M,N]
        mask.scatter_(1, sim_max_idx, 1.)  # binary [B,M,N] 在mask上将每一个像素点对应的最大相似度的中心点的索引处置为1
        sim = sim * mask
        value2 = rearrange(value, 'b c w h -> b (w h) c')  # [B,N,C]

        # aggregate step, out shape [B,M,C]
        value_centers = rearrange(value_centers, 'b c w h -> b (w h) c')  # [B,C_W*C_H,C]
        out = ((value2.unsqueeze(dim=1) * sim.unsqueeze(dim=-1)).sum(dim=2) + value_centers) / (
                sim.sum(dim=-1, keepdim=True) + 1.0)  # [B,M,C]

        out = rearrange(out, "(b f1 f2) (p1 p2) c -> b (f1 f2 p1 p2) c", f1=self.fold, f2=self.fold, p1=self.proposal,
                        p2=self.proposal)
        mask = rearrange(mask, "(b f1 f2) m (w h) -> b (f1 f2) m w h", f1=self.fold, f2=self.fold, w=W // self.fold,
                         h=H // self.fold)
        return mask, out

    def gen_labels(self, gt, mask):
        B, W, H = gt.shape
        f, p = self.fold, self.proposal

        # ------- gen spix_map ------
        coef = torch.arange(0, p * p, dtype=mask.dtype, device=mask.device).view(1, 1, p * p, 1, 1)
        small_imgs = (mask * coef).sum(dim=2)
        bias = torch.arange(0, f * f * p * p, step=p * p, dtype=mask.dtype, device=mask.device).view(1, f * f, 1, 1)
        small_imgs = small_imgs + bias
        spix_map = rearrange(small_imgs, "b (f1 f2) w h -> b (f1 w) (f2 h)", f1=f, f2=f)

        # ------- gen labels ------ 
        gt_reshape = rearrange(gt, "b (f1 w) (f2 h) -> b (f1 f2) w h", f1=f, f2=f) + 1
        # 将gt_reshape与mask相乘，
        gt_filt = gt_reshape.unsqueeze(2) * mask  # Shape: [b, f*f, p*p, w, h]
        # 将gt_filt展平
        gt_filt = gt_filt.flatten(start_dim=3).long()  # Shape: [b, f*f, p*p, w*h]

        # 计算gt_filt沿最后一个维度上，每一个数值的个数
        count = torch.zeros((B, f * f, p * p, self.classes + 1), dtype=torch.long,
                            device=gt.device)  # Shape: [b, f*f, p*p, classes + 1]
        count.scatter_add_(dim=3, index=gt_filt, src=torch.ones_like(gt_filt))  # Shape: [b, f*f, p*p, classes + 1]
        labels = rearrange(count[..., 1:], "b (f1 f2) (p1 p2) c-> b c (f1 f2 p1 p2)", f1=f, f2=f, p1=p,
                           p2=p).float()  # Shape: [b, c, f*p*f*p]
        return labels, spix_map

    def forward(self, x, gt, feat, sdf_data):
        mask, center_feat = self.cluster_forward(x, feat, sdf_data)
        with torch.no_grad():
            labels, spix_map = self.gen_labels(gt, mask)
        return center_feat, labels, spix_map

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
