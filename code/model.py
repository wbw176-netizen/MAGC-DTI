import torch.nn as nn
import torch.nn.functional as F
import torch
from dgllife.model.gnn import GCN


def binary_cross_entropy(pred_output, labels):
    loss_fct = torch.nn.BCEWithLogitsLoss()
    n = torch.sigmoid(torch.squeeze(pred_output, 1))
    loss = loss_fct(torch.squeeze(pred_output, 1), labels)
    return n, loss

def cross_entropy_logits(linear_output, label, weights=None):
    if linear_output.size(1) < 2:
        p = torch.sigmoid(linear_output)
        linear_output = torch.cat([1-p, p], dim=1)
    
    class_output = F.log_softmax(linear_output, dim=1)
    n = F.softmax(linear_output, dim=1)[:, 1] 
    max_class = class_output.max(1)
    y_hat = max_class[1] 
    if weights is None:
        loss = nn.NLLLoss()(class_output, label.type_as(y_hat).view(label.size(0)))
    else:
        losses = nn.NLLLoss(reduction="none")(class_output, label.type_as(y_hat).view(label.size(0)))
        loss = torch.sum(weights * losses) / torch.sum(weights)
    return n, loss


def entropy_logits(linear_output):
    probs = F.softmax(linear_output, dim=1)
    log_probs = F.log_softmax(linear_output, dim=1)
    entropy = -torch.sum(probs * log_probs, dim=1)
    return entropy.mean()



class MSAFI(nn.Module):
   
    def __init__(self, in_planes, out_planes, head=4, reduction_ratio=16):
        super(MSAFI, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.head = head
        
        self.conv1x1 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv3x3 = nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1)
        self.conv5x5 = nn.Conv2d(in_planes, out_planes, kernel_size=5, padding=2)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(out_planes*3, out_planes*3 // reduction_ratio, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_planes*3 // reduction_ratio, out_planes*3, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.mha = nn.MultiheadAttention(out_planes, head, batch_first=True)
        
        self.norm = nn.LayerNorm(out_planes)

    def forward(self, x, mask=None):
        x1 = self.conv1x1(x)
        x3 = self.conv3x3(x)
        x5 = self.conv5x5(x)
        
        feat_cat = torch.cat([x1, x3, x5], dim=1)
        b, c, h, w = feat_cat.size()
        
        attention = self.fc(self.avg_pool(feat_cat))
        
        feat_refined = feat_cat * attention
        
        x1_att, x3_att, x5_att = torch.split(feat_refined, self.out_planes, dim=1)
        x_fused = x1_att + x3_att + x5_att

        if mask is not None:
            x_fused = x_fused * mask[:, None, None, :].to(x_fused.dtype)
        
        x_reshape = x_fused.permute(0, 2, 3, 1).reshape(b, h*w, self.out_planes)

        key_padding_mask = None
        if mask is not None:
            key_padding_mask = ~mask[:, None, :].expand(-1, h, -1).reshape(b, h * w)

        attn_out, _ = self.mha(
            x_reshape,
            x_reshape,
            x_reshape,
            key_padding_mask=key_padding_mask
        )
        
        out = self.norm(attn_out + x_reshape)

        if mask is not None:
            valid = mask[:, None, :].expand(-1, h, -1).reshape(b, h * w, 1)
            out = out * valid.to(out.dtype)

        out = out.view(b, h, w, self.out_planes).permute(0, 3, 1, 2)
        
        return out

class AGICA(nn.Module):
    def __init__(self, nf=128, num_heads=4, dropout=0.1, num_cascades=2):
        super(AGICA, self).__init__()

        self.nf = nf
        self.num_heads = num_heads
        self.head_dim = max(1, nf // num_heads)
        self.hidden_dim = self.head_dim * num_heads
        self.num_cascades = num_cascades

        self.drug_transform = nn.ModuleList()
        self.protein_transform = nn.ModuleList()
        self.fusion_modules = nn.ModuleList()

        for i in range(num_cascades):
            self.drug_transform.append(self._make_residual_block(nf))
            self.protein_transform.append(self._make_residual_block(nf))

            fusion = nn.ModuleDict({
                'drug_fusion': nn.Conv2d(nf * 2, nf, 1, 1, 0),
                'protein_fusion': nn.Conv2d(nf * 2, nf, 1, 1, 0),
                'shared_fusion': nn.Conv2d(nf * 2, nf, 1, 1, 0),

                'drug_norm': nn.BatchNorm2d(nf),
                'protein_norm': nn.BatchNorm2d(nf),
                'shared_norm': nn.BatchNorm2d(nf),

                'q_proj': nn.Conv2d(nf, self.hidden_dim, 1, 1, 0),
                'k_proj': nn.Conv2d(nf, self.hidden_dim, 1, 1, 0),
                'v_proj_drug': nn.Conv2d(nf, self.hidden_dim, 1, 1, 0),
                'v_proj_protein': nn.Conv2d(nf, self.hidden_dim, 1, 1, 0),

                'out_proj_drug': nn.Conv2d(self.hidden_dim, nf, 1, 1, 0),
                'out_proj_protein': nn.Conv2d(self.hidden_dim, nf, 1, 1, 0),

                'gate_drug': nn.Sequential(
                    nn.Conv2d(nf * 2, nf, 1, 1, 0),
                    nn.Sigmoid()
                ),
                'gate_protein': nn.Sequential(
                    nn.Conv2d(nf * 2, nf, 1, 1, 0),
                    nn.Sigmoid()
                ),

                'ffn_drug': nn.Sequential(
                    nn.Conv2d(nf, nf * 4, 1),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Conv2d(nf * 4, nf, 1)
                ),
                'ffn_protein': nn.Sequential(
                    nn.Conv2d(nf, nf * 4, 1),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Conv2d(nf * 4, nf, 1)
                ),

                'pyramid_drug': nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(nf, nf, 1),
                    nn.GELU()
                ),
                'pyramid_protein': nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(nf, nf, 1),
                    nn.GELU()
                )
            })
            self.fusion_modules.append(fusion)

        self.shared_init = nn.Sequential(
            nn.Conv2d(nf * 2, nf, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(nf, nf, 1, 1, 0)
        )

        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _make_residual_block(self, nf):
        return nn.Sequential(
            nn.Conv2d(nf, nf, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(nf, nf, 3, 1, 1)
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def _apply_sequence_mask(x, mask):
        if mask is None:
            return x
        return x * mask[:, None, :, None].to(x.dtype)

    @staticmethod
    def _masked_average_pool(x, mask):
        if mask is None:
            return x.mean(dim=(2, 3), keepdim=True)
        weights = mask[:, None, :, None].to(x.dtype)
        denominator = weights.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
        return (x * weights).sum(dim=(2, 3), keepdim=True) / denominator

    def _apply_attention(
        self,
        q,
        k,
        v,
        query_mask=None,
        key_mask=None
    ):
        b, c, q_h, q_w = q.shape
        _, _, k_h, k_w = k.shape
        q_len = q_h * q_w
        k_len = k_h * k_w

        q = q.view(b, self.num_heads, self.head_dim, q_len).permute(0, 1, 3, 2)
        k = k.view(b, self.num_heads, self.head_dim, k_len).permute(0, 1, 3, 2)
        v = v.view(b, self.num_heads, self.head_dim, k_len).permute(0, 1, 3, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim ** -0.5)

        if key_mask is not None:
            attn = attn.masked_fill(
                ~key_mask[:, None, None, :],
                torch.finfo(attn.dtype).min
            )

        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)

        out = out.permute(0, 1, 3, 2).contiguous().view(b, c, q_h, q_w)
        out = self._apply_sequence_mask(out, query_mask)

        return out

    def forward(self, x_drug, x_protein, drug_mask=None, protein_mask=None):

        x_drug_res = x_drug
        x_protein_res = x_protein

        drug_context = self._masked_average_pool(x_drug, drug_mask)
        protein_context = self._masked_average_pool(x_protein, protein_mask)
        x_shared = self.shared_init(
            torch.cat([drug_context, protein_context], dim=1)
        )

        for i in range(self.num_cascades):
            drug_block = self.drug_transform[i]
            protein_block = self.protein_transform[i]
            fusion = self.fusion_modules[i]

            drug_feat = self._apply_sequence_mask(
                drug_block(x_drug),
                drug_mask
            )
            protein_feat = self._apply_sequence_mask(
                protein_block(x_protein),
                protein_mask
            )

            drug_len = drug_feat.shape[2]
            protein_len = protein_feat.shape[2]
            protein_context = self._masked_average_pool(
                protein_feat,
                protein_mask
            ).expand(-1, -1, drug_len, -1)
            drug_context = self._masked_average_pool(
                drug_feat,
                drug_mask
            ).expand(-1, -1, protein_len, -1)

            drug_center = fusion['drug_norm'](fusion['drug_fusion'](
                torch.cat([
                    x_shared.expand(-1, -1, drug_len, -1),
                    protein_context
                ], dim=1)))
            protein_center = fusion['protein_norm'](fusion['protein_fusion'](
                torch.cat([
                    x_shared.expand(-1, -1, protein_len, -1),
                    drug_context
                ], dim=1)))

            q_drug = fusion['q_proj'](drug_center)
            q_protein = fusion['q_proj'](protein_center)

            k_drug = fusion['k_proj'](drug_feat)
            k_protein = fusion['k_proj'](protein_feat)

            v_drug = fusion['v_proj_drug'](drug_feat)
            v_protein = fusion['v_proj_protein'](protein_feat)

            drug_attn = self._apply_attention(
                q_drug,
                k_protein,
                v_protein,
                query_mask=drug_mask,
                key_mask=protein_mask
            )
            protein_attn = self._apply_attention(
                q_protein,
                k_drug,
                v_drug,
                query_mask=protein_mask,
                key_mask=drug_mask
            )

            drug_out = fusion['out_proj_drug'](drug_attn)
            protein_out = fusion['out_proj_protein'](protein_attn)

            drug_gate = fusion['gate_drug'](
                torch.cat([drug_feat, drug_out], dim=1)
            )
            protein_gate = fusion['gate_protein'](
                torch.cat([protein_feat, protein_out], dim=1)
            )

            drug_gated = drug_feat * (1 - drug_gate) + drug_out * drug_gate
            protein_gated = protein_feat * (1 - protein_gate) + protein_out * protein_gate

            drug_ffn = self._apply_sequence_mask(
                fusion['ffn_drug'](drug_gated) + drug_gated,
                drug_mask
            )
            protein_ffn = self._apply_sequence_mask(
                fusion['ffn_protein'](protein_gated) + protein_gated,
                protein_mask
            )

            drug_global = fusion['pyramid_drug'](
                self._masked_average_pool(drug_ffn, drug_mask)
            )
            protein_global = fusion['pyramid_protein'](
                self._masked_average_pool(protein_ffn, protein_mask)
            )

            x_drug = self._apply_sequence_mask(
                drug_ffn + drug_global + x_drug,
                drug_mask
            )
            x_protein = self._apply_sequence_mask(
                protein_ffn + protein_global + x_protein,
                protein_mask
            )

            x_shared = fusion['shared_norm'](fusion['shared_fusion'](
                torch.cat([drug_global, protein_global], dim=1))) + x_shared

        x_drug = self._apply_sequence_mask(
            x_drug + x_drug_res,
            drug_mask
        )
        x_protein = self._apply_sequence_mask(
            x_protein + x_protein_res,
            protein_mask
        )

        return x_drug, x_protein, x_shared

class MolecularGCN(nn.Module):
    def __init__(self, in_feats, dim_embedding=128, padding=True, hidden_feats=None, activation=None):
        super(MolecularGCN, self).__init__()
        if hidden_feats is None:
            hidden_feats = [dim_embedding]
        self.init_transform = nn.Linear(in_feats, dim_embedding, bias=False)
        self.padding = padding
        if padding:
            with torch.no_grad():
                self.init_transform.weight[:, -1].fill_(0)
        if activation is None:
            activation = [F.relu for _ in hidden_feats]
        elif callable(activation):
            activation = [activation for _ in hidden_feats]
        self.gnn = GCN(in_feats=dim_embedding, 
                      hidden_feats=hidden_feats, 
                      activation=activation,
                      allow_zero_in_degree=True) 
        self.output_feats = hidden_feats[-1]

    def forward(self, batch_graph):
        node_feats = batch_graph.ndata.pop('h')
        if self.padding:
            node_mask = node_feats[:, -1] == 0
        else:
            node_mask = torch.ones(
                node_feats.shape[0],
                dtype=torch.bool,
                device=node_feats.device
            )
        node_feats = self.init_transform(node_feats)
        node_feats = self.gnn(batch_graph, node_feats)
        batch_size = batch_graph.batch_size
        
        batch_num_nodes = batch_graph.batch_num_nodes()
        max_nodes = max(batch_num_nodes).item()
        
        output = torch.zeros(batch_size, max_nodes, self.output_feats, device=node_feats.device)
        output_mask = torch.zeros(
            batch_size,
            max_nodes,
            dtype=torch.bool,
            device=node_feats.device
        )
        
        start_idx = 0
        for i in range(batch_size):
            num_nodes = batch_num_nodes[i].item()
            output[i, :num_nodes, :] = node_feats[start_idx:start_idx+num_nodes, :]
            output_mask[i, :num_nodes] = node_mask[start_idx:start_idx+num_nodes]
            start_idx += num_nodes

        output = output * output_mask.unsqueeze(-1).to(output.dtype)
        return output, output_mask

class MSAA(nn.Module):
    def __init__(self, embedding_dim, num_filters, num_head, padding=True):
        super(MSAA, self).__init__()
        if padding:
            self.embedding = nn.Embedding(26, embedding_dim, padding_idx=0)
        else:
            self.embedding = nn.Embedding(26, embedding_dim)
        in_ch = [embedding_dim] + num_filters
        self.in_ch = in_ch[-1]

        self.msafi = MSAFI(in_planes=in_ch[0], out_planes=in_ch[1], head=num_head)
        
        self.feature_align = nn.Sequential(
            nn.Conv2d(in_ch[1], in_ch[1], 1),
            nn.BatchNorm2d(in_ch[1]),
            nn.ReLU()
        )

    def forward(self, v):
        mask = v != 0
        v = self.embedding(v.long())  # [B, seq_len, embedding_dim]
        v = v.transpose(2, 1)  # [B, embedding_dim, seq_len]
        
        v = self.msafi(v.unsqueeze(-2), mask=mask).squeeze(-2)
        v = v * mask.unsqueeze(1).to(v.dtype)
        
        v = v.unsqueeze(-1)  # [B, embedding_dim, seq_len, 1]
        v = self.feature_align(v)
        v = v * mask[:, None, :, None].to(v.dtype)
        
        return v, mask

class MPRC(nn.Module):

    def __init__(self, in_dim, hidden_dim, out_dim, binary=1, dropout_rate=None):
        super(MPRC, self).__init__()
        self.proj1 = nn.Linear(in_dim, in_dim)
        self.bn1 = nn.BatchNorm1d(in_dim)


        self.proj2 = nn.Linear(in_dim, in_dim) 
        self.bn2 = nn.BatchNorm1d(in_dim)
        self.fuse = nn.Linear(in_dim * 3, hidden_dim)
        self.bn_fuse = nn.BatchNorm1d(hidden_dim)

        self.hidden_fc = nn.Linear(hidden_dim, out_dim)
        self.bn_hidden = nn.BatchNorm1d(out_dim)
        self.output_fc = nn.Linear(out_dim, binary)

        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()

    def forward(self, x):
        x_identity = x

        x_axis1 = self.relu(self.bn1(self.proj1(x)))

        x_axis2 = self.relu(self.bn2(self.proj2(x)))

        x_concat = torch.cat([x_axis1, x_axis2, x_identity], dim=1)
        x_fused = self.dropout(self.relu(self.bn_fuse(self.fuse(x_concat))))

        x_hidden = self.dropout(self.relu(self.bn_hidden(self.hidden_fc(x_fused))))
        output = self.output_fc(x_hidden)

        return output

class BiEncoderFeatureExtractor(nn.Module):
    
    def __init__(self, drug_dim, protein_dim, out_dim, dropout=0.1):
        super(BiEncoderFeatureExtractor, self).__init__()
        
        self.drug_encoder = nn.Sequential(
            nn.Linear(drug_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim)
        )
        
        self.protein_encoder = nn.Sequential(
            nn.Linear(protein_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim)
        )
        
        self.out_dim = out_dim
    
    @staticmethod
    def _encode(encoder, features):
        if features.dim() == 2:
            mask = torch.ones(
                features.shape[0],
                1,
                dtype=torch.bool,
                device=features.device
            )
            encoded = encoder(features).unsqueeze(1)
        elif features.dim() == 3:
            mask = features.abs().sum(dim=-1) != 0
            encoded = encoder(features)
        else:
            raise ValueError(
                "Pre-extracted features must have shape [B, D] or [B, L, D]"
            )
        encoded = encoded * mask.unsqueeze(-1).to(encoded.dtype)
        return encoded.transpose(1, 2).unsqueeze(-1), mask

    def forward(self, drug_feat, protein_feat):
        drug_out, drug_mask = self._encode(self.drug_encoder, drug_feat)
        protein_out, protein_mask = self._encode(
            self.protein_encoder,
            protein_feat
        )

        return drug_out, protein_out, drug_mask, protein_mask


class MAGC_DTI(nn.Module):
    def __init__(self, device='cuda', **config):
        super(MAGC_DTI, self).__init__()
        drug_in_feats = config["DRUG"]["NODE_IN_FEATS"]
        drug_embedding = config["DRUG"]["NODE_IN_EMBEDDING"]
        drug_hidden_feats = config["DRUG"]["HIDDEN_LAYERS"]
        protein_emb_dim = config["PROTEIN"]["EMBEDDING_DIM"]
        num_filters = config["PROTEIN"]["NUM_FILTERS"]
        mlp_in_dim = config["DECODER"]["IN_DIM"]
        mlp_hidden_dim = config["DECODER"]["HIDDEN_DIM"]
        mlp_out_dim = config["DECODER"]["OUT_DIM"]
        mlp_dropout_rate = config["DECODER"]["DROPOUT_RATE"]
        drug_padding = config["DRUG"]["PADDING"]
        protein_padding = config["PROTEIN"]["PADDING"]
        out_binary = config["DECODER"]["BINARY"]
        protein_num_head = config['PROTEIN']['NUM_HEAD']
        cross_num_head = config["AGICA_CROSS_ATTENTION"]['NUM_HEAD']
        cross_emb_dim = config['AGICA_CROSS_ATTENTION']['EMBEDDING_DIM']
        cross_dropout_rate = config['AGICA_CROSS_ATTENTION']['AGICA_DROPOUT_RATE']
        
        self.use_pretrained = config["PRETRAINED"]["USE_ESM2"] or config["PRETRAINED"]["USE_CHEMBERT"]
        pretrained_out_dim = cross_emb_dim  
        
        if self.use_pretrained:
            drug_pretrained_dim = config["PRETRAINED"]["CHEMBERT_DIM"] if config["PRETRAINED"]["USE_CHEMBERT"] else drug_embedding
            protein_pretrained_dim = config["PRETRAINED"]["ESM2_DIM"] if config["PRETRAINED"]["USE_ESM2"] else protein_emb_dim
            
            self.feature_extractor = BiEncoderFeatureExtractor(
                drug_dim=drug_pretrained_dim,
                protein_dim=protein_pretrained_dim,
                out_dim=pretrained_out_dim,
                dropout=cross_dropout_rate
            )
        else:
            self.drug_extractor = MolecularGCN(in_feats=drug_in_feats, dim_embedding=drug_embedding, padding=drug_padding, hidden_feats=drug_hidden_feats)
            self.protein_extractor = MSAA(protein_emb_dim, num_filters, protein_num_head, protein_padding)
            self.feature_extractor = None

        self.agica = AGICA(nf=cross_emb_dim, num_heads=cross_num_head, dropout=cross_dropout_rate, num_cascades=1)

        self.fusion_weights = nn.Parameter(torch.ones(3) / 3)

        self.joint_fusion = nn.Linear(cross_emb_dim * 3, mlp_in_dim)
        self.fusion_norm = nn.LayerNorm(mlp_in_dim)

        self.mlp_classifier = MPRC(mlp_in_dim, mlp_hidden_dim, mlp_out_dim, dropout_rate=mlp_dropout_rate)

    def forward(self, bg_d, v_p, mode="train"):
        if self.use_pretrained and not hasattr(bg_d, 'ndata'):
            v_d, v_p, drug_mask, protein_mask = self.feature_extractor(
                bg_d,
                v_p
            )
        else:
            v_d, drug_mask = self.drug_extractor(bg_d)
            v_p, protein_mask = self.protein_extractor(v_p)
            v_d = v_d.transpose(1, 2).unsqueeze(-1)

        v_d_enhanced, v_p_enhanced, x_s_enhanced = self.agica(
            v_d,
            v_p,
            drug_mask=drug_mask,
            protein_mask=protein_mask
        )

        v_d_pooled = self._masked_max_pool(v_d_enhanced, drug_mask)
        v_p_pooled = self._masked_max_pool(v_p_enhanced, protein_mask)
        x_s_pooled = x_s_enhanced.squeeze(-1).squeeze(-1)
        

        weights = F.softmax(self.fusion_weights, dim=0)
        f = torch.cat([
            v_d_pooled * weights[0],
            v_p_pooled * weights[1],
            x_s_pooled * weights[2]
        ], dim=1) 
        
        f = self.fusion_norm(self.joint_fusion(f))
        
        score = self.mlp_classifier(f)
        
        if mode == "train":
            return v_d_enhanced, v_p_enhanced, f, score
        elif mode == "eval":
            return v_d_enhanced, v_p_enhanced, score, None

    @staticmethod
    def _masked_max_pool(features, mask):
        values = features.squeeze(-1)
        masked_values = values.masked_fill(
            ~mask[:, None, :],
            torch.finfo(values.dtype).min
        )
        pooled = masked_values.max(dim=2).values
        has_valid_value = mask.any(dim=1, keepdim=True)
        return torch.where(has_valid_value, pooled, torch.zeros_like(pooled))
