import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import config

# sin-cose embedding module
class Embedder(nn.Module):
    def __init__(self, **kwargs):
        super(Embedder, self).__init__()
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs["input_dims"]
        out_dim = 0
        if self.kwargs["include_input"]:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs["max_freq_log2"]
        N_freqs = self.kwargs["num_freqs"]

        if self.kwargs["log_sampling"]:
            freq_bands = 2.0 ** torch.linspace(0.0, max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.0**0.0, 2.0**max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs["periodic_fns"]:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def forward(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


class FeedForward(nn.Module):
    def __init__(self, dim, hid_dim, dp_rate):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, dim)
        self.dp = nn.Dropout(dp_rate)
        self.activ = nn.ReLU()

    def forward(self, x):
        x = self.dp(self.activ(self.fc1(x)))
        x = self.dp(self.fc2(x))
        return x


# Subtraction-based efficient attention
class Attention2D(nn.Module):
    def __init__(self, dim, dp_rate):
        super(Attention2D, self).__init__()
        self.q_fc = nn.Linear(dim, dim, bias=False)
        self.k_fc = nn.Linear(dim, dim, bias=False)
        self.v_fc = nn.Linear(dim, dim, bias=False)
        self.pos_fc = nn.Sequential(
            nn.Linear(4, dim // 8),
            nn.ReLU(),
            nn.Linear(dim // 8, dim),
        )
        self.attn_fc = nn.Sequential(
            nn.Linear(dim, dim // 8),
            nn.ReLU(),
            nn.Linear(dim // 8, dim),
        )
        self.out_fc = nn.Linear(dim, dim)
        self.dp = nn.Dropout(dp_rate)

    def forward(self, q, k, pos, mask=None):
        q = self.q_fc(q)
        k = self.k_fc(k)
        v = self.v_fc(k)

        pos = self.pos_fc(pos)
        attn = k - q[:, :, None, :] + pos
        attn = self.attn_fc(attn)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(attn, dim=-2)
        attn = self.dp(attn)

        x = ((v + pos) * attn).sum(dim=2)
        x = self.dp(self.out_fc(x))
        return x


# View Transformer
class Transformer2D(nn.Module):
    def __init__(self, dim, ff_hid_dim, ff_dp_rate, attn_dp_rate):
        super(Transformer2D, self).__init__()
        self.attn_norm = nn.LayerNorm(dim, eps=1e-6)
        self.ff_norm = nn.LayerNorm(dim, eps=1e-6)

        self.ff = FeedForward(dim, ff_hid_dim, ff_dp_rate)
        self.attn = Attention2D(dim, attn_dp_rate)

    def forward(self, q, k, pos, mask=None):
        residue = q
        x = self.attn_norm(q)
        x = self.attn(x, k, pos, mask)
        x = x + residue

        residue = x
        x = self.ff_norm(x)
        x = self.ff(x)
        x = x + residue

        return x


# attention module for self attention.
# contains several adaptations to incorportate positional information (NOT IN PAPER)
#   - qk (default) -> only (q.k) attention.
#   - pos -> replace (q.k) attention with position attention.
#   - gate -> weighted addition of  (q.k) attention and position attention.
class Attention(nn.Module):
    def __init__(self, dim, n_heads, dp_rate, attn_mode="qk", pos_dim=None):
        super(Attention, self).__init__()
        if attn_mode in ["qk", "gate"]:
            self.q_fc = nn.Linear(dim, dim, bias=False)
            self.k_fc = nn.Linear(dim, dim, bias=False)
        if attn_mode in ["pos", "gate"]:
            self.pos_fc = nn.Sequential(
                nn.Linear(pos_dim, pos_dim), nn.ReLU(), nn.Linear(pos_dim, dim // 8)
            )
            self.head_fc = nn.Linear(dim // 8, n_heads)
        if attn_mode == "gate":
            self.gate = nn.Parameter(torch.ones(n_heads))
        self.v_fc = nn.Linear(dim, dim, bias=False)
        self.out_fc = nn.Linear(dim, dim)
        self.dp = nn.Dropout(dp_rate)
        self.n_heads = n_heads
        self.attn_mode = attn_mode

    def forward(self, x, pos=None, ret_attn=False):
        if self.attn_mode in ["qk", "gate"]:
            q = self.q_fc(x)
            q = q.view(x.shape[0], x.shape[1], self.n_heads, -1).permute(0, 2, 1, 3)
            k = self.k_fc(x)
            k = k.view(x.shape[0], x.shape[1], self.n_heads, -1).permute(0, 2, 1, 3)
        v = self.v_fc(x)
        v = v.view(x.shape[0], x.shape[1], self.n_heads, -1).permute(0, 2, 1, 3)

        if self.attn_mode in ["qk", "gate"]:
            attn = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(q.shape[-1])
            attn = torch.softmax(attn, dim=-1)
        elif self.attn_mode == "pos":
            pos = self.pos_fc(pos)
            attn = self.head_fc(pos[:, :, None, :] - pos[:, None, :, :]).permute(0, 3, 1, 2)
            attn = torch.softmax(attn, dim=-1)
        if self.attn_mode == "gate":
            pos = self.pos_fc(pos)
            pos_attn = self.head_fc(pos[:, :, None, :] - pos[:, None, :, :]).permute(0, 3, 1, 2)
            pos_attn = torch.softmax(pos_attn, dim=-1)
            gate = self.gate.view(1, -1, 1, 1)
            attn = (1.0 - torch.sigmoid(gate)) * attn + torch.sigmoid(gate) * pos_attn
            attn /= attn.sum(dim=-1).unsqueeze(-1)
        attn = self.dp(attn)

        out = torch.matmul(attn, v).permute(0, 2, 1, 3).contiguous()
        out = out.view(x.shape[0], x.shape[1], -1)
        out = self.dp(self.out_fc(out))
        if ret_attn:
            return out, attn
        else:
            return out


# Ray Transformer
class Transformer(nn.Module):
    def __init__(
        self, dim, ff_hid_dim, ff_dp_rate, n_heads, attn_dp_rate, attn_mode="qk", pos_dim=None
    ):
        super(Transformer, self).__init__()
        self.attn_norm = nn.LayerNorm(dim, eps=1e-6)
        self.ff_norm = nn.LayerNorm(dim, eps=1e-6)

        self.ff = FeedForward(dim, ff_hid_dim, ff_dp_rate)
        self.attn = Attention(dim, n_heads, attn_dp_rate, attn_mode, pos_dim)

    def forward(self, x, pos=None, ret_attn=False):
        residue = x
        x = self.attn_norm(x)
        x = self.attn(x, pos, ret_attn)
        if ret_attn:
            x, attn = x
        x = x + residue

        residue = x
        x = self.ff_norm(x)
        x = self.ff(x)
        x = x + residue

        if ret_attn:
            return x, attn.mean(dim=1)[:, 0]
        else:
            return x


class GNT(nn.Module):
    def __init__(self, args, in_feat_ch=32, posenc_dim=3, viewenc_dim=3, ret_alpha=False):
        super(GNT, self).__init__()
        self.rgbfeat_fc = nn.Sequential(
            nn.Linear(in_feat_ch + 3, args.netwidth),
            nn.ReLU(),
            nn.Linear(args.netwidth, args.netwidth),
        )

        # NOTE: Apologies for the confusing naming scheme, here view_crosstrans refers to the view transformer, while the view_selftrans refers to the ray transformer
        self.view_selftrans = nn.ModuleList([]) # RAY
        self.view_crosstrans = nn.ModuleList([]) # VIEW
        self.q_fcs = nn.ModuleList([])
        for i in range(args.trans_depth):
            # view transformer
            view_trans = Transformer2D(
                dim=args.netwidth,
                ff_hid_dim=int(args.netwidth * 4),
                ff_dp_rate=0.1,
                attn_dp_rate=0.1,
            )
            if i%2==0:
                self.view_crosstrans.append(view_trans)
            else:
                self.view_crosstrans.append(view_trans)

            # ray transformer
            ray_trans = Transformer(
                dim=args.netwidth,
                ff_hid_dim=int(args.netwidth * 4),
                n_heads=4,
                ff_dp_rate=0.1,
                attn_dp_rate=0.1,
            )
            self.view_selftrans.append(ray_trans)
            

            # mlp
            if i % 2 == 0:
                q_fc = nn.Sequential(
                    nn.Linear(args.netwidth + posenc_dim + viewenc_dim, args.netwidth),
                    nn.ReLU(),
                    nn.Linear(args.netwidth, args.netwidth),
                )
            else:
                q_fc = nn.Identity()
            self.q_fcs.append(q_fc)

        self.posenc_dim = posenc_dim
        self.viewenc_dim = viewenc_dim
        self.ret_alpha = ret_alpha
        self.norm = nn.LayerNorm(args.netwidth)
        self.rgb_fc = nn.Linear(args.netwidth, 3)
        self.relu = nn.ReLU()
        self.pos_enc = Embedder(
            input_dims=3,
            include_input=True,
            max_freq_log2=9,
            num_freqs=10,
            log_sampling=True,
            periodic_fns=[torch.sin, torch.cos],
        )
        self.view_enc = Embedder(
            input_dims=3,
            include_input=True,
            max_freq_log2=9,
            num_freqs=10,
            log_sampling=True,
            periodic_fns=[torch.sin, torch.cos],
        )

    def forward(self, rgb_feat, ray_diff, mask, pts, ray_d):
        # compute positional embeddings
        viewdirs = ray_d
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1, 3]).float()
        viewdirs = self.view_enc(viewdirs)
        pts_ = torch.reshape(pts, [-1, pts.shape[-1]]).float()
        pts_ = self.pos_enc(pts_)
        pts_ = torch.reshape(pts_, list(pts.shape[:-1]) + [pts_.shape[-1]])
        viewdirs_ = viewdirs[:, None].expand(pts_.shape)
        embed = torch.cat([pts_, viewdirs_], dim=-1)
        input_pts, input_views = torch.split(embed, [self.posenc_dim, self.viewenc_dim], dim=-1)

        # project rgb features to netwidth
        rgb_feat = self.rgbfeat_fc(rgb_feat)
        rgb_feat = rgb_feat
        # q_init -> maxpool
        q = rgb_feat.max(dim=2)[0]

        # transformer modules
        for i, (crosstrans, q_fc, selftrans) in enumerate(
            zip(self.view_crosstrans, self.q_fcs, self.view_selftrans)
        ):
            # view transformer to update q
            q = crosstrans(q, rgb_feat, ray_diff, mask) if i%2==0 else crosstrans(q, rgb_feat, ray_diff, mask)
            # embed positional information
            if i % 2 == 0:
                q = torch.cat((q, input_pts, input_views), dim=-1)
                q = q_fc(q)
            # ray transformer
            #q = selftrans(q, ret_attn=self.ret_alpha)
            # Add Gaussian noise to q
            #noise_std = 0.2
            #q = q + torch.randn_like(q)
            q = selftrans(q, ret_attn=self.ret_alpha)
            # 'learned' density
            if self.ret_alpha:
                q, attn = q
        # normalize & rgb
        h = self.norm(q)
        outputs = self.rgb_fc(h.mean(dim=1))
        if self.ret_alpha:
            return torch.cat([outputs, attn], dim=1)
        else:
            return outputs
        


# # Weiner Filter
class Weiner(nn.Module):
    def __init__(self, psf):
        super(Weiner, self).__init__()
        self.psf = torch.nn.Parameter(psf.unsqueeze(0), requires_grad=True)

    
    def de_pad(self, f, im_h = 378, im_w = 504, p_h = 1518, p_w = 2012):

        depad_im = torch.zeros((im_h,im_w,3))
        ph,pw,iw,ih = int(p_h/2), int(p_w/2), int(im_w/2), int(im_h/2)
        depad_im[:,:,:] = f[ph-ih:ph+ih,pw-iw:pw+iw,:]
        return depad_im
    
    def pad(self,im,im_h,im_w,p_h=1518,p_w=2012):
        pad_im= torch.zeros((1,3,p_h,p_w))
        ph,pw,iw,ih = int(p_h/2), int(p_w/2), int(im_w/2), int(im_h/2)
        pad_im[:,:,ph-ih:ph+ih+1,pw-iw:pw+iw] = im
        return pad_im
    
    def wiener_deconvolution(self, x, psf,Gamma=0.000045):

        x_ = x.permute(0,2,1,3,4)
        psf_ = psf.permute(0,2,1,3,4)
        """
        Input:
        x: measurement of full size.
        psf: psf that was used to obtain measurement
        Gamma: regularization parameter
        Output:
        out: deconvolved image
        """
        for i, j in enumerate([3]):
            x = x_[:,:,i,:,:]
            psf = psf_[:,:,i,:,:]
            #print(x.shape,psf.shape)
            H = torch.fft.fft2(torch.fft.fftshift(psf,(2,3)))
            
            max_H = H.abs().reshape(H.abs().shape[0],H.abs().shape[1],-1).max(2)[0]
            
            max_H = max_H.unsqueeze(2).unsqueeze(3)
            
            H = H / (max_H + 1e-6)

            Habsq = H.abs()**2 
            
            W = torch.conj_physical(H)/(Habsq+Gamma)   

            X = torch.fft.fft2((x))
            max_X = X.abs().reshape(X.abs().shape[0],X.abs().shape[1],-1).max(2)[0]
            max_X = max_X.unsqueeze(2).unsqueeze(2)
            X = X / (max_X + 1e-6)

            outf = X*W

            out = F.relu((torch.fft.ifft2(outf)).real)#.squeeze()
            
            _, och, oh, ow = out.shape

            out = out.reshape(out.shape[0],-1)
            omax = out.max(1,keepdim=True)[0]
            out = out/(omax+1e-6)
            out = out.reshape(out.shape[0],och,oh,ow)

            f = F.relu(out)

            f = f.squeeze().permute(1,2,0)
            f = ((f-f.min())/(f.max()-f.min()))

            deconv_image = self.de_pad(f)
        
        return deconv_image


    def forward(self, input):

        result = []
        input = input.permute(0,1,4,2,3)
        for x in input[0]:
            x = x.unsqueeze(0)
            x =torch.tile(x,(2,1,1,1)).unsqueeze(0)
            #print(x.shape)
            psf = self.pad(self.psf,self.psf.shape[2],self.psf.shape[3],1518,2012).to("cuda")
            x = self.wiener_deconvolution(x,psf.unsqueeze(0))
            result.append(x)
        result = torch.stack(result,dim=0)

        return result






if __name__=="__main__":
    import cv2
    import imageio
    psf = cv2.imread("/data2/RBA_Intern/GNT2/trainable_psf/GNT2/data/trainpsf.png")
    psf = torch.from_numpy(psf).float().permute(2,0,1)
    print(psf.shape)

    # Code for cropping PSF
    # psf_ind = (psf[:,:] != [0,0,0]).nonzero()
    # print(psf_ind)
    # psf_h_start = psf_ind[0].min()
    # psf_h_stop = psf_ind[0].max()

    # psf_w_start = psf_ind[1].min()
    # psf_w_stop = psf_ind[1].max()

    # print(psf[psf_h_start:psf_h_stop,psf_w_start:psf_w_stop,:].shape)
    # cv2.imwrite("psf.png",psf[psf_h_start:psf_h_stop,psf_w_start:psf_w_stop,:])
    
    
    layer = Weiner(psf=psf).to("cuda")
    input = cv2.imread("/data2/RBA_Intern/GNT2/trainable_psf/GNT2/data/blurred/ibrnet_collected_1/howardzhou_003_stream/images_2/PXL_20201022_001956178.png")
    gt = cv2.imread("/data2/RBA_Intern/GNT2/trainable_psf/GNT2/data/refined_data/ibrnet_collected_1/howardzhou_003_stream/images_2/PXL_20201022_001956178.png")
    #input = input[600:1000,600:1000,:]
    input = torch.from_numpy(input).float().unsqueeze(0).unsqueeze(0)
    input = torch.nn.functional.pad(input,(0,0,(2012 - input.shape[-2])//2,(2012 - input.shape[-2])//2,(1518-input.shape[-3])//2,(1518 - input.shape[-3])//2))
    print(input.shape)
    out = layer(input.to("cuda"))
    print(out.shape)
    output = np.concatenate([out[0].detach().cpu().numpy()*255,gt],axis=1)
    print(output.shape)
    cv2.imwrite("output.png",output)#,out[0].detach().cpu().numpy()*255)