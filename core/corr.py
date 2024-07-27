import torch
import torch.nn.functional as F
from core.utils.utils import bilinear_sampler

try:
    import corr_sampler
except:
    pass

try:
    import alt_cuda_corr
except:
    # alt_cuda_corr is not compiled
    pass


class CorrSampler(torch.autograd.Function):
    @staticmethod
    def forward(ctx, volume, coords, radius):
        ctx.save_for_backward(volume,coords)
        ctx.radius = radius
        corr, = corr_sampler.forward(volume, coords, radius)
        return corr
    @staticmethod
    def backward(ctx, grad_output):
        volume, coords = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_volume, = corr_sampler.backward(volume, coords, grad_output, ctx.radius)
        return grad_volume, None, None


class CorrBlock1D:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4, input_image_num=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []
        self.input_image_num = input_image_num
        # all pairs correlation
        corr = CorrBlock1D.corr(fmap1, fmap2, input_image_num)

        for j in range(len(corr)):
            batch, h, w, _, L = corr[j].shape
            corr_temp = corr[j].reshape(batch*h*w, 1, 1, L)
            self.corr_pyramid.append(corr_temp)
            for i in range(self.num_levels-1):
                corr_temp = F.avg_pool2d(corr_temp, [1,2], stride=[1,2])
                self.corr_pyramid.append(corr_temp)

    def __call__(self, coords,coords0):
        r = self.radius
        out_pyramid = []

        ##### j=0:Left, j=1:Right, j=2:Top, j=3:Bottom ########
        coords = coords[:, :1].permute(0, 2, 3, 1)
        coords0_tb = coords0[:, 1:].permute(0, 2, 3, 1)
        coords0 = coords0[:, :1].permute(0, 2, 3, 1)
        disp = coords-coords0

        batch, h1, w1, _ = coords.shape

        for j in range(int(len(self.corr_pyramid)/self.num_levels)):
            for i in range(self.num_levels):
                corr = self.corr_pyramid[j*self.num_levels+i]
                dx = torch.linspace(-r, r, 2*r+1)
                dx = dx.view(2*r+1, 1).to(coords.device)
                if j ==0 : 
                    x0 = dx + (coords0-disp).reshape(batch*h1*w1, 1, 1, 1) / 2**i
                elif j==1:
                    x0 = dx + coords.reshape(batch*h1*w1, 1, 1, 1) / 2**i
                elif j==2: 
                    x0 = dx + (coords0_tb-disp).reshape(batch*h1*w1, 1, 1, 1) / 2**i
                else:
                    x0 = dx + (coords0_tb+disp).reshape(batch*h1*w1, 1, 1, 1) / 2**i
                y0 = torch.zeros_like(x0)

                coords_lvl = torch.cat([x0,y0], dim=-1)
                corr = bilinear_sampler(corr, coords_lvl)
                corr = corr.view(batch, h1, w1, -1)

                ########### Flip Left and Top ################
                if j==0 or j==2: 
                    corr = torch.flip(corr, dims=[3])
                #################################

                out_pyramid.append(corr.permute(0, 3, 1, 2))

        out = torch.cat(out_pyramid, dim=1)

        return out.contiguous().float()
        

    @staticmethod
    def corr(fmap1, fmap2, input_image_num):
        ## if the 2 feature extractor is used for Left&Right and Top&Bottom.
        if len(fmap1.shape) == 5:
            B, S1, D, H1, W1 = fmap1.shape
            _, S, _, H2, W2 = fmap2.shape
            corr_list = []
            for s in range(2):
                fmap1_m = fmap1[:,0].permute(0, 2, 3, 1)
                fmap2_m = fmap2[:,s].permute(0, 2, 1, 3)
                corr = torch.matmul(fmap1_m, fmap2_m).unsqueeze(3).contiguous()
                corr_list.append(corr / torch.sqrt(torch.tensor(D).float()))
            
            for s in range(2,S):
                fmap1_m = fmap1[:,1].permute(0, 3, 2, 1)
                fmap2_m = fmap2[:,s].permute(0, 3, 1, 2)
                corr = torch.matmul(fmap1_m, fmap2_m).permute(0, 2, 1, 3).unsqueeze(3).contiguous()
                corr_list.append(corr / torch.sqrt(torch.tensor(D).float()))
            return corr_list
        
        B, D, H1, W1 = fmap1.shape
        _, S, _, H2, W2 = fmap2.shape
        corr_list = []
        for s in range(2):
            fmap1_m = fmap1.permute(0, 2, 3, 1)
            fmap2_m = fmap2[:,s].permute(0, 2, 1, 3)
            corr = torch.matmul(fmap1_m, fmap2_m).unsqueeze(3).contiguous()
            corr_list.append(corr / torch.sqrt(torch.tensor(D).float()))
        
        for s in range(2,S):
            fmap1_m = fmap1.permute(0, 3, 2, 1)
            fmap2_m = fmap2[:,s].permute(0, 3, 1, 2)
            corr = torch.matmul(fmap1_m, fmap2_m).permute(0, 2, 1, 3).unsqueeze(3).contiguous()
            corr_list.append(corr / torch.sqrt(torch.tensor(D).float()))
        return corr_list


# class CorrBlockFast1D:
#     def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
#         self.num_levels = num_levels
#         self.radius = radius
#         self.corr_pyramid = []
#         # all pairs correlation
#         corr = CorrBlockFast1D.corr(fmap1, fmap2)
#         batch, h1, w1, dim, w2 = corr.shape
#         corr = corr.reshape(batch*h1*w1, dim, 1, w2)
#         for i in range(self.num_levels):
#             self.corr_pyramid.append(corr.view(batch, h1, w1, -1, w2//2**i))
#             corr = F.avg_pool2d(corr, [1,2], stride=[1,2])

#     def __call__(self, coords):
#         out_pyramid = []
#         bz, _, ht, wd = coords.shape
#         coords = coords[:, [0]]
#         for i in range(self.num_levels):
#             corr = CorrSampler.apply(self.corr_pyramid[i].squeeze(3), coords/2**i, self.radius)
#             out_pyramid.append(corr.view(bz, -1, ht, wd))
#         return torch.cat(out_pyramid, dim=1)

#     @staticmethod
#     def corr(fmap1, fmap2):
#         B, D, H, W1 = fmap1.shape
#         _, _, _, W2 = fmap2.shape
#         fmap1 = fmap1.view(B, D, H, W1)
#         fmap2 = fmap2.view(B, D, H, W2)
#         corr = torch.einsum('aijk,aijh->ajkh', fmap1, fmap2)
#         corr = corr.reshape(B, H, W1, 1, W2).contiguous()
#         return corr / torch.sqrt(torch.tensor(D).float())


# class PytorchAlternateCorrBlock1D:
#     def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
#         self.num_levels = num_levels
#         self.radius = radius
#         self.corr_pyramid = []
#         self.fmap1 = fmap1
#         self.fmap2 = fmap2

#     def corr(self, fmap1, fmap2, coords):
#         B, D, H, W = fmap2.shape
#         # map grid coordinates to [-1,1]
#         xgrid, ygrid = coords.split([1,1], dim=-1)
#         xgrid = 2*xgrid/(W-1) - 1
#         ygrid = 2*ygrid/(H-1) - 1

#         grid = torch.cat([xgrid, ygrid], dim=-1)
#         output_corr = []
#         for grid_slice in grid.unbind(3):
#             fmapw_mini = F.grid_sample(fmap2, grid_slice, align_corners=True)
#             corr = torch.sum(fmapw_mini * fmap1, dim=1)
#             output_corr.append(corr)
#         corr = torch.stack(output_corr, dim=1).permute(0,2,3,1)

#         return corr / torch.sqrt(torch.tensor(D).float())

#     def __call__(self, coords):
#         r = self.radius
#         coords = coords.permute(0, 2, 3, 1)
#         batch, h1, w1, _ = coords.shape
#         fmap1 = self.fmap1
#         fmap2 = self.fmap2
#         out_pyramid = []
#         for i in range(self.num_levels):
#             dx = torch.zeros(1)
#             dy = torch.linspace(-r, r, 2*r+1)
#             delta = torch.stack(torch.meshgrid(dy, dx), axis=-1).to(coords.device)
#             centroid_lvl = coords.reshape(batch, h1, w1, 1, 2).clone()
#             centroid_lvl[...,0] = centroid_lvl[...,0] / 2**i
#             coords_lvl = centroid_lvl + delta.view(-1, 2)
#             corr = self.corr(fmap1, fmap2, coords_lvl)
#             fmap2 = F.avg_pool2d(fmap2, [1, 2], stride=[1, 2])
#             out_pyramid.append(corr)
#         out = torch.cat(out_pyramid, dim=-1)
#         return out.permute(0, 3, 1, 2).contiguous().float()


# class AlternateCorrBlock:
#     def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
#         raise NotImplementedError
#         self.num_levels = num_levels
#         self.radius = radius

#         self.pyramid = [(fmap1, fmap2)]
#         for i in range(self.num_levels):
#             fmap1 = F.avg_pool2d(fmap1, 2, stride=2)
#             fmap2 = F.avg_pool2d(fmap2, 2, stride=2)
#             self.pyramid.append((fmap1, fmap2))

#     def __call__(self, coords):
#         coords = coords.permute(0, 2, 3, 1)
#         B, H, W, _ = coords.shape
#         dim = self.pyramid[0][0].shape[1]

#         corr_list = []
#         for i in range(self.num_levels):
#             r = self.radius
#             fmap1_i = self.pyramid[0][0].permute(0, 2, 3, 1).contiguous()
#             fmap2_i = self.pyramid[i][1].permute(0, 2, 3, 1).contiguous()

#             coords_i = (coords / 2**i).reshape(B, 1, H, W, 2).contiguous()
#             corr, = alt_cuda_corr.forward(fmap1_i, fmap2_i, coords_i, r)
#             corr_list.append(corr.squeeze(1))

#         corr = torch.stack(corr_list, dim=1)
#         corr = corr.reshape(B, -1, H, W)
#         return corr / torch.sqrt(torch.tensor(dim).float())
