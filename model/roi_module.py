#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
from collections import namedtuple
from string import Template

import cupy, torch
import cupy as cp
import torch as t
from torch.autograd import Function

from model.utils.roi_cupy import kernel_backward, kernel_forward

Stream = namedtuple('Stream', ['ptr'])


@cupy.memoize(for_each_device=True)
def load_kernel(kernel_name, code, **kwargs):
    cp.cuda.runtime.free(0)
    code = Template(code).substitute(**kwargs)
    kernel_code = cupy.cuda.compile_with_cache(code)
    return kernel_code.get_function(kernel_name)


CUDA_NUM_THREADS = 1024


def GET_BLOCKS(N, K=CUDA_NUM_THREADS):
    return (N + K - 1) // K

class RoIPoolFunction(Function):
    def __init__(ctx, pooled_height, pooled_width, spatial_scale):
        ctx.pooled_width = pooled_width
        ctx.pooled_height = pooled_height
        ctx.spatial_scale = spatial_scale
        ctx.feature_size = None

    def forward(ctx, features, rois): 
        ctx.feature_size = features.size()           
        batch_size, num_channels, data_height, data_width = ctx.feature_size
        num_rois = rois.size(0)
        output = features.new(num_rois, num_channels, ctx.pooled_height, ctx.pooled_width).zero_()
        ctx.argmax = features.new(num_rois, num_channels, ctx.pooled_height, ctx.pooled_width).zero_().int()
        ctx.rois = rois
        if not features.is_cuda:
            _features = features.permute(0, 2, 3, 1)
            roi_pooling.roi_pooling_forward(ctx.pooled_height, ctx.pooled_width, ctx.spatial_scale,
                                            _features, rois, output)
        else:
            roi_pooling.roi_pooling_forward_cuda(ctx.pooled_height, ctx.pooled_width, ctx.spatial_scale,
                                                 features, rois, output, ctx.argmax)

        return output

    def backward(ctx, grad_output):
        assert(ctx.feature_size is not None and grad_output.is_cuda)
        batch_size, num_channels, data_height, data_width = ctx.feature_size
        grad_input = grad_output.new(batch_size, num_channels, data_height, data_width).zero_()

        roi_pooling.roi_pooling_backward_cuda(ctx.pooled_height, ctx.pooled_width, ctx.spatial_scale,
                                              grad_output, ctx.rois, grad_input, ctx.argmax)

        return grad_input, None
# class RoI(Function):
#     """
#     NOTE：only CUDA-compatible
#     """

#     # def __init__(self, outh, outw, spatial_scale):
#     #     self.forward_fn = load_kernel('roi_forward', kernel_forward)
#     #     self.backward_fn = load_kernel('roi_backward', kernel_backward)
#     #     self.outh, self.outw, self.spatial_scale = outh, outw, spatial_scale
    
#     @staticmethod
#     def forward(ctx, x, rois,outh, outw, spatial_scale):
#         # NOTE: MAKE SURE input is contiguous too
#         ctx.outh, ctx.outw,ctx.spatial_scale =  outh, outw, spatial_scale
#         ctx.forward_fn =  load_kernel('roi_forward', kernel_forward)
#         ctx.backward_fn = load_kernel('roi_backward', kernel_backward)
#         x = x.contiguous()
#         rois = rois.contiguous()
#         ctx.in_size = B, C, H, W = x.size()
#         ctx.N = N = rois.size(0)
#         output = t.zeros(N, C, ctx.outh, ctx.outw).cuda()
#         ctx.argmax_data = t.zeros(N, C, ctx.outh, ctx.outw).int().cuda()
#         ctx.rois = rois
#         args = [x.data_ptr(), rois.data_ptr(),
#                 output.data_ptr(),
#                 ctx.argmax_data.data_ptr(),
#                 ctx.spatial_scale, C, H, W,
#                 ctx.outh, ctx.outw,
#                 output.numel()]
#         stream = Stream(ptr=torch.cuda.current_stream().cuda_stream)
#         ctx.forward_fn(args=args,
#                         block=(CUDA_NUM_THREADS, 1, 1),
#                         grid=(GET_BLOCKS(output.numel()), 1, 1),
#                         stream=stream)
#         ctx.save_for_backward(output)
#         return output
#     @staticmethod
#     def backward(ctx, grad_output):
#         ##NOTE: IMPORTANT CONTIGUOUS
#         # TODO: input
#         output,= ctx.saved_tensors
#         grad_output = grad_output * output
#         grad_output = grad_output.contiguous()
#         B, C, H, W = ctx.in_size
#         grad_input = t.zeros(ctx.in_size).cuda()
#         stream = Stream(ptr=torch.cuda.current_stream().cuda_stream)
#         args = [grad_output.data_ptr(),
#                 ctx.argmax_data.data_ptr(),
#                 ctx.rois.data_ptr(),
#                 grad_input.data_ptr(),
#                 ctx.N, ctx.spatial_scale, C, H, W, ctx.outh, ctx.outw,
#                 grad_input.numel()]
#         ctx.backward_fn(args=args,
#                          block=(CUDA_NUM_THREADS, 1, 1),
#                          grid=(GET_BLOCKS(grad_input.numel()), 1, 1),
#                          stream=stream
#                          )
#         #grad_input = grad_input * output
#         return grad_output, None


class RoIPooling2D(t.nn.Module):

    def __init__(self, outh, outw, spatial_scale):
        super(RoIPooling2D, self).__init__()
        self.outh,self.outw,self.spatial_scale = outh, outw, spatial_scale
        #self.RoI = RoI()
        self.RoI = RoIPoolFunction(self.outh,self.outw,self.spatial_scale)

    def forward(self, x, rois):
        return self.RoI.apply(x, rois)
# class RoI(Function):
#     """
#     NOTE：only CUDA-compatible
#     """

#     def __init__(self, outh, outw, spatial_scale):
#         self.forward_fn = load_kernel('roi_forward', kernel_forward)
#         self.backward_fn = load_kernel('roi_backward', kernel_backward)
#         self.outh, self.outw, self.spatial_scale = outh, outw, spatial_scale
#     @staticmethod
#     def forward(self, x, rois):
#         # NOTE: MAKE SURE input is contiguous too
#         x = x.contiguous()
#         rois = rois.contiguous()
#         self.in_size = B, C, H, W = x.size()
#         self.N = N = rois.size(0)
#         output = t.zeros(N, C, self.outh, self.outw).cuda()
#         self.argmax_data = t.zeros(N, C, self.outh, self.outw).int().cuda()
#         self.rois = rois
#         args = [x.data_ptr(), rois.data_ptr(),
#                 output.data_ptr(),
#                 self.argmax_data.data_ptr(),
#                 self.spatial_scale, C, H, W,
#                 self.outh, self.outw,
#                 output.numel()]
#         stream = Stream(ptr=torch.cuda.current_stream().cuda_stream)
#         self.forward_fn(args=args,
#                         block=(CUDA_NUM_THREADS, 1, 1),
#                         grid=(GET_BLOCKS(output.numel()), 1, 1),
#                         stream=stream)
#         return output
#     @staticmethod
#     def backward(self, grad_output):
#         ##NOTE: IMPORTANT CONTIGUOUS
#         # TODO: input
#         grad_output = grad_output.contiguous()
#         B, C, H, W = self.in_size
#         grad_input = t.zeros(self.in_size).cuda()
#         stream = Stream(ptr=torch.cuda.current_stream().cuda_stream)
#         args = [grad_output.data_ptr(),
#                 self.argmax_data.data_ptr(),
#                 self.rois.data_ptr(),
#                 grad_input.data_ptr(),
#                 self.N, self.spatial_scale, C, H, W, self.outh, self.outw,
#                 grad_input.numel()]
#         self.backward_fn(args=args,
#                          block=(CUDA_NUM_THREADS, 1, 1),
#                          grid=(GET_BLOCKS(grad_input.numel()), 1, 1),
#                          stream=stream
#                          )
#         return grad_input, None


# class RoIPooling2D(t.nn.Module):

#     def __init__(self, outh, outw, spatial_scale):
#         super(RoIPooling2D, self).__init__()
#         self.RoI = RoI(outh, outw, spatial_scale)

#     def forward(self, x, rois):
#         return self.RoI.apply(x, rois)

def test_roi_module():
    ## fake data###
    B, N, C, H, W, PH, PW = 2, 8, 4, 32, 32, 7, 7

    bottom_data = t.randn(B, C, H, W).cuda()
    bottom_rois = t.randn(N, 5)
    bottom_rois[:int(N / 2), 0] = 0
    bottom_rois[int(N / 2):, 0] = 1
    bottom_rois[:, 1:] = (t.rand(N, 4) * 100).float()
    bottom_rois = bottom_rois.cuda()
    spatial_scale = 1. / 16
    outh, outw = PH, PW

    # pytorch version
    module = RoIPooling2D(outh, outw, spatial_scale)
    x = t.autograd.Variable(bottom_data, requires_grad=True)
    rois = t.autograd.Variable(bottom_rois)
    output = module(x, rois)
    output.sum().backward()

    def t2c(variable):
        npa = variable.data.cpu().numpy()
        return cp.array(npa)

    def test_eq(variable, array, info):
        cc = cp.asnumpy(array)
        neq = (cc != variable.data.cpu().numpy())
        assert neq.sum() == 0, 'test failed: %s' % info

    # chainer version,if you're going to run this
    # pip install chainer
    import chainer.functions as F
    from chainer import Variable
    x_cn = Variable(t2c(x))

    o_cn = F.roi_pooling_2d(x_cn, t2c(rois), outh, outw, spatial_scale)
    test_eq(output, o_cn.array, 'forward')
    F.sum(o_cn).backward()
    test_eq(x.grad, x_cn.grad, 'backward')
    print('test pass')
