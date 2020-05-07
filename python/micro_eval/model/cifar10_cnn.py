import collections
import json
import numpy as np

import tvm
from tvm import relay

from micro_eval import util


CIFAR10_CLASSES = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']


# Generated random params, keyed by (data_layout, kernel_layouts).
GENERATED_RANDOM_PARAMS = {}


_RANDOM_BOUNDS = {
    'mean_data': (130, 140),
    'conv0_weight': (-30, 30),
    'conv0_bias': (-3, 3),
    'conv1_weight': (-30, 30),
    'conv1_bias': (-3, 3),
    'conv2_weight': (-30, 30),
    'conv2_bias': (-3, 3),
    'dense0_weight': (-30, 30),
    'dense0_bias': (-3, 3),
}


def _gen_random_params(mod, param_shapes):
    _cache_key = (tuple(k, v.shape) for k in sorted(param_shapes.keys()))
    if _cache_key not in GENERATED_RANDOM_PARAMS:
        # generate random params
        params = {}
        for param in mod['main'].params[1:]:
            name = param.name_hint
            low, high = _RANDOM_BOUNDS[name]
            rand_tensor = param_shapes[name].gen_rand_tensor(low, high)
            params[param.name_hint] = tvm.nd.array(rand_tensor)

        GENERATED_RANDOM_PARAMS[_cache_key] = params

    return GENERATED_RANDOM_PARAMS[_cache_key]


_CMSIS_PARAM_SHAPES = {
    'mean_data': util.LabelledShape(N=1, H=32, W=32, C=3, dtype='uint8'),
    'conv0_weight': util.LabelledShape(O=32, H=5, W=5, I=3, dtype='int8'),
    'conv0_bias': util.LabelledShape(B=32, dtype='int8'),
    'conv1_weight': util.LabelledShape(O=32, H=5, W=5, I=32, dtype='int8'),
    'conv1_bias': util.LabelledShape(B=32, dtype='int8'),
    'conv2_weight': util.LabelledShape(O=64, H=5, W=5, I=32, dtype='int8'),
    'conv2_bias': util.LabelledShape(B=64, dtype='int8'),
    'dense0_weight': util.LabelledShape(O=10, I=1024, dtype='int8'),
    'dense0_bias': util.LabelledShape(B=10, dtype='int8'),
}


def _load_cmsis_params(mod, param_shapes):
    with open(f'{util.get_repo_root()}/data/cifar10_cnn_params.json') as f:
        cmsis_params = json.load(f)

    params = {}
    for formal_param in mod['main'].params[1:]:  # exclude data
        name = formal_param.name_hint
        print('name', name, _CMSIS_PARAM_SHAPES[name].dtype, len(cmsis_params[name]))
        cmsis_tensor = util.LabelledTensor(
            data=np.array(cmsis_params[name], dtype=_CMSIS_PARAM_SHAPES[name].dtype, copy=True).reshape(_CMSIS_PARAM_SHAPES[name].shape),
            shape=_CMSIS_PARAM_SHAPES[name])

        param_shape = param_shapes[name]
        relay_shape = util.LabelledShape(
            zip(param_shape.layout, [x.value for x in formal_param.checked_type.shape]),
            dtype=param_shape.dtype)

        assert param_shape.dims == relay_shape.dims
        param = cmsis_tensor.resize(param_shape)
        params[name] = tvm.nd.array(param.data, tvm.cpu(0))

    return params


def gen_cifar10_cnn(data_layout, kernel_layouts, op_strategy='direct', use_random_params=False):
    # kernel layouts are specified per conv, but if only a single layout is
    # passed, that layout is used for all convs
    if isinstance(kernel_layouts, str):
        kernel_layouts = [kernel_layouts] * 3
    # TODO change relay/op/tensor/unary.cc _make.clip to accept exprs instead of doubles
    # TODO discrepancies between outputs might be a result of the bias_add op
    # not matching the semantics of the CMSIS bias add.

    data_shape = util.LabelledShape.from_dims_and_layout(dict(N=1, C=3, H=32, W=32), data_layout, dtype='uint8')
    conv0_weight_shape = util.LabelledShape.from_dims_and_layout(dict(H=5, W=5, I=3, O=32), kernel_layouts[0], dtype='int8')
    if op_strategy in ('direct_simd', 'partial_im2col'):
        # to fit our SIMD intrinsic, we make the 'C' dimension a multiple of 4
        data_shape = util.LabelledShape.from_dims_and_layout(dict(N=1, C=4, H=32, W=32), data_layout, dtype='uint8')
        conv0_weight_shape = util.LabelledShape.from_dims_and_layout(dict(H=5, W=5, O=32, I=4), kernel_layouts[0], dtype='int8')
    print('data_shape', data_shape)
    print('conv0_weight_shape', conv0_weight_shape)

    param_shapes = collections.OrderedDict([
        ('data', data_shape),
        ('mean_data', data_shape),
        ('conv0_weight', conv0_weight_shape),
        ('conv0_bias', util.LabelledShape(B=32, dtype='int8')),
        ('conv1_weight', util.LabelledShape.from_dims_and_layout(dict(O=32, I=32, H=5, W=5), kernel_layouts[1], dtype='int8')),
        ('conv1_bias', util.LabelledShape(B=32, dtype='int8')),
        ('conv2_weight', util.LabelledShape.from_dims_and_layout(dict(O=64, I=32, H=5, W=5), kernel_layouts[2], dtype='int8')),
        ('conv2_bias', util.LabelledShape(B=64, dtype='int8')),
        ('dense0_weight', util.LabelledShape(O=10, I=1024, dtype='int8')),
        ('dense0_bias', util.LabelledShape(B=10, dtype='int8')),
    ])
    bias_add_axis = data_layout.index('C')
    params = []
    for p, s in param_shapes.items():
        joined_shape = ', '.join(str(x) for x in s.shape)
        params.append(f'        %{p}: Tensor[({joined_shape}), {s.dtype}]')
    param_args = ',\n'.join(params)
    print('params', param_args)
    mod = relay.fromtext(f"""
    v0.0.4
    def @main({param_args}) {{
        %0 = cast(cast(%data, "int16") - cast(%mean_data, "int16"), "int8");
        %1 = nn.conv2d(
             %0,
             %conv0_weight,
             padding=[2, 2],
             channels=32,
             kernel_size=[5, 5],
             data_layout="{data_layout}",
             kernel_layout="{kernel_layouts[0]}",
             out_dtype="int32");
      %2 = nn.bias_add(%1, cast(%conv0_bias, "int32"), axis={bias_add_axis});
      %3 = right_shift(%2, 9);
      %4 = cast(%3, "int8");
      %5 = nn.max_pool2d(%4,
             pool_size=[3, 3],
             strides=[2, 2],
             layout="{data_layout}",
             ceil_mode=True);
      %6 = nn.relu(%5);
      %7 = nn.conv2d(
             %6,
             %conv1_weight,
             padding=[2, 2],
             channels=32,
             kernel_size=[5, 5],
             data_layout="{data_layout}",
             kernel_layout="{kernel_layouts[1]}",
             out_dtype="int32");
      %8 = nn.bias_add(%7, cast(%conv1_bias, "int32"), axis={bias_add_axis});
      %9 = right_shift(%8, 9);
      %10 = cast(%9, "int8");
      %11 = nn.relu(%10);
      %12 = nn.avg_pool2d(cast(%11, "int32"),
              pool_size=[3, 3],
              strides=[2, 2],
              count_include_pad=True,
              layout="{data_layout}",
              ceil_mode=True);
      %13 = nn.conv2d(cast(%12, "int8"),
              %conv2_weight,
              padding=[2, 2],
              channels=64,
              kernel_size=[5, 5],
              data_layout="{data_layout}",
              kernel_layout="{kernel_layouts[2]}",
              out_dtype="int32");
      %14 = nn.bias_add(%13, cast(%conv2_bias, "int32"), axis={bias_add_axis});
      %15 = right_shift(%14, 9);
      %16 = cast(%15, "int8");
      %17 = nn.relu(%16);
      %18 = nn.avg_pool2d(cast(%17, "int32"),
              pool_size=[3, 3],
              strides=[2, 2],
              count_include_pad=True,
              layout="{data_layout}",
              ceil_mode=True);
      %19 = nn.batch_flatten(%18);
      %20 = nn.dense(%19, %dense0_weight, units=10, out_dtype="int32");
      %21 = nn.bias_add(%20, left_shift(cast(%dense0_bias, "int32"), 3), axis=-1);
      %22 = right_shift(%21, 5);
      cast(%22, "int8")
    }}
    """)
    print('mod', mod.astext())
    if use_random_params:
        params = _gen_random_params(mod, data_layout, kernel_layouts)
    else:
        params = _load_cmsis_params(mod, param_shapes)

    return mod, params
