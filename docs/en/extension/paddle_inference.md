# Inference Framework

## Introducation

There are varieties of ways to save a model in PaddlePaddle.

1. persisitable model saved by fluid.save_persistabels
    It is always used for finetuning and resuming training. It contains params files and every file represents a variable, structure information is not included though. The architecture is necessary when loading persistable model.

    ```
    resnet50-vd-persistable/
    ├── bn2a_branch1_mean
    ├── bn2a_branch1_offset
    ├── bn2a_branch1_scale
    ├── bn2a_branch1_variance
    ├── bn2a_branch2a_mean
    ├── bn2a_branch2a_offset
    ├── bn2a_branch2a_scale
    ├── ...
    └── res5c_branch2c_weights
    ```
2. inference model saved by fluid.io.save_inference_model
    It is always used to inference and deploy, the difference is inference model contains model structure information which is stored in the `model` metioned in the below  .
    ```
    resnet50-vd-persistable/
    ├── bn2a_branch1_mean
    ├── bn2a_branch1_offset
    ├── bn2a_branch1_scale
    ├── bn2a_branch1_variance
    ├── bn2a_branch2a_mean
    ├── bn2a_branch2a_offset
    ├── bn2a_branch2a_scale
    ├── ...
    ├── res5c_branch2c_weights
    └── model
    ```
   paddle can combine all params file `params`
    ```
    resnet50-vd
    ├── model
    └── params
    ```

There are two kinds of engines, inference engine and training engine. both of them can predict, the difference is inference engine doesn't do back propagation, and it can fuse some operators and optimize memory to improve efficiency. PaddleClas introduces these three ways to predict.

1. inference engine load inference model
2. training engine load persistable model
3. training engine load inference model

and follow these steps:
+ build engine
+ organize data
+ execute predicator
+ analysis result


## Export model

Export inference model from pretrained model or checkpoint, it will feed in the inference engine.

```python
import fluid

from ppcls.modeling.architectures.resnet_vd import ResNet50_vd

place = fluid.CPUPlace()
exe = fluid.Executor(place)
startup_prog = fluid.Program()
infer_prog = fluid.Program()
with fluid.program_guard(infer_prog, startup_prog):
    with fluid.unique_name.guard():
        image = create_input()
        image = fluid.data(name='image', shape=[None, 3, 224, 224], dtype='float32')
        out = ResNet50_vd.net(input=input, class_dim=1000)

infer_prog = infer_prog.clone(for_test=True)
fluid.load(program=infer_prog, model_path=persistable model path, executor=exe)

fluid.io.save_inference_model(
        dirname='./output/',
        feeded_var_names=[image.name],
        main_program=infer_prog,
        target_vars=out,
        executor=exe,
        model_filename='model',
        params_filename='params')
```

Run the `tools/export_model.py` will export model：

```python
python tools/export_model.py \
    --m=model name \
    --p=persistable model path \
    --o=model and params save path
```

## inference engine load inference model

Run the `tools/infer/predict.py`

```
python ./predict.py \
    -i=./test.jpeg \
    -m=./resnet50-vd/model \
    -p=./resnet50-vd/params \
    --use_gpu=1 \
    --use_tensorrt=True
```

Attributes：
+ `image_file`(i): picture path `./test.jpeg`
+ `model_file`(m): model path `./resnet50-vd/model`
+ `params_file`(p): params path `./resnet50-vd/params`
+ `batch_size`(b): batch size
+ `ir_optim`: optimize `IR` or not, default: True
+ `use_tensorrt`: use TesorRT engine or not, default: True
+ `gpu_mem`: initial GPU size(M)
+ `use_gpu`: use gpu or not, default: True
+ `enable_benchmark`: switch benchmark or not, default: False
+ `model_name`: model name

Note:
if switch benchmark on, it will use tersorrt to predict.


build predict engine:

```python
from paddle.fluid.core import AnalysisConfig
from paddle.fluid.core import create_paddle_predictor
config = AnalysisConfig(model_file path, params_file path)
config.enable_use_gpu(8000, 0)
config.disable_glog_info()
config.switch_ir_optim(True)
config.enable_tensorrt_engine(
        precision_mode=AnalysisConfig.Precision.Float32,
        max_batch_size=1)

config.switch_use_feed_fetch_ops(False)

predictor = create_paddle_predictor(config)
```

Run the predictor

```python
import numpy as np

input_names = predictor.get_input_names()
input_tensor = predictor.get_input_tensor(input_names[0])
input = np.random.randn(1, 3, 224, 224).astype("float32")
input_tensor.reshape([1, 3, 224, 224])
input_tensor.copy_from_cpu(input)
predictor.zero_copy_run()
```

more information about attributes please refer to [Paddle Python inference API](https://www.paddlepaddle.org.cn/documentation/docs/zh/advanced_guide/inference_deployment/inference/python_infer_cn.html), More ionformation about deployment please refer to[Paddel C++ inference API](https://www.paddlepaddle.org.cn/documentation/docs/zh/advanced_guide/inference_deployment/inference/native_infer.html), compiled inference library please refer to [Paddle C++ Library](https://www.paddlepaddle.org.cn/documentation/docs/zh/advanced_guide/inference_deployment/inference/build_and_install_lib_cn.html)。


Please compile first when using TensorRt engine, please to [Paddle compile](https://www.paddlepaddle.org.cn/documentation/docs/zh/install/compile/fromsource.html)

## Training engine load persistable model

Run the `tools/infer/infer.py`


```python
python tools/infer/infer.py \
    --i=image path \
    --m=model path \
    --p=persistable model path \
    --use_gpu=True
```

Attributes:
+ `image_file`(i): imagebb path`./test.jpeg`
+ `model_file`(m):model file path`./resnet50-vd/model`
+ `params_file`(p): params file path`./resnet50-vd/params`
+ `use_gpu`: use gpu or not, default: True


build inference engine

```python
import fluid
from ppcls.modeling.architectures.resnet_vd import ResNet50_vd

place = fluid.CPUPlace()
exe = fluid.Executor(place)
startup_prog = fluid.Program()
infer_prog = fluid.Program()
with fluid.program_guard(infer_prog, startup_prog):
    with fluid.unique_name.guard():
        image = create_input()
        image = fluid.data(name='image', shape=[None, 3, 224, 224], dtype='float32')
        out = ResNet50_vd.net(input=input, class_dim=1000)
infer_prog = infer_prog.clone(for_test=True)
fluid.load(program=infer_prog, model_path=persistable model path, executor=exe)
```

execute:

```python
outputs = exe.run(infer_prog,
        feed={image.name: data},
        fetch_list=[out.name],
        return_numpy=False)
```

please refer to [fluid.Executor](https://www.paddlepaddle.org.cn/documentation/docs/zh/api_cn/executor_cn/Executor_cn.html) for more infomation

## training engine load inference model

Run the `tools/infer/py_infer.py`

```python
python tools/infer/py_infer.py \
    --i=image path \
    --d=model dir \
    --m=model path \
    --p=params path \
    --use_gpu=True
```
+ `image_file`(i): image path, like: `./test.jpeg`
+ `model_file`(m): model path, like: `./resnet50_vd/model`
+ `params_file`(p): params file path, like: `./resnet50_vd/params`
+ `model_dir`(d): model dir, like: `./resent50_vd`
+ `use_gpu`: use gpu or not, default: True


Build engine:

```python
import fluid

place = fluid.CPUPlace()
exe = fluid.Executor(place)
[program, feed_names, fetch_lists] = fluid.io.load_inference_model(
        save+path,
        exe,
        model_filename=model path,
        params_filename=params path)
compiled_program = fluid.compiler.CompiledProgram(program)
```

> `load_inference_model` support both combined files and uncombined file

execute

```python
outputs = exe.run(compiled_program,
        feed={feed_names[0]: data},
        fetch_list=fetch_lists,
        return_numpy=False)
```

please refer to [fluid.Executor](https://www.paddlepaddle.org.cn/documentation/docs/zh/api_cn/executor_cn/Executor_cn.html) for more information
