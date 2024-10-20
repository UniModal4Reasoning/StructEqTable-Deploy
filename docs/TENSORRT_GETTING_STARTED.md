# Getting Started
[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) is used for model inference speeding up.  

All the codes are successfully tested in the following enviroments:
* Linux (18.04, 20.04, 22.04)
* Python 3.10
* Pytorch 2.0 or higher
* CUDA 12.1 or higher
* TensorRT-LLM 0.11.0 (stable version)

### 1. Conda or Python Environment Preparation


* Please follow the step 1, 2 from the [official tutorial](https://nvidia.github.io/TensorRT-LLM/installation/linux.html) of TensorRT-LLM to install the environment.  

Note we used the TensorRT-LLM **stable version `0.11.0`**.
``` bash
# Installing on Linux
Step 1. Retrieve and launch the docker container (optional).

    You can pre-install the environment using the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit) to avoid manual environment configuration.

    ```bash
    # Obtain and start the basic docker image environment (optional).
    docker run --rm --ipc=host --runtime=nvidia --gpus all --entrypoint /bin/bash -it nvidia/cuda:12.4.1-devel-ubuntu22.04
    ```
    Note: please make sure to set `--ipc=host` as a docker run argument to avoid `Bus error (core dumped)`.

Step 2. Install TensorRT-LLM.

    ```bash
    # Install dependencies, TensorRT-LLM requires Python 3.10
    apt-get update && apt-get -y install python3.10 python3-pip openmpi-bin libopenmpi-dev git git-lfs

    # Install the latest preview version (corresponding to the main branch) of TensorRT-LLM.
    # If you want to install the stable version (corresponding to the release branch), please
    # remove the `--pre` option.
    pip3 install tensorrt_llm==0.11.0 --extra-index-url https://pypi.nvidia.com

    # Check installation
    python3 -c "import tensorrt_llm"
    ```

    Please note that TensorRT-LLM depends on TensorRT. In earlier versions that include TensorRT 8,
    overwriting an upgraded to a new version may require explicitly running `pip uninstall tensorrt`
    to uninstall the old version.
```
* Once you successfully execute `python3 -c "import tensorrt_llm"`, it means that you have completed Environment Preparation.  

Tips: If you want to install the environment manually, please note that the version of Python require >= 3.10


### 2. Model Compilation
You can refer to the [official tutorial](https://nvidia.github.io/TensorRT-LLM/quick-start-guide.html) to complete the model compilation, or follow our instructions and use the provided scripts to implement it.

#### 2.1 Download [StructEqTable checkpoints](https://huggingface.co/U4R/StructTable-base/tree/v0.2)
```
cd StructEqTable-Deploy

# using huggingface-cli download checkpoint
huggingface-cli download --resume-download --local-dir-use-symlinks False U4R/StructTable-base --local-dir ckpts/StructTable-base

```
After above steps, the files to directory of StructEqTable-Deploy as follows:  
```
StructEqTable-Deploy
├── ckpts
│   ├── StructTable-base 
├── docs
├── struct_eqtable
├── tools
```

#### 2.2 Convert Checkpoint and Build Engine
We provide a script to help users quickly implement model compilation.

``` bash
cd StructEqTable-Deploy/tools
# execute the script to quickly compile the model.
bash scripts/build_tensorrt.sh 
```
After the script runs successfully, the built models can be found in `ckpts/StructTable-base-TensorRT`.  
The file structure in the path `ckpts/StructTable-base-TensorRT` should be as follows:  
```
ckpts
├── StructTable-base 
├── StructTable-base-TensorRT 
│   ├── trt_engines 
│   ├── trt_models
│   ├── visual_engiens
```

#### 2.3 Run Quickly Demo
Run the demo/demo.py with TensorRT mode.

``` bash
cd StructEqTable-Deploy/tools/demo

python demo.py \
  --image_path ./demo.png \
  --ckpt_path ../../ckpts/StructTable-base \
  --output_format latex
  --tensorrt ../../ckpts/StructTable-base-TensorRT
```

You may get output as follows:
```
total cost time: 0.88s
Table 0 LATEX format output:
\begin{tabular}{|c|c|c|c|}
\hline
Quantity $\backslash$ Unit System & International System SI (kg-m-s) & Traditional aeronautical (lb-ft-s) & Traditional structural (lb-inch-s) \\
\hline
Mass (translational inertia), $m$ & kilogram mass (kg) & slug = lb-s$^2$/f & lb-s$^2$/inch \\
\hline
Length, translational motion & meter (m) & foot (ft) & inch (in.) \\
\hline
Time, $t$ & second (s) & second (s) & second (s) \\
\hline
Force, translational action & newton (N) = kg-m/s$^2$ & pound force (lb) & pound force (lb) \\
\hline
Translational stiffness constant, $k$ & N/m & lb/ft & lb/inch \\
\hline
Translational damping constant, $c$ & N/(m/s) = N-s/m & lb/(ft/s) = lb-s/ft & lb/(inch/s) = lb-s/inch \\
\hline
Angle, rotational motion & radial (rad), which is dimensionless & radial (rad), which is dimensionless & radial (rad), which is dimensionless \\
\hline
Rotational inertia, $J$ & kg-m$^2$ & slug-ft$^2$ = lb-s$^2$ - ft & lb-s$^2$ - inch \\
\hline
Moment or torque, rotational action & N-m & lb-ft & lb-inch \\
\hline
Rotational stiffness constant, $k_\theta$ & (N-m)/rad = N-m & (lb-ft)/rad = lb-ft & (lb-inch)/rad = lb-inch \\
\hline
Rotational damping constant, $c_\theta$ & (N-m)/(rad/s) = N-m-s & (lb-ft)/(rad/s) = lb-ft-s & (lb-inch)/(rad/s) = lb-inch-s \\
\hline
\end{tabular}
```


### 3. Table Visualization
You can copy the output LaTeX code into [demo.tex](../tools/demo/demo.tex), then use [Overleaf](https://www.overleaf.com/project) or Visual Studio Code LaTeX Workshop Extension for table visualization.

![](./imgs/demo.png)