# EUVlitho
EUVlitho includes the following two program sets.
1. Electromagnetic simulator for high-NA EUV lithography (elitho)
2. CNN reproducing the results of the electromagnetic simulations for fast EUV lithography simulation

The following paper explains the details of the programs.  
- H. Tanabe, M. Shimode and A. Takahashi, "Rigorous electromagnetic simulator for extreme ultraviolet lithography and convolutional neural network reproducing electromagnetic simulations," JM3 24(2025)024201. https://doi.org/10.1117/1.JMM.24.2.024201.
- H. Tanabe, M. Sugiyama, M. Shimode and A. Takahashi, "STCC formula including polarization and M3D effects in high-NA EUV lithography," SPIE Advance Lithography 2026, 13979-15, to be published. 
## 1. EUV lithography simulator (elitho)
See details in [this repository](https://github.com/takahashi-edalab/elitho).


## 2. CNN for fast EUV lithography simulation
### 2.1 Requirements
- CNN test (python) : Pytorch Ligtning (2.1.0) and CUDA (12.9).
- New data set generation (C++): OpenBLAS, Eigen (3.4.0), CUDA (12.9) and cuda_toolkit (12.9).  
### 2.2 CNN test
#### 2.2.1 Traing test data
Expand cnn/cnn/data.tar.gz.
#### 2.2.2 CNN model training
Execute cnn/cnn/model/re0/re0.py, im0/im0.py, rex/rex.py, imx/imx.py, rey/rey.py, imy/imy.py.
#### 2.2.3 CNN prediction
Execute cnn/cnn/predict/predict.py. Input: mask.bin (mask pattern). Output: inputxx.csv (M3D parameters). If you want to change the mask pattern, modify cnn/intensity/mask/mask.cpp.
### 2.3 Image intensity calculation
#### 2.3.1 Abbe's theory with M3D parameters (linear approximation)
Execute cnn/intensity/linear/linear.py. Inputs: mask.bin, inputxx.csv. Outputs: intft.csv (image intensity of the thin mask model), intlinear.csv (image intensity calculated by Abbe's theory with M3D parameters).
#### 2.3.2 STCC-SOCS formula with M3D parameters
Execute cnn/intensity/stccsocs/socs.py. Inputs: mask.bin, inputxx.csv. Output: intsocs.csv (image intensity calculated by STCC-SOCS formula with M3D parameters). When this program is executed at the first time, the eigen functins of the SOCS model are calcuated and stored. This calculation will not be repeated in the second time.
#### 2.3.3 Abbe's theory using the amplitude calculated by the electromagnetic simulation
Compile cnn/intensity/abbe/intenisty.cpp and exectute it. Input: mask.bin, Output: emint.csv (Image intensity calculated by the electromagnetic simulation).
### 2.4 New data set generation
Change the directory to cnn/cnn/data/train or validate. Modify mask.cpp to generate new mask patterns. Modify makem3d according to your enviroment. Modify "ndata" in m3d.cpp and compile it by "make -f makem3d." The calculation may take ~ 1 min for each mask pattern. After the calculation execute compress.py to rearrange the inputs for CNN.

### 2.5 References
- H. Tanabe, S. Sato, and A. Takahashi, “Fast EUV lithography simulation using convolutional neural network,” JM3 20(2021)041202. https://doi.org/10.1117/1.JMM.20.4.041202
- H. Tanabe and A. Takahashi, “Data augmentation in extreme ultraviolet lithography simulation using convolutional neural network,” JM3 21(2022)041602. https://doi.org/10.1117/1.JMM.21.4.041602
- H. Tanabe, A. Jinguji, and A. Takahashi, “Evaluation of convolutional neural network for fast extreme violet lithography simulation using 3nm node mask patterns,” JM3 22(2023)024201.  https://doi.org/10.1117/1.JMM.22.2.024201
- H. Tanabe, A. Jinguji and A. Takahashi, “Accelerating extreme ultraviolet lithography simulation with weakly guiding approximation and source position dependent transmission cross coefficient formula,” JM3 23(2024)014201. https://doi.org/10.1117/1.JMM.23.1.014201
