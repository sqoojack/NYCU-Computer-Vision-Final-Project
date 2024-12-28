<p align="center">
  <h2 align="center">Computer Vision Final Project Group18</h2>
  






### Installation

Create a conda environment and install PyTorch:

```
conda create -n sgi2v python=3.12.4
conda activate sgi2v
conda install pytorch=2.3.1 torchvision=0.18.1 pytorch-cuda=11.8 -c pytorch -c nvidia
```

Install packages:
```
pip install -r requirements.txt
```


## :paintbrush: Usage

  ### Quick start with a notebook
  You can run ```demo.ipynb```, which contains all the implementations (along with a light explanation) of our pipeline.

  ### Reproducing qualitative results

  Alternatively, you can generate example videos demonstrated on the project website by running:

  ```
  Original Paper: python inference.py --input_dir <input_path> --output_dir <output_path>
  Ours modify: python inference_try.py --input_dir <input_path> --output_dir <output_path>
  ```
  
  


