**安装说明**

```bash
# cpu版本
conda create --name d2l python=3.9 -y
conda activate d2l
pip install torch==2.0.0 torchvision==0.15.1
pip install d2l==1.0.3

## GPU版本
conda install pytorch=2.0 torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```
