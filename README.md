# PCECGAN

### Representitive Results
![representive_results](/assets/show_3.png)

### Overal Architecture
![architecture](/assets/arch.png)

## Environment Preparing
```
python3.5
```
You should prepare at least 3 1080ti gpus or change the batch size. 


```pip install -r requirement.txt``` </br>
```mkdir model``` </br>
Download VGG pretrained model from [[Google Drive 1]](https://drive.google.com/file/d/1IfCeihmPqGWJ0KHmH-mTMi_pn3z3Zo-P/view?usp=sharing), [[2]](https://drive.google.com/file/d/190BBev58S1QRS2nDKQR5Ijx04_GOJgW6/view?usp=sharing) and then put them into the directory `model`.

### Training process
Before starting training process, you should launch the `visdom.server` for visualizing.

```nohup python -m visdom.server -port=8097```

then run the following command

```python scripts/script.py --train```

### Testing process

Download [pretrained model](https://drive.google.com/file/d/1AkV-n2MdyfuZTFvcon8Z4leyVb0i7x63/view?usp=sharing) and put it into `./checkpoints/enlightening`

Create directories `../test_dataset/testA` and `../test_dataset/testB`. Put your test images on `../test_dataset/testA` (And you should keep whatever one image in `../test_dataset/testB` to make sure program can start.)

Run

```python scripts/script.py --predict```

### Dataset preparing

Training data [[Google Drive trainA]](https://drive.google.com/drive/folders/1vIFu4dX6A14mah1URBDUtnhJJfN6ihFc?usp=sharing) and [[Google Drive trainB]](https://drive.google.com/drive/folders/1W0MyqV2Mu2fZof_TV8h6HF3J6QFbEZsr?usp=sharing) (unpaired images collected from multiple datasets)

Testing data [[Google Drive testA]](https://drive.google.com/drive/folders/1slK0mKf1AwWZfIkpSuygJRdTzw7HTYXo?usp=sharing)


