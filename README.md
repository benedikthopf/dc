# Clip representation from DINOSAUR slots

Imitates a [CLIP](https://github.com/openai/CLIP) embedding based on [DINOSAUR](https://arxiv.org/abs/2209.14860) slots

## Install

> ```conda env create -f environment.yml```
> ```conda activate dc```

## Training

see
> `python train.py --help`
and the `.sbatch` files in the `sbatches` folder

## Acknowledgements

- `slate_transformer.py` and `slate_utils.py` have been taken from `https://github.com/singhgautam/slate`.
- `slot_attention.py` and `dataset.py` have been taken from `https://github.com/evelinehong/slot-attention-pytorch`. 
- `dinosaur.py` has been built based on the paper from `https://arxiv.org/abs/2209.14860`.
