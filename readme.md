# ADL Group Assignment


### Weakly-Supervised Section
Weakly-supervised requires SAMv2. The command below clones this repository into weakly_supervised/sam2

```
git submodule update --init;
```

Once cloned, install SAMv2 by running:

```
cd weakly_supervised/sam2;
pip install -e . ;
```

Rather than running ```download_ckpts.sh``` you can just manually load the ```sam2.1_hiera_large.pt``` weights by running:

```
curl -L -o ../checkpoints/sam2.1_hiera_large.pt "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt";
```


There are also some [utils](utils) for logging and evaluation

