# RetNet
An implementation of [Retentive Network: A Successor to Transformer
for Large Language Models](https://arxiv.org/pdf/2307.08621.pdf) in PyTorch.

## About this repository
This is a minimal, pure pytorch implementation of RetNet. RetNet paper: [Retentive Network: A Successor to Transformer
for Large Language Models](https://arxiv.org/pdf/2307.08621.pdf) in PyTorch.

The contributors(s) to this repository are not authors of the original paper. All credit for the idea and formulation of RetNet goes to the original authors.

The purpose of this repository is to aid scientific and technological understanding and advancement.


## Features implemented:
* Single-scale and MultiScale retention:
  - parallel paradigm
  - recurrent paradigm
* Multi-layer retentive network with FFN and LayerNorm

## Features coming soon:
* chunkwise recurrent paradigm (for efficient inference)

## Usage and Examples:
* See scripts prefixed with `test_` for examples of basic usage

## Differences to original paper
While the paper contains no explicit references to complex algebra, this implementation uses `torch.ComplexFloat` (64-bit) for parameters and data throughput. This has some limitations due to there not yet being torch support for half-precision complex types. It also requires twice the amount of memory as real-valued data at 32-bit precision.


## Known Issues
* Error propagation: it appears in this implementation, errors propagate differently between the parallel usage and the recurrant usage. It's not clear whether this is a bug yet, but it sometimes causes tests to fail.

## References:
```
@misc{sun2023retentive,
      title={Retentive Network: A Successor to Transformer for Large Language Models}, 
      author={Yutao Sun and Li Dong and Shaohan Huang and Shuming Ma and Yuqing Xia and Jilong Xue and Jianyong Wang and Furu Wei},
      year={2023},
      eprint={2307.08621},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```