# RetNet
An implementation of [Retentive Network: A Successor to Transformer
for Large Language Models](https://arxiv.org/pdf/2307.08621.pdf) in PyTorch.

## About this repository
This is a minimal, pure pytorch implementation of RetNet. RetNet paper: [Retentive Network: A Successor to Transformer
for Large Language Models](https://arxiv.org/pdf/2307.08621.pdf).

The contributors(s) to this repository are not authors of the original paper. All credit for the idea and formulation of RetNet goes to the original authors.

The purpose of this repository is to aid scientific and technological understanding and advancement. The code prioritizes correctness and readability over optimization.

## Features implemented
* Single-scale and MultiScale retention:
  - parallel paradigm
  - recurrent paradigm
  - chunkwise paradigm
* Multi-layer retentive network with FFN and LayerNorm
  - parallel paradigm
  - recurrent paradigm
  - chunkwise paradigm
* Causal language model (CLM) built on top of the the retentive network

## Usage and Examples:
* See scripts prefixed with `test_` for examples of basic usage

## Positional Encodings
The main implementation in `src/` uses [Microsoft's xPos](https://github.com/microsoft/torchscale/blob/main/torchscale/component/xpos_relative_position.py) for positional encoding.

The implementation in `src/complex` uses complex values to encode position, which requires parameter and data throughput types to be `torch.ComplexFloat` (64-bit). This has some limitations due to there not yet being torch support for half-precision complex types. It also requires twice the amount of memory as real-valued data at 32-bit precision.

## Contributions
All contributions are welcome. Please see [issues](https://github.com/Jamie-Stirling/RetNet/issues) for an idea of what needs doing.

If you would like to contribute to this project, please fork it and submit a pull request for review.

## References
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
