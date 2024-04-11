# Flash Hyperbolic Attention Minimal

A minimal re-implementation of Flash Attention with CUDA and PyTorch. The official [implementation](https://github.com/Dao-AILab/flash-attention) can be quite daunting for a CUDA beginner (like myself), so this repo tries to be small and educational.

* The end goal of this repo is to implement Flash Attention-like kernels for the various hyperbolic attention algorithms, finally making them production-ready.
* This was forked from [Peter Kim](https://github.com/tspeterkim)'s [flash-attention-minimal](https://github.com/tspeterkim/flash-attention-minimal) repo.
* The variable names follow the notations from the original [paper](https://arxiv.org/abs/2205.14135).

## Usage

### Prerequisite

* PyTorch (with CUDA)
* `Ninja` for loading in C++

### Benchmark

Compare the wall-clock time between manual attention and minimal flash attention:

```bash
python bench.py
```

Sample output on an RTX 3060 for the forward pass (Br = Bc = 32):

```bash
=== profiling manual attention (forward pass) ===
...
Self CPU time total: 375.381ms
Self CUDA time total: 377.542ms

=== profiling minimal flash attention 1 (forward pass) ===
...
Self CPU time total: 527.162ms
Self CUDA time total: 108.211ms

=== profiling minimal flash attention 2 (forward pass) ===
...
Self CPU time total: 343.248ms
Self CUDA time total: 4.048ms
```

That's a 3.5x & 94x speedup for Flash Attention 1 & 2, respectively!

Sample output on an RTX 3060 for the backward pass (Br = Bc = 16):

```bash
=== profiling manual attention (backward pass) ===
...
Self CPU time total: 65.457ms
Self CUDA time total: 67.838ms

=== profiling minimal flash attention 1 (backward pass) === 
...
Self CPU time total: 1.013s
Self CUDA time total: 4.615ms

=== profiling minimal flash attention 2 (backward pass) === 
...
Self CPU time total: 1.023s
Self CUDA time total: 814.000us
```

That's a 15x & 83x speedup for Flash Attention 1 & 2, respectively!

### I don't have a GPU

Try out this [online colab demo](https://colab.research.google.com/gist/tspeterkim/143bc7be7a845656817cf94c5228598e/demo-flash-attention-minimal.ipynb).

## Caveats

* In the inner loop, I assign each thread to a row of the output matrix. This differs from the original implementation.
* This thread-per-row simplification makes the matrix multiplications very slow. This is probably why for longer
sequences and larger block sizes, this gets slower than the manual implementation.
* Q,K,Vs are in float32, unlike the original implementation which uses float16.
* The block size is [fixed](https://github.com/tspeterkim/flash-attention-minimal/blob/9b7ca8ef4e6afdbfeb149a9cd488c8dea9af9ad6/flash.cu#L85) at compile time to 32.

## Todos

* [ ] Speed up matmults
* [ ] Dynamically set block size

## Contributors

* [Franz Cesista](https://github.com/leloykun), Implemented the backward pass for the Flash Attention 1 algorithm & both the forward and backward passes for the Flash Attention 2 algorithm.
* [Peter Kim](https://github.com/tspeterkim), Implemented the forward pass for the minimal Flash Attention 1 algorithm. See [original repo](https://github.com/tspeterkim/flash-attention-minimal)
