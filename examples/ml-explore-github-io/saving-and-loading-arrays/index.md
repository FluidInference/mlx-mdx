---
title: Saving and Loading Arrays
source_url: https://ml-explore.github.io/mlx/build/html/usage/saving_and_loading.html
retrieved_at: 2025-10-17T22:35:49Z
---
# Saving and Loading Arrays

MLX supports multiple array serialization formats.

The `load()` function will load any of the supported serialization formats. It determines the format from the extensions. The output of `load()` depends on the format.

Here’s an example of saving a single array to a file:

```python
>>> a = mx.array([1.0])
>>> mx.save("array", a)
```

The array `a` will be saved in the file `array.npy` (notice the extension is automatically added). Including the extension is optional; if it is missing it will be added. You can load the array with:

```python
>>> mx.load("array.npy")
array([1], dtype=float32)
```

Here’s an example of saving several arrays to a single file:

```python
>>> a = mx.array([1.0])
>>> b = mx.array([2.0])
>>> mx.savez("arrays", a, b=b)
```

For compatibility with `numpy.savez()` the MLX `savez()` takes arrays as arguments. If the keywords are missing, then default names will be provided. This can be loaded with:

```python
>>> mx.load("arrays.npz")
{'b': array([2]), 'arr_0': array([1]), }
```

In this case `load()` returns a dictionary of names to arrays.

The functions `load()` and `save_safetensors()` and `save_gguf()` are similar to `savez()`, but they take as input a `dict` of string names to arrays:

```python
>>> a = mx.array([1.0])
>>> b = mx.array([2.0])
>>> mx.save_safetensors("arrays", {"a": a, "b": b})
```
```markdown
```python
>>> a = mx.array([1.0])
>>> b = mx.array([2.0])
>>> mx.savez("arrays", a, b=b)
```
```markdown
```python
>>> a = mx.array([1.0])
>>> b = mx.array([2.0])
>>> mx.save_safetensors("arrays", {"a": a, "b": b})
