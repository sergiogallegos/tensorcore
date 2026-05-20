# TensorCore Improvement TODO

This list tracks the eight review priorities for turning TensorCore into a smaller,
correct, educational ML library.

## 1. Fix build and test truth

- [x] Keep root `CMakeLists.txt` in sync with the actual source and test files.
- [x] Register runnable tests with CTest.
- [ ] Remove stale build artifacts from the working tree and keep them ignored.
- [ ] Add CI once the local build is stable.

## 2. Shrink the public API to implemented features

- [x] Audit every declaration in `include/tensorcore/*.hpp`.
- [x] Move incomplete modules out of the default build.
- [x] Hide incomplete operations and utility APIs from the default header surface.
- [ ] Move remaining incomplete APIs to an `experimental` namespace.
- [x] Keep README feature lists aligned with implemented and tested behavior.

## 3. Make placeholders fail loudly

- [x] Replace placeholder returns with `std::runtime_error`.
- [x] Add tests that assert incomplete features throw clear errors.
- [ ] Avoid silent fake results in educational examples.

## 4. Implement core tensor semantics

- [ ] Add real scalar broadcasting.
- [ ] Resolve the shape-vs-data constructor ambiguity for calls like `Tensor({3, 4})`.
- [ ] Add shape broadcasting for element-wise operations.
- [ ] Implement axis reductions.
- [ ] Implement transpose with arbitrary axes.
- [ ] Implement slicing/index ranges.
- [ ] Add stride-aware tests for all of the above.

## 5. Redesign parameters and gradients

- [ ] Store gradients separately from parameter values.
- [ ] Let optimizers update real model parameters, not copies.
- [ ] Implement `zero_grad()` against actual gradient buffers.
- [ ] Add a tiny training test where loss decreases.

## 6. Integrate autograd with tensor operations

- [ ] Decide whether `Tensor` or `GraphNode` is the primary differentiable object.
- [ ] Make examples use one clear autograd model.
- [ ] Add gradient checks for add, multiply, matmul, sum, and activation functions.

## 7. Rewrite documentation to match reality

- [x] Split "working", "experimental", and "roadmap" sections.
- [x] Remove claims about complete gradients where implementations are partial.
- [ ] Add learning notes near each implemented algorithm.

## 8. Add lesson-quality examples

- [ ] Add formula-first examples for linear regression, MLP, softmax regression, and backpropagation.
- [ ] Compare tiny fixed examples against NumPy or PyTorch.
- [ ] Keep optimized versions separate from reference implementations.
