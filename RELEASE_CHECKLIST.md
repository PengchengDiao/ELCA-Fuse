# Release checklist

- [x] Keep the public directory source-only; do not include model weights.
- [x] Exclude datasets, experiment caches, generated results, and paper files.
- [x] Replace machine-specific paths with repository-relative CLI options.
- [x] Document all required checkpoint roles without redistributing checkpoints.
- [ ] Run syntax, import/CLI, SAHCA, and forbidden-file checks.
- [ ] Record the environment used for the final paper runs (Python, PyTorch,
  CUDA, cuDNN, operating system, and GPU).
- [ ] Add official MSRS and LLVIP download links after checking redistribution
  terms.
- [ ] Add the accepted-paper BibTeX entry, DOI, and repository URL.
- [ ] Confirm authorship and licensing for every retained source file.
- [ ] Run end-to-end MSRS and LLVIP inference with the private checkpoints.
- [ ] Tag the exact source commit used for the reported paper results.
