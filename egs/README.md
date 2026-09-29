# Dataset manifests

Each dataset split is a folder holding a `data.jsonl` (or `data.jsonl.gz`) manifest with one
line per audio file. Build one with:

```bash
python -m gscodec.data.audio_dataset /path/to/LibriTTS/train egs/libriTTS_train/data.jsonl
```

The `config/dset/audio/*.yaml` files map the `train`, `valid`, `evaluate` and `generate`
splits to these folders; override any of them on the command line, e.g.
`datasource.train=egs/my_corpus`.
