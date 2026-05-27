# Dataset Layout

Place each dataset in ImageFolder format:

```text
data/
  Data1/
    train/
      cocci/
      healthy/
      ncd/
      salmo/
    val/
      cocci/
      healthy/
      ncd/
      salmo/
    test/
      cocci/
      healthy/
      ncd/
      salmo/
```

The code maps common folder aliases such as `Coccidiosis`, `Healthy`,
`New Castle Disease`, and `Salmonella` to the canonical label order used by the
models.

