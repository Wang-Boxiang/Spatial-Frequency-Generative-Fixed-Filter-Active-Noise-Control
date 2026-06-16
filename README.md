# Spatial-Frequency Cued GFANC

This repository contains the Fig. 10 reproduction notebook for:

**Spatial-Frequency Cued Generative Fixed-Filter Active Noise Control Based on Deep Learning in Reverberant Environments**.

The released notebook demonstrates the spatial-frequency GFANC pipeline under different reverberation times. It uses a trained CRNN to estimate the source grid location from four-channel reference signals, selects the corresponding pretrained control filter, reconstructs the spatial-frequency control filter from the CRNN regression output, and plots noise reduction versus `RT60`.

## Repository Scope

This public release is intentionally scoped to the Fig. 10 simulation only.

Included:

- `notebooks/fig10_nr_vs_rt60.ipynb`: executable Fig. 10 notebook with embedded output.
- `CRNN.py`: CRNN model definition.
- `MyDataLoader_2D_BX.py`: four-channel STFT feature construction.
- `utilities_funcs.py`: microphone geometry, AWGN, and filter utilities.
- `Fixed_filter_noise_cancellation_2x1x1.py`: fixed-filter GFANC controller.
- `CNN_models/2D_v1002_CRNN_merged_v5.pth`: trained CRNN checkpoint used by Fig. 10.
- `Pre_trained_CFs_T60_*/CF_0.2_30_120.mat`: pretrained control filters for `RT60 = 0.1 ... 0.9 s`.
- `SecondaryPath_final.mat`: identified secondary path.
- `finaltest_noises/3-188726-A-35.wav` and `finaltest_noises/compre.wav`: real-noise examples used in Fig. 10.

Not included:

- training data and test data generation scripts
- full training datasets
- unrelated paper-figure scripts
- `DFT_Filter_Decompose.py`; the small filter-decomposition routine used by Fig. 10 is embedded directly in the notebook

## Environment

The code was verified with Python 3.10 and the existing `ANC` conda environment used for the paper experiments.

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

Key dependencies include `numpy`, `scipy`, `pandas`, `matplotlib`, `torch`, `torchaudio`, `soundfile`, and `gpuRIR`.

## Run Fig. 10

From the repository root:

```bash
jupyter notebook notebooks/fig10_nr_vs_rt60.ipynb
```

Then run all cells.

The notebook will:

1. load the trained CRNN checkpoint,
2. synthesize four-channel reference signals for `RT60 = 0.1 ... 0.9 s`,
3. run CRNN localization for each 0.5 s segment,
4. select the pretrained control filter from the predicted source grid,
5. reconstruct the spatial-frequency GFANC filter,
6. compute the noise reduction for four noise cases, and
7. display the Fig. 10 curve directly in the notebook.

The notebook also saves a standalone preview image:

```text
outputs/fig10_nr_vs_rt60.png
```

## Expected Prediction Check

For the included Fig. 10 setup, the target source grid is:

```text
distance = 0.2 m
elevation = 30 deg
azimuth = 120 deg
```

The executed notebook includes a prediction summary table. For all 9 reverberation times and all 4 noise cases, the CRNN predicts:

```text
(0.2 m, 30 deg, 120 deg)
```

for all 10 segments in each case.

## License

This project is released under the GNU General Public License v3.0. See `LICENSE`.
