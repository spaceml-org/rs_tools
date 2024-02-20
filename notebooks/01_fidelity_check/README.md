# Fidelity Checks

Here we are developing basic fidelity checks for ITI data.

1. Field of Gaussian sources that vairy in intensity, size and position-angle.
1. Same field but in field of non-uniform noise.
1. Sharp-edged structures that vairy in intensity, size, aspect ratio and PA.
1. Same field but with non-uniform noise.
1. Pure noise field that varies on different length scales.

## Gaussian tests

First we generate a catalogue of Gaussians and use this to create a
test image. The positions of the Gaussian features are set on a
regular grid on the square image, perturbed by ```jitter``` pixels.

```
# Create a catalogue of Gaussian features
python mk_test_cat.py \
    --out testcat.csv \
    --num-features 10 \
    --jitter 3

# Make a test image from the catalogue
python mk_test_image.py \
    --cat testcat.csv \
    --out test \
    --size 500
```

Once the image has been generated, we run an ITI model on it and
analyse the pre- and post-model files. A comparison can be used to
highlight systemic changes.

```
# Run the automatic analysis routine
python analyse_test_image.py \
    --cat testcat.csv \
    --image test.tif  
```

