# Fourier Transform & Filtering in Frequency Domain

<div align="justify" >

In this assignment, it was implemented functions in order to filter images in the frequency
domain using the Discrete Fourier Transform. The program follows the following steps:

## Read the parameters:
a) Filename for the input image ( I ). </br>
b) Filename for the reference image ( H ). </br>
c) Filter index i ∈ M = {0, 1, 2, 3, 4}. </br>
d) Filter Parameters respective to each index;

## Use the filters below:
a) index i = 0 - Ideal Low-pass - with radius r. </br>
b) index i = 1 - Ideal High-pass - with radius r. </br>
c) index i = 2 - Ideal Band-stop - with radius r0 and r1. </br>
d) index i = 3 - Laplacian High-pass. </br>
e) index i = 4 - Gaussian Low-pass - with σ1 and σ2; </br>

## Process the input images:
a) Generate the Fourier Spectrum ( F ( I ) ) for the input image I. </br>
b) Filter F ( I ) multiplying it by the input filter Mi. </br>
c) Generate the filtered image G back in the space domain. </br>
d) Compare the output image ( G ) with the reference image ( H ). </br>

## Comparing against reference

The program compares the restored image G against the reference image H.
This comparison uses the root mean squared error (RMSE).

$$ RMSE = \sqrt{\frac{\sum_i\sum_j(H(i,j) - G(i,j))^2}{M \cdot N}} $$

where $`M × N`$ is the resolution of images H and G.

## Examples

In folder `images` there are _.png_ files to test. At the end of execution, the input image I, the
restored image G, the filter used, and H images are displayed, allowing for visual comparison.

![Execution Example](Example.png)

In folder `test_cases_in_out` there are files `caseX.in` and `casesX.out` where 
_.in_ contains input examples and _.out_ contains the respective output. Note that, when run
locally, RMSE values may vary by 0.75 due to specific machine errors.

To run `main.py` with an example, use that line: `python3 main.py < test_cases_in_out/caseX.in`
