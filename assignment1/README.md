# Superresolution and Enhancement

<div align="justify" >

In this assignment, three distinct image enhancement techniques were implemented,
as well as a superresolution method based on multiple views of the same image.

The general structere of this code is:

1. Find and load all low resolution images Li ∈ L that match the basename _imglow_
(i.e. filenames that start with _imglow_)
2. Apply the selected enhancement method F to all low resolution images, using
parameter γ when appropriate
3. Combine the low resolution images into a high resolution version supe_H
4. Compare super_H against reference image H using Root Mean Squared Error (RMSE)

## Input Parameters

The following parameters were used as input in the following order:

1. basename imglow for low resolution images li ∈ L. The basename references the
start of the filenames for 4 low resolution images l1, l2, l3, l4
2. filename imghigh for the high resolution image H
3. enhancement method identifier F (0, 1, 2 or 3)
4. enhancement method parameter γ for F = 3

## Image Enhancement

There are three options for Image Enhancement, with Option 0 indicating that
no enhancement is to be done:

### Option 0: No Enhancement
No enhancement technique is applied to the image, and instead, it skips to 
the superresolution step.

### Options 1 and 2:
They are histogram-based methods while Option 3 uses pixel-based Gamma
correction.

## Comparing against reference

The program compares the enhanced image H against the reference image 
super_H. This comparison uses the root mean squared error (RMSE).

$$ RMSE = \sqrt{\frac{\sum_i\sum_j(H(i,j) - superH(i,j))^2}{N^2}} $$

where $`N × N`$ is the resolution of images H and super_H .

## Examples

In folder `images` there are _.png_ files to test. At the end of execution, superH and H images
are displayed, allowing for visual comparison.

![Execution Example](Example.png)

In folder `test_cases_in_out` there are files `caseXX.in` and `casesXX.out` where 
_.in_ contains input examples and _.out_ contains the respective output. Note that, when run
locally, RMSE values may vary by 0.75 due to specific machine errors.

To run `main.py` with an example, use that line: `python3 main.py < test_cases_in_out/caseXX.in`
