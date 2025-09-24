# Hyperbolic Gaussian-Based Mean Shift (HypeGBMS)

Implementation of Hyperbolic GBMS clustering algorithm on the Poincaré ball model.

## Requirements
- torch
- numpy
- scikit-learn
- scipy

## Example Dataset
You can run the algorithm on any tabular dataset.  
For quick testing, you can use scikit-learn datasets (e.g. `load_iris` or `load_wine`).

| Dataset      | Samples | Dimensions | Classes |
|-------------|---------|-----------|---------|
| Iris        | 150     | 4         | 3       |
| Wine        | 178     | 13        | 3       |
| Zoo         | 101     | 16        | 7       |

## Installation & Running
### Installing
Clone the repository and install dependencies:

```bash
git clone https://github.com/arnab37seal/HypeGBMS.git
cd HypeGBMS
pip install -r requirements.txt
pip install -e .

