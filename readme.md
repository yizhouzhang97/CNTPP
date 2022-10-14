# Counterfactual Neural Temporal Point Process for Misinformation Impact Estimation on Social Media(Neurips 2022)

This code is pytorch version of implementation for Counterfactual Neural Temporal Point Process for Misinformation Impact Estimation on Social Media.

## Data

### Synthetic data

Refer to '/data/syn_data'

### Real world data

We will publish the desensitized raw data and the point process data obtained after our processing after the paper is accepted as there is a limit to the upload capacity of Openreview platform.

## Train the proposed model:

 ```python
 python run_cuda.py
 ```

We will save the trained mode in the folder of 'model'.

## Get Individual treatment effects (ITE)

```python
 python getATE.py
 ```

 We will save the ITE score in the file of 'model/ITE.pkl'.

 
### Reference

github : https://github.com/omitakahiro/NeuralNetworkPointProcess 



