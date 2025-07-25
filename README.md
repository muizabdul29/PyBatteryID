# PyBatteryID

<div>

[![release](https://img.shields.io/github/v/release/muizabdul29/PyBatteryID)](https://github.com/muizabdul29/PyBatteryID/releases)
[![DOI](https://zenodo.org/badge/704093134.svg)](https://doi.org/10.5281/zenodo.15437221)

</div>

**PyBatteryID** — a shorthand for **Py**thon **Battery** Model **ID**entification, is an open-source library for data-driven battery model identification in the linear parameter-varying (LPV) framework. 

> Briefly, the LPV framework can be considered as an upgrade to the linear time-invariant (LTI) framework for systems whose dynamics cannot be considered time-invariant — for instance, batteries exhibit varying dynamics as the battery state-of-charge (SOC), temperature, current magnitude, and current direction vary. Refer to the book [Modeling and Identification of Linear Parameter-Varying Systems](https://link.springer.com/book/10.1007/978-3-642-13812-6) for a formal introduction to the LPV systems.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install PyBatteryID.

```bash
pip install pybatteryid
```

## Regarding the required experiments

There are a few experiments that the user may need to perform before using PyBatteryID, namely,

1. **Battery electromotive force (EMF) data**: Also referred to as the open-circuit voltage (OCV). The user needs to provide the (SOC, EMF) points that cover the SOC range expected in the test application. PyBatteryID performs interpolation using the given points to determine an EMF value for a certain SOC value. Note that by design, PyBatteryID DOES NOT perform extrapolation outside the given SOC range, because this can lead to poor modelling results.

2. **Current-voltage data**: The user must provide a sufficiently informative identification dataset, comprising current-voltage data (and optionally, temperature data if temperature-dependent models are desired). Please DO NOT expect a low-quality dataset, such as an HPPC test or a random drive cycle, to give you good models. In our paper [1], we have emphasised as much as possible that an informative dataset is crucial for obtaining high-quality (accurate, sparse, etc.) models, while presenting a current profile design that can lead to suitable identification datasets (see [examples/4_input_design.ipynb](/examples/4_input_design.ipynb)). Also note that this requirement holds for *any* modelling activity, and not specific to PyBatteryID or batteries, namely, it is a general principle of the **[System Identification](https://en.wikipedia.org/wiki/System_identification)** to use informative identification datasets.

## Basic usage

In the following, an example usage of PyBatteryID has been demonstrated for modelling the battery overpotential using the LPV framework while assuming the battery electromotive force (EMF) to be known *a priori* via appropriate experiments, such as GITT, or low-current cycling. In effect, the battery voltage output at a given time instant can then be calculated by using the EMF value at that instant and evaluating the overpotential using the identified LPV model.

> It is recommended that the user follows the International System of Units (SI) while using PyBatteryID. For example, the battery capacity should be specified in Coulombs, time in seconds, current in amperes, and voltage in volts. For the temperature, both Celsius or Kelvin can be used as long as the user stays consistent and adjusts the temperature-related basis functions accordingly. Note that the temperature is in Celsius in the [example](/examples/3_1_nmc_with_temperature_identification.ipynb) provided with the package.

#### 1. Initialize model structure

The very first step is to create an instance of the `ModelStructure` class which requires the capacity of the studied battery and the model sampling time.

```python
model_structure = ModelStructure(battery_capacity=3440, sampling_period=1)
```

#### 2. Load EMF data

The next step is to provide the battery EMF data to the `ModelStructure` class using the `add_emf_function` method. The expected argument is a dictionary with keys `soc_values` and `voltage_values` representing a list of SOC values and a list of the corresponding EMF values, respectively. 

```python
model_structure.add_emf_function({'soc_values': soc_values,
                                  'voltage_values': voltage_values})
```

The EMF function can be made temperature-dependent by providing two extra arguments, namely, (1) `reference_temperature_value` that refers to the temperature corresponding to the (`soc_values`, `voltage_values`), and (2) `dVdT_values` that describes how the EMF depends on temperature assuming linear temperature dependence.

```python
model_structure.add_emf_function({'soc_values': soc_values,
                                  'voltage_values': voltage_values,
                                  'reference_temperature_value': reference_temperature_value,
                                  'dVdT_values': dVoltage_dTemperature_values})
```

#### 3. Add basis functions

PyBatteryID has reserved certain symbols for the scheduling-variable components, namely, `s`, `|i|`, `d` and `T` for the SOC, current magnitude, current direction and temperature, respectively. It allows the user to conveniently describe certain functional forms of the aforementioned variables by detecting them automatically. In the following, these functional forms are described using the symbol $\square$ as a placeholder for a scheduling-variable component and $\diamond$ for associated hyperparameter(s).

- $\square$ — No functional transformation.
- `1/`$\square$ — Inverse of the variable.
- `log[` $\square$ `]` — Logarithmic transformation of the variable.
- `exp[`$\diamond$`*sqrt[` $\square$ `]]` — Square-root of the variable followed by exponential transformation, where the hyperparameter can act as a normalization factor. See [[1]](#1).
- `exp[` $\diamond_0$ `*[` $\square$ `]^` $\diamond_1$ `]` — Raising the variable to an arbitrary power followed by exponential transformation. Note that the variable can be multiplied by a number and also added to a number. See below for an example.
- $\square$`[` $\diamond_0$ `, ` $\diamond_1$ `]` — Low-pass filtering of the variable using first-order difference equation. The two hyperparameters correspond to $\varepsilon_0$ and $\varepsilon_1$, where $\varepsilon_0$ gets chosen when the variable is nonzero, and $\varepsilon_1$ otherwise; see [[1]](#1) for more details. Note that such an operation only makes sense with the current direction.

For instance, one may use the following basis functions,

```python
model_structure.add_basis_functions(['1/s', 'log[s]', 's',
                                     'exp[0.05*sqrt[|i|]]', 'exp[[0.00366*T+1]^-1]', 'd[0.01,0.99]'])
```

#### 4. Run optimization routine(s) to identify model

PyBatteryID allows specification of multiple optimization routines in a pipeline, which can then run in a sequential manner. Indeed, the user needs to provide an identification dataset in the form of a dictionary with keys `initial_soc`, `time_values`, `current_values`, `voltage_values`, and (optional) `temperature_values`. To obtain a battery model, the function `identify_model` can be used which has the following arguments,

- `dataset` — The identification dataset.
- `model_structure` — Instance of the `ModelStructure` class.
- `model_order` — Model order $n$.
- `nonlinearity_order` — Nonlinearity order $l$.
- `optimizers` — A list of optimization routines to run in a sequential manner for a regression setting $A\cdot\theta=y$; where $A$ is the regression matrix and $y$ the output vector. Currently, the available options are `lasso.cvxopt`, `lassocv.sklearn` and `ridge.sklearn`.

The output of the function `identify_model` represents an instance of a class `Model` in the PyBatteryID package containing all the necessary information regarding the identified battery model.

```python
model = identify_model(identification_dataset, model_structure,
                       model_order=3, nonlinearity_order=3,
                       optimizers=['lassocv.sklearn', 'ridgecv.sklearn'])
```

##### A short note on `optimizers`:

The available optimization routines can be specified using the name of the algorithm suffixed by the name of the package providing the algorithm, for instance, `lasso.cvxopt` and `lassocv.sklearn`. Note that the optimizer `lasso.cvxopt` uses fixed regularization parameter as $\lambda_1 = 1$, whereas `lassocv.sklearn` performs cross-validated LASSO regression. Among the two options, `lassocv.sklearn` is recommended since it has been configured to take significantly less time than the optimizer `lasso.cvxopt`. Furthermore, `lassocv.sklearn` may result in more stable models since the cross-validation can lead to adequate regularization in the case of less informative dataset and/or less representative model structure (i.e., inappropriate candidate model terms).

#### 5. Simulate voltage using the identified model

After the identification of a battery model, we can simulate the voltage output for a certain current profile. In this regard, the method `simulate_model` can be used which requires the following arguments,

- `model` — The identified model.
- `dataset` — A dictionary with keys `initial_soc`, `time_values`, `current_values`, `voltage_values` and (optional) `temperature_values`. The `voltage_values` key corresponds to a list of initial voltage values. Note that the number of initial values should be at least equal to the model order. The initial SOC value corresponds to the SOC value at the first time instant `time_values[0]`.

An example usage of the method `simulate_model` can be given as follows,

```python
voltage_simulated = simulate_model(model, dataset)
```

## Relevant publications

<a id="1">[1]</a> A.M.A. Sheikh, M.C.F. Donkers, and H.J. Bergveld, “A comprehensive approach to sparse identification of linear parameter-varying models for lithium-ion batteries using improved experimental design,” *Journal of Energy Storage, 2024*. https://doi.org/10.1016/j.est.2024.112581

<a id="2">[2]</a> A.M.A. Sheikh, M.C.F. Donkers, and H.J. Bergveld, “Investigating Identification Input Designs for Modelling Lithium-ion Batteries with Hysteresis using LPV Framework,” *2024 American Control Conference (ACC)*. https://doi.org/10.23919/ACC60939.2024.10644893

<a id="3">[3]</a> A.M.A. Sheikh, M.C.F. Donkers, and H.J. Bergveld, “Towards temperature-dependent linear
parameter-varying models for lithium-ion batteries using novel experimental design,” *Journal of Energy Storage, 2025*. https://doi.org/10.1016/j.est.2025.116311

## Get in touch

For general inquiries about using the package, you can either [start a discussion](https://github.com/muizabdul29/PyBatteryID/discussions) or email at [a.m.a.sheikh@tue.nl](mailto:a.m.a.sheikh@tue.nl) (Muiz Sheikh).

## License
PyBatteryID is an open-source library licensed under the BSD-3-Clause license. For more information, see [LICENSE](LICENSE.txt).