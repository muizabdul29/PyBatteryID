# PyBatteryID

**PyBatteryID** — a shorthand for **Py**thon **Battery** Model **ID**entification, is an open-source library for data-driven battery model identification in the linear parameter-varying (LPV) framework. 

> Briefly, the LPV framework can be considered as an upgrade to the linear time-invariant (LTI) framework for systems whose dynamics cannot be considered time-invariant — for instance, batteries exhibit varying dynamics as the battery state-of-charge (SOC), temperature, current magnitude, and current direction vary. Refer to the book [Modeling and Identification of Linear Parameter-Varying Systems](https://link.springer.com/book/10.1007/978-3-642-13812-6) for a formal introduction to the LPV systems.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install PyBatteryID.

```bash
pip install pybatteryid
```

## Basic usage

In the following, an example usage of PyBatteryID has been demonstrated for modelling the battery overpotential using the LPV framework while assuming the battery electromotive force (EMF) to be known *a priori* via appropriate experiments, such as GITT, or low-current cycling. In effect, the battery voltage output at a given time instant can then be calculated by using the EMF value at that instant and evaluating the overpotential using the identified LPV model.

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

#### 3. Add basis functions

PyBatteryID has reserved certain symbols for the scheduling-variable components, namely, `s`, `|i|` and `d` for the SOC, current magnitude and current direction, respectively. It allows the user to conveniently describe certain functional forms of the aforementioned variables by detecting them automatically. In the following, these functional forms are described using the symbol $\square$ as a placeholder for a scheduling-variable component and $\diamond$ for associated hyperparameter(s).

- $\square$ — No functional transformation.
- `1/`$\square$ — Inverse of the variable.
- `log[` $\square$ `]` — Logarithmic transformation of the variable.
- `exp[`$\diamond$`*sqrt[` $\square$ `]]` — Square-root of the variable followed by exponential transformation, where the hyperparameter can act as a normalization factor. See [[1]](#1).
- $\square$`[` $\diamond_0$ `, ` $\diamond_1$ `]` — Low-pass filtering of the variable using first-order difference equation. The two hyperparameters correspond to $\varepsilon_0$ and $\varepsilon_1$, where $\varepsilon_0$ gets chosen when the variable is nonzero, and $\varepsilon_1$ otherwise; see [[1]](#1) for more details. Note that such an operation only makes sense with the current direction.

For instance, one may use the following basis functions,

```python
model_structure.add_basis_functions(['1/s', 'log[s]', 's',
                                     'exp[0.05*sqrt[|i|]]', 'd[0.01,0.99]'])
```

#### 4. Run optimization routine(s) to identify model

PyBatteryID allows specification of multiple optimization routines in a pipeline, which can then run in a sequential manner. Indeed, the user needs to provide an identification dataset in the form of a dictionary with keys `time`, `current` and `voltage`. To obtain a battery model, the method `identify` can be used which has the following arguments,

- `dataset` — The identification dataset.
- `initial_soc` — The battery SOC value corresponding to the first time instant in `dataset`.
- `model_order` — Model order $n$.
- `nonlinearity_order` — Nonlinearity order $l$.
- `optimizers` — A list of optimization routines to run in a sequential manner for a regression setting $A\cdot\theta=y$; where $A$ is the regression matrix and $y$ the output vector. Currently, the available options are `lasso.cvxopt`, `lasso.sklearn` and `ridge.sklearn`.

The output of the method `identify` represents an instance of a class `Model` in the PyBatteryID package containing all the necessary information regarding the identified battery model.

```python
model = model_structure.identify(dataset, initial_soc=0.982677,
                                 model_order=1, nonlinearity_order=1,
                                 optimizers=['lasso.cvxopt', 'ridge.sklearn'])
```

##### A short note on `optimizers`:

The available optimization routines can be specified using the name of the algorithm suffixed by the name of the package providing the algorithm, for instance, `lasso.cvxopt` and `lasso.sklearn`. Note that the optimizer `lasso.cvxopt` uses fixed regularization parameter as $\lambda_1 = 1$, whereas `lasso.sklearn` performs cross-validated LASSO regression. Among the two options, `lasso.sklearn` is recommended since it has been configured to take significantly less time than the optimizer `lasso.cvxopt` at the cost of slight accuracy loss. Furthermore, `lasso.sklearn` may result in more stable models since the cross-validation can lead to adequate regularization in the case of less informative dataset and/or less representative model structure (i.e., inappropriate candidate model terms).

#### 5. Simulate voltage using the identified model

After the identification of a battery model, we can simulate the voltage output for a certain current profile. In this regard, the method `simulate` can be used which requires the following arguments,

- `model` — The identified model.
- `dataset` — A dictionary with keys `time`, `current` and `voltage`. The `voltage` key corresponds to a list of initial voltage values. Note that the number of initial values should be at least equal to the model order.
- `initial_soc` — The initial SOC value corresponding to the first time instant in `dataset`.

An example usage of the method `simulate` can be given as follows,

```python
voltage_simulated = model_structure.simulate(model, dataset, initial_soc=0.97973)
```

## Relevant publications

<a id="1">[1]</a> A.M.A. Sheikh, M.C.F. Donkers, and H.J. Bergveld, “A comprehensive approach to sparse identification of linear parameter-varying models for lithium-ion batteries using improved experimental design,” *Journal of Energy Storage, 2024*.

<a id="2">[2]</a> A.M.A. Sheikh, M.C.F. Donkers, and H.J. Bergveld, “Investigating Identification Input Designs for Modelling Lithium-ion Batteries with Hysteresis using LPV Framework,” *2024 American Control Conference (ACC)*.

## License
PyBatteryID is an open-source library licensed under the BSD-3-Clause license. For more information, see [LICENSE](LICENSE.txt).