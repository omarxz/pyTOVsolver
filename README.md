# pyTOVSolver

`pyTOVSolver` is a python-based project designed to solve the modified Tolman-Oppenheimer-Volkoff (TOV) equations for scalar-tensor theories for different equations of state (EOS). It is used to model neutron stars and other compact objects in astrophysics. The solver is efficient and allows for parallel computations of various initial central densities.

This project was part of a research study in which we solved the modified TOV equations for an axion that is coupled to matter. You can read more about this research in the related publication: [Cosmology and Astrophysics of CP-Violating Axions](https://arxiv.org/abs/2408.02294) in which we studied the consequences of the CP-violating axion-like particles (ALPs) on the cosmological history and compact objects. If you have used the code for a publication, please cite our paper.


## Features

- Solves TOV equations for Scalar-tensor theories of gravity.
- Computes mass-radius relations for neutron stars.
- Parallel computation for multiple stars with different central densities.
- Automatic detection of available CPU cores for parallel execution.


## Environment Setup
Instead of manually installing dependencies, the project provides an `environment.yml` file to streamline virtual environment creation. It is recommended to use this file with conda to create the necessary environment. This is the same enviroment I used to develop the project and use it without issues.

You can create the environment using the following command:

```bash
conda env create -f environment.yml
```
If you'd like to specify a custom name when creating the environment, you can use:

```bash
conda env create -f environment.yml --name custom-env-name
```
Otherwise, the environment's default name is `pyTOVSolver_venv` and can be activated by running this
```bash
conda activate pyTOVSolver_venv
```
## Running the Solver
The main code for solving the TOV equations is run through the `parallel_full_sol.py` script. This script triggers the parallelized computation for a range of central densities for neutron stars:

```bash
python parallel_full_sol.py
```
Once the computation is complete, you can open the provided notebook for inference, `inference_M-R.ipynb`and make plots for the runs you need. However, this approach is not recommended as the optimization process for a single star could take up to ~30 minutes.

## Recommended Workflow
The recommended approach is to use an HPC (High-Performance Computing) cluster. A sample `slurm_scripts` directory is provided. You can edit the sample Slurm script with the desired parameters (e.g., g_s_N, m_a, f_a). The script will automatically copy parallel_full_sol.py, edit the copy with your parameters, and save it without modifying the original file for the purpose of debugging if needed. This method allows you to leverage the power of HPC clusters for faster computation, especially for running different sets of model parameters without modifying the main structure.

## Extracting Data with extract_run_save.py
The extract_run_save.py script is only used if the main code stops halfway and you need to extract the central fields with their corresponding densities. This use case is rare, but the script provides functionality for recovery and partial data extraction.

```bash
python extract_run_save.py
```
Make sure to configure your input parameters, such as `num_of_stars` and `rho_c_values`, in the script.

Parallelization
The `parallel_full_sol.py` script is designed to automatically detect the number of CPU cores available on the machine and parallelize the computations accordingly. This is done using Python's `ProcessPoolExecutor` from the `concurrent.future`s module. By distributing the tasks across all available CPU cores, the script maximizes computational efficiency without requiring manual intervention.

For instance, the code snippet below demonstrates how rho_c_values are processed in parallel:

```python
if __name__ == "__main__":
    rho_c_values = [10**i for i in np.linspace(np.log10(4e14), np.log10(1.778e16), num_of_stars)]
    # Use ProcessPoolExecutor to parallelize the optimization
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(compute_for_rho_c, rho_c) for rho_c in rho_c_values]
        results = [future.result() for future in futures if future.result() is not None]
```
The ProcessPoolExecutor automatically utilizes all available CPU cores, ensuring efficient parallel processing across the machine's resources.
