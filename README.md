[![DOI](https://img.shields.io/badge/DOI-10.5281/zenodo.21026496-blue)](https://doi.org/10.5281/zenodo.21026496) [![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/)

# Unified Prebiotic DNA Selection Framework (UPDSF) v3.1

UPDSF v3.1 is a high-performance stochastic simulation engine designed to model the chemical evolution and selection of DNA over RNA in prebiotic environments

## Author: Seyed Mohammad Reza Hashemi (Reza Hashemi)

Environment: 🐍 Python 3.8+

https://doi.org/10.5281/zenodo.20988680

https://doi.org/10.5281/zenodo.20825578

https://doi.org/10.5281/zenodo.20733760

https://doi.org/10.5281/zenodo.20759622

https://doi.org/10.5281/zenodo.20771213

https://doi.org/10.5281/zenodo.18594133

## Overview
UPDSF v3.1 is a high-performance stochastic simulation engine designed to model the chemical evolution and selection of DNA over RNA in prebiotic environments. It specifically focuses on Wet-Dry Cycles within hydrothermal fields, incorporating mineral catalysis and pH-dependent hydrolysis rates.

This framework demonstrates how specific environmental pressures—such as hydration/dehydration cycles and mineral interfaces (Montmorillonite)—provide a selective advantage to Thymine-based polymers (DNA) over Uracil-based ones (RNA).

## Key Features (v3.1)
- Wet-Dry Dynamics: Continuous water activity (a_w) modeling with transitions between hydrated and dehydrated states.
- Multiprocessing Support: Parallel execution of 2D sensitivity grids for rapid research.
- pH-Dependent Kinetics: Integration of Arrhenius equations with acid-base catalysis and mineral protection factors.
- Stochastic Engine: Modified Gillespie-style simulation for discrete molecular events.
- Automated Optimization: Finds the "Evolutionary Sweet Spot" for DNA enrichment.

## Installation
Clone the repository and install the dependencies:
git clone https://github.com/mrhashemi2000/UPDSF_v3_1.git
cd UPDSF_v3_1

pip install -r requirements.txt

## Run the analysis
python UPDSF_v3_1.py

## Scientific Logic
The core selection pressure in this model is derived from the 43:1 RNA/DNA hydrolysis ratio observed during extreme dehydration. The framework explores a 2D landscape of:
- Temperature: 25°C to 60°C
- pH: 5.0 to 9.0
This framework is the computational implementation of the Matter World Hypothesis (MWH) and the Chemical Darwinism series, as detailed in the published perspective: Intelligence-Augmented (IA) Chemical Darwinism under the Matter World Hypothesis: A Theoretical Framework for the Origin of Life https://doi.org/10.5281/zenodo.18594133 .

## Results Visualization
The framework generates 9-panel diagnostic plots showing:
1. Thymine Enrichment Surfaces
2. DNA Fraction Heatmaps
3. Phase Dynamics (Wet/Dry/Transition)
4. Kinetic Stability Profiles

## Citation 
If you use this framework in your research, please cite it as: Hashemi, S. M. R. (2026). Unified Prebiotic DNA Selection Framework UPDSF_v3_1 
https://doi.org/10.5281/zenodo.21026496

## References

Ferris, J. P., Hill, A. R., Liu, R., & Orgel, L. E. (1996)
Synthesis of long prebiotic oligomers on mineral surfaces.
Nature, 381(6577), 59–61.
DOI: 10.1038/381059a0

Cleaves, H. J. (2010)
The origin of the biologically coded amino acids.
Journal of Theoretical Biology, 263(4), 490–498.
DOI: 10.1016/j.jtbi.2009.12.014

Wet-Dry Cycle Papers

Ross, D. S., & Deamer, D. W. (2016)
Dry/Wet Cycling and the Thermodynamics and Kinetics of Prebiotic Polymer Synthesis.
Life, 6(3), 28.
DOI: 10.3390/life6030028

Damer, B., & Deamer, D. (2020)
The Hot Spring Hypothesis for an Origin of Life.
Astrobiology, 20(4), 429–442.
DOI: 10.1089/ast.2019.2045
