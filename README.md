# Multispecies-Occupancy-model-using-Turing
Rota et al (2016) multispecies occupancy model using Turing

## What is this?

This repository contains the Julia code to implement the multispecies occupancy model published by [Rota et al (2016)](https://besjournals.onlinelibrary.wiley.com/doi/full/10.1111/2041-210X.12587) in Methods in Ecology and Evolution.
The Stan code used by the authors to fit the co-occurrence probability of four carnivore species has been adapted to Julia language in order to fit simulated data on two arbitrary species. The Stan code can be accessed [here](https://datadryad.org/stash/dataset/doi:10.5061/dryad.pq624) ("model3.stan" file).

### DataSimulation.jl
DataSimulation.jl generates detections history for two species at 100 sites surveyed 10 times, as well as all the inputs required by the model. This file must be run first.
### MultispeciesOccupancyModel.jl 
MultispeciesOccupancyModel.jl fits Rota et al's multispecies model to the simulated data. The natural parameters are estimated as:

f<sub>1</sub> = x'<sub>&alpha;</sub> * &alpha;

f<sub>2</sub> = x'<sub>&beta;</sub> * &beta;

f<sub>12</sub> = &gamma;

where x'<sub>&alpha;</sub> and x'<sub>&beta;</sub> are vectors of covariates and &alpha;, &beta;, and &gamma; are vectors of parameters (see Rota et al 2016 for more details). Also, species detectability is not assumed to be constant and it is estimated as a function of covariates.
