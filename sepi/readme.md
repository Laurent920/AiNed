sepi_train.ipynb        ← Step 1: train with asynctorch, save .pt checkpoint



sepi_convert.py         ← Step 2: convert .pt → .json for the SEPI engine



tmlr_sepi_conf.yaml     ← Step 3: configure SEPI (architecture, inference params)


sepi_tmlr_28052026.py   ← Step 4: run async inference (or training) with JAX + MPI
