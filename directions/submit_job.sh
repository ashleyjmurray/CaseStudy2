#!/bin/bash                                                                                                                                                                                                                                                                     
#SBATCH --account=sta440-f20                                                                                                                                                                                                                                                    
#SBATCH -p common                                                                                                                                                                                                                                                               
#SBATCH -N1                                                                                                                                                                                                                                                                     
#SBATCH -c1                                                                                                                                                                                                                                                                     
#SBATCH --mem=50G                                                                                                                                                                                                                                                               

module load Python/3.8.1
python all_code.py
