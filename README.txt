Must be on the Greene Cluster due to the memory of the GPUs. BigPurple does not have GPUs with enough memory to run this pipeline. 

Once on Greene, set up a folder structure that follows this outline: 

LocationForSingluarityContainer
	|- SingularityContainer.ext3
ExperimentalTopLevel
	|- inputs
	|	|- Data
	|	|	|- SampleFolder1
	|	|	|	|- TargetImage.tif
	|	|	|  ...
	|	|	|- SampleFolderN
	|	|	|	|- TargetImage.tif
	|	|- samples.csv
	|- output
	|- scripts
	|	|- PythonCode.py
	|	|- ShellScriptToRunPythonCode.sh
	
This is the folder structure used to run the attached files (located in the scratch directory):

practenv
	|- nspytorch.ext3
lionnet
	|- inputs
	|	|- train3D
	|	|	|- C1-FB323A_CSC_Rd1_1
	|	|	|	|- gap.tif
	|	|	|	|- image.tif
	|	|	|	|- label.tif	
	|	|	|  ...
	|	|	|- C1-FB323A_CSC_Rd1_23
	|	|	|	|- gap.tif
	|	|	|	|- image.tif
	|	|	|	|- label.tif
	|	|- samples.csv
	|- output
	|- scripts
	|	|- 3D_Unet.py
	|	|- 3D_Unet_run.sh

Add the included samples.csv file and the input image files to the inputs folder (not included). 

Set up a Singularity container in the corresponding folder following the instructions from the NYU High Performance Computing site: https://sites.google.com/nyu.edu/nyu-hpc/hpc-systems/greene/software/singularity-with-miniconda 

Create a conda environment within the container and use the 3D_Unet_env.txt file to populate the environment with the necessary packages. 

Add the 3D_Unet_run.sh and 3D_Unet.py scripts to the scripts folder. Modify the 3D_Unet_run.sh so it points to the correct Singularity container and so that the three command line arguments are correct. The first is the output folder, the second is the number of epochs to run, the third is the learning rate to use.  

Use SBATCH to run the 3D_Unet_run.sh script. 
	
