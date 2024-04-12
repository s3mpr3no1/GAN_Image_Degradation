# GAN Image Degradation
Exploration of GAN resilience to training set degradation

## Steps 

Install Miniconda 

Open Anaconda Prompt and enter the following

	conda create --name tf python=3.9
	conda activate tf
	conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0

	pip install --upgrade pip
	pip install tensorflow==2.10.1
	pip install -r requirements.txt

Change the following paths in the python files to where you're storing data

	Execute_Experiment.py 	Line 22
	cGAN_BigBatch.py 	Line 45 (optional - default value)
Change the following path to point to the parent folder

	ArtData.py 		Line 28

Change the following list to include the desired degradations

	Execute_Experiment.py	Line 29

Change the following list to include the desired seeds (should be [21, 42, 123, 666, 420])

	Execute_Experiment.py	Line 33

Run the following command

	python Execute_Experiment.py
