# Create a virtual environment with Conda (and install Python)
- Install miniconda (or Anaconda)
  - Download the installer for your operating system at: https://docs.conda.io/projects/miniconda/en/latest/
- Only MacOS: execute the following commands in terminal:
  - `~/miniconda3/bin/conda init bash`
  - `~/miniconda3/bin/conda init zsh`
- Only Windows: run "Anaconda Prompt (miniconda3)"
- Create a new virtual environment and activate it with the following commands (in terminal on MacOS and Anaconda Prompt on Windows)
  - `conda create -n myenvname python=3.13`
  - `conda activate myenvname`

# Run the experiment
- Check if you have git installed
  - `git --version`
- In case you don't, install git in the virtual environment
  - `pip install git`
- Clone the git repository to a directory of your choosing
  - Navigate to the directory: e.g., cd Users/max/Downloads (MacOS)
  - Clone the directory with `git clone https://github.com/Mittenbuhler/teamproject_26_muzero.git`
- Navigate to this directory
  - `cd path_name` (e.g., `Users/max/Desktop/teamproject_26_muzero` for MacOS)
- Install the required packages (make sure that the virtual environment is active)
  - `pip install -r requirements.txt`