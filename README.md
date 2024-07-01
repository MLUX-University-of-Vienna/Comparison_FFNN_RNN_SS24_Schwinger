# BachelorThesis

## Prerequisites needed to run the .ipynb files:
- Python 3.12.2
- A dataset in JSON format to train the model

## Installation
1. Open a terminal and navigate to the folder (`cd /path/to/save`), where you want to save the project.
2. Clone the repository via the command `git clone https://github.com/RogueRefiner/BachelorThesis.git`
3. Navigate to the root directory of the cloned repository `cd BachelorThesis`
4. Install all necessary modules to run the project via the command `pip install -r requirements.txt`  

## How to run the project:
1. Update the `path_to_dataset` in line 16 of a main python file to the path to your dataset. <br>
    The main files are:   
    * smallFNN.py <br>
    * largeFNN.py <br>
    * smallRNN.py <br>
    * largeRNN.py <br>
2. Launch a terminal and navigate to the folder with the main files (normally `cd /path/to/save/BachelorThesis`).
3. Execute a main file with the command `python <filename>` e.g. `python smallFNN.py`
