# README

**Group 04**
* 112527005: 李岳峻
* 112527008: 尤力民
* 112527011: 蔡睿芸

## How to run our code?

### 1. Install the requirement packages with certain python version
   ```
   # for python 3.8
   pip install numpy==1.20.1
   pip install pandas==1.2.3
   pip install scikit-learn==0.23.2
   pip install scipy==1.5.3
   pip install sklearn==0.0
   pip install torch==1.7.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html
   pip install torch-cluster==1.5.9 -f https://data.pyg.org/whl/torch-1.7.1+cu110.html
   pip install torch-geometric==1.7.1
   pip install torch-scatter==2.0.7 -f https://data.pyg.org/whl/torch-1.7.1+cu110.html
   pip install torch-sparse==0.6.9 -f https://data.pyg.org/whl/torch-1.7.1+cu110.html
   pip install torch-spline-conv==1.2.1 -f https://data.pyg.org/whl/torch-1.7.1+cu110.html
   pip install torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
   pip install torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html
   pip install matplotlib
   pip install langchain-community
   pip install huggingface_hub
   ```
   
### 2. Create dataset for training & testing 
**(Please make your source directory is under the folder `PS2`)**
   ```
   python test_raw.py
   python test_process.py
   ```

### 3. Run training steps 
**(Please make your source directory is under the folder `PS2`)**
   ```
   python test_run.py
   ```
   
### 4. Recommendation

+ **Beverage recommendation**
    + Step:1 User name input for beverage recommendation 
    (Please make your source directory is under the folder `PS2`)
      ```
      python create_user_data.py
      
      ##Then enter the user name you want to predict, it will generate the graph information
      ```
   + Step:2 Model prediction for beverage recommendation 
   (Please make your source directory is under the folder `PS2`)
     ```
     python ps2_testing.py
     
     ##it will generate the output according the user's comment
     ```

+ **Beverage Shop Recommendation**
    + Beverage Shop Recommendation 
    (Please make your source directory is under the folder `PS2`)
      ```
      python recommand.py
      
      ## it will read the user name as input and will generate the recommendation output for specific user.
      ```

## Final Version
20240612 https://drive.google.com/drive/folders/11Dv4itrVBEubCe8MA7P-BUg73Blj00Gi?usp=sharing updated by Limit