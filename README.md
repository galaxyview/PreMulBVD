# PreMulBVD

### 1.Extract AST and pseudo-code from binary code (Decomplier)
You can use script `ASTExtraction/extract_all_ast.py` to extract AST and pseudo-code from binary code. (You need to set path options in the script.)

### 2.Packaging all generated databases into input data for the model (Parser)
You can use the script `graphcodebert_dataconclusion.py` to preprocess the AST and pseudo-code data, which includes normalizing the pseudo-code and transforming the AST into a graph structure. Finally, the script will generate a `.pth` file available for model training after the run is complete.

### 3.Training model (Model)
(1) The script `graphcodebert_model.py` contains the fine-tuned model and the dataset model used in this paper. 

(2) The script `graphcodebert_train.py` are used to train and test the model. 

(3) The scripts `graphcodebert_train4onlyast.py` and `graphcodebert_train4onlycode.py` are in the application to the RQ2 ablation experiments, which can be trained using only AST and pseudo-code, respectively.

(4) The script `graphcodebert_validate.py` is used to test the model.
