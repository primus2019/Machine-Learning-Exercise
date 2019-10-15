# README

This is the illustrative document for exercise 2 of Machine Learning Course. 

## Content

- files
- network
- environment
- run

---

### Files in the package

1. *Exercise2_20190919.pdf* - The assignment file.
2. *MLP_with_BP.py* - The running file.
3. *network.py* - The MLP class file.
4. *utils.py* - The util functions file.
5. *exercise_2_ML\_季林成\_2017012775\_经71.pdf* - The experiment report file.
6. Given dataset files
   1. *train_10gene.csv*
   2. *train_10gene_sub.csv*
   3. *train_label.csv*
   4. *train_label_sub.csv*
   5. *test_10gene.csv*
   6. *test2_10gene.csv*
   7. *test_label.csv*
   8. *test2_label.csv*
7. Preprocessing dataset files
   1. *unquoted_(train_10gene/train_10gene_sub/train_lable/train_label_sub/test_10gene/test_label/test2_10gene/test2_label).csv*
   2. *transposed_unquoted_(train_10gene/train_10gene_sub/train_lable/train_label_sub/test_10gene/test_label/test2_10gene/test2_label).csv*
8. *lr_train_1/2_test_1/2(_e300).log* - train log files
9. *lr_train_1/2_test_1/2(_e300).png* - screenshot files
10. README.md

---

### Network structure

The network is a three-layered MLP with BP algorithm, seeing *network.py*.

The functions includes forward() backward() and test()

##### The forward function:

$$\rm \textbf{x}_1 = [^\textbf{1}_\textbf{x}] $$

$$\rm \textbf{a}_1 = \textbf{w}_1 \dot{} \textbf{x}_1 $$

$$\rm \textbf{y}_1 = \textbf{sig}(\textbf{a}_1) $$

$$\rm \textbf{x}_2 = [^\textbf{1}_{\textbf{y}_1}] $$

$$\rm \textbf{a}_2 = \textbf{w}_2 \dot{} \textbf{x}_2 $$

$$\rm \textbf{y}_2 = \textbf{sig}(\textbf{a}_2) $$

$$\rm \textbf{x}_3 = [^\textbf{1}_{\textbf{y}_2}] $$

$$\rm \textbf{a}_3 = \textbf{w}_3 \dot{} \textbf{x}_3 $$

$$\rm \textbf{y}_3 = \textbf{sig}(\textbf{a}_3) $$

Then returns $\rm \textbf{y}_3 $.

##### The backward function

$$\rm dw_3=learning\ rate*\frac{dw_3}{da_3}\dot{}\frac{da_3}{dy_3}\dot{}\frac{dy_3}{de}\dot{}\frac{de}{dE}$$

$$\rm w_3 = w_3 - dw_3 $$

Adjust $\rm w_3, w_2,$and $\rm w1 $.

##### The test function

Same as the forward function except that no self data is changed.

---

### Programming environment

Python 3.7.x(stable), Numpy, Pandas

---

### How to run the code

1. open *MLP\_with\_BP.py*
2. search for "train_data" (or train\_target, test\_data, test\_target, learning\_rate, epochs) to locate the main code block at the end
3. change the setting with new ones
4. run *MLP\_with\_BP.py*
5. NOTE: when warning info "parell table not identical" appears, please run the code again