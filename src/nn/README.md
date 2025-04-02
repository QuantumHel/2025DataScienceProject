# Permutation-based Approach

1. The code includes several encoders, including Transformer-based encoder, which makes training rather time-consuming and does not improve enough.

   **The current configuration is highly recommended: CNN-based encoder + Transformer-based decoder.**

2. The class `TableauPermutationDataset` is flexible. Please refer to <https://github.com/traffictse/2025DataScienceProject/blob/94e9b64160145f40ae8ba605f8951cbce3266f38/src/nn/permutation_math.py#L697-L715> for its example usage.
   `training_data_perm.pkl` contains 3200 4-qubit data points, while `training_data_perm_4_qubit.pkl` contains 9600 4-qubit data points.
   **By experiments, it is sufficient to train with 3200 data points with 50 epochs, which takes around 20 minutes or even fewer** (See `src/loss_history3-4.png`). If training with 9600 data points, it would be better to increase to 100 epochs or 200 epochs (See `src/loss_history3-4.png`).
3. `permutation_math.py` contains mixed code for **Model 2** and **Model 3**.
   There are 2 example usage of **Model 3** at the end of `permutation_math.py`. (See <https://github.com/traffictse/2025DataScienceProject/blob/94e9b64160145f40ae8ba605f8951cbce3266f38/src/nn/permutation_math.py#L1473-L1492>). I usually uncomment the example usage code and simply run `permutation_math.py` for **fast debugging**.
   These 2 example usage illustrate using **Model 3** without and with supervised-learning-based fine-tuning. **Without SL fine-tuning is recommended** as fine-tuning takes time and basically does not improve.
   There is a function of RL fine-tuning but commented out, because it is not ready to use and sortof put aside.
   Remember to comment out the example usage code in `permutation_math.py` when running `nn_eval_main.py`, otherwise an error will be raised.
4. If deciding to use SL fine-tuning, please remember to use different datasets for pre-training and fine-tuning respectively (See https://github.com/traffictse/2025DataScienceProject/blob/94e9b64160145f40ae8ba605f8951cbce3266f38/src/nn_eval_main.py#L169-L177) and **call the right model** in `nn_eval_main.py` (See <https://github.com/traffictse/2025DataScienceProject/blob/94e9b64160145f40ae8ba605f8951cbce3266f38/src/nn_eval_main.py#L205-L211>).
5. The loss history will by default be **plotted over epochs not over batches**, saved as `.png` and `.pkl`, and the model will be saved as `.pth`.
6. `dummy_perm_data_gen.py` is to generate training data in the desired format for this approach. All needed to do is to specify the qubit number here (See <https://github.com/traffictse/2025DataScienceProject/blob/94e9b64160145f40ae8ba605f8951cbce3266f38/src/nn/dummy_perm_data_gen.py#L52-L54>) and run `dummy_perm_data_gen.py`. The training data will be saved like `training_data_perm_4_qubit.pkl`. It would roughly take 30-40 hours to generate 3200 5-qubit data points, which I terminated in the haflway.
7. It is recommended to test initial performance over just 50-100 valuations. Just for time saving.
8. Good luck!
