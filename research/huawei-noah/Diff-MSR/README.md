# WSDM24_Diff-MSR
Code for Diff-MSR accepted to WSDM 2024. Take **FNN** as an example and the **Music** domain is the cold-start domain on **Douban** dataset.

**Stage 1**

**Goal**: Pre-train the backbone model on all domains and we get the FNN backbone model 'douban_fnn_train_v2_6.pt' in chkpt folder.

**Run**: python douban_main_all.py --model_name fnn --job 6

**Stage 2**

**Goal**: Train the diffusion model for the sparse domain, and We get two diffusion models for unclick and click data, i.e., 'fnn_douban_music_diff0_0.001_500_0.0002_other_pred_v_0_v2_2.pt' and 'fnn_douban_music_diff1_0.001_500_0.0002_other_pred_v_0_v2_2.pt' in chkpt folder.

**Run**: python douban_diff.py --indexx 0 --batch_size 512 --learning_rate 1e-3 --T 500 --beta 0.0002 --schedule other --objective pred_v --auto_normalize 0 --job 2 --model_name fnn

python douban_diff1.py --indexx 0 --batch_size 512 --learning_rate 1e-3 --T 500 --beta 0.0002 --schedule other --objective pred_v --auto_normalize 0 --job 2 --model_name fnn

**Stage 3**

**Goal**: Train the classifier and we get the 'fnn_douban_classifier_500_0.0002_other_pred_v_0_v2_2.pt' in chkpt folder.

**Run**: python douban_classifier_adversarial.py --beta 0.0002 --schedule other --objective pred_v --step 70 --T 500 --job 2 --model_name fnn --learning_rate 1e-3

**Stage 4**

**Goal**: Improve the model on the sparse domain. 

**Run**: python douban_augment_final_v3.py --indexx 0 --learning_rate 2e-3 --T 500 --beta 0.0002 --schedule other --objective pred_v --auto_normalize 0 --job 1 --step1 30 --step2 50 --model_name fnn
