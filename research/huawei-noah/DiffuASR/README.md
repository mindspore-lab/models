# Diffusion Augmentation for Sequential Recommendation

This is the implementation of the paper "Diffusion Augmentation for Sequential Recommendation".

You can implement our model according to the following steps:

1. The handle dataset should be put into ``./data/<dataset>/handled/``
2. Install the necessary packages. Run the command:

   ```
   pip install -r requirements.txt
   ```
3. To get train the DiffuASR and augment the corresponding dataset, please run the command:

   ```
   bash ./experiments/diffusion.bash
   ```
4. Finally, you can run the following bash to test the augmention performance for Bert4Rec:

   ```
   bash ./experiments/bert4rec.bash
   ```
