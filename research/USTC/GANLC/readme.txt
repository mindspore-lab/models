This code is implemented by the framework of mindspore for the paper "Solving Low-Dose CT Reconstruction via GAN with Local Coherence"

The dataset of mayor clinic can be downloaded from  https://drive.google.com/drive/folders/1SHN-yti3MgLmmW_l0agZRzMVtp0kx6dD

The models.py concludes our models of generator, discriminator and optical estimator.

The main.py is our main documents, and it concludes train and inference.
For example, we can train as
" python main.py --mode train --max_epochs 10  --input_medical_dir ./download/mayor_clinic"
and test as:
" python main.py --mode inference --input_medical_dir ./download/mayor_clinic --g_checkpoint generator.ckpt --f_checkpoint fnet.ckpt"
