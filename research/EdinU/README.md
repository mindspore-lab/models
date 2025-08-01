# Banyan: A MindSpore Implementation

This is a **MindSpore** implementation of the ICML 2025 paper [Banyan: Improved Representation Learning with Explicit Structure]. It contains the code for **evaluation** of the Banyan model on the Ascend platform.

If you are just interested in the implementation of the model, it is defined in:

- `src/models.py` (contains the core Banyan code, migrated to MindSpore)
- `src/funcs.py` (contains the diagonal message passing functions)
- `src/model_utils.py` (contains helper functions for the running encoder portion of the model)

## 📑 Development Plan

Here is the development plan of the project:

- Core Model:
    - [x] Inference
    - [x] Evaluation
    - [ ] Training
- Intrinsic Evaluations:
    - [x] SST Evaluation
    - [x] STS/STR Evaluation
- Extrinsic Evaluations:
    - [x] Classification (SentEval)
    - [x] Retrieval (BEIR)
- Platform Support:
    - [x] Ascend NPU Support
    - [ ] Distributed Training (Data Parallel)
    - [ ] Graph Kernel Fusion optimization

## 📦 Requirements

</div>
1.  Install requirements
`conda env create -f requirements.yaml`

## 🚀 Getting Started

### Get Data + Tokenizers

Create a `data` directory.
Download data from https://huggingface.co/mxuax/Banyan_model_weight_data
We support the following languages:

- Afrikaans: af
- Amharic: am
- Arabic: ar
- English: en
- Spanish: es
- Hausa: ha
- Hindi: hi
- Indonesian: id
- Marathi: mr
- Telugu: te

Most of the test datasets are already in the `data` directory. However, to replicate the results in the paper you will need to get the pre-training corpora we used. Run the following to do so, you can edit the list in the file if you are only interested in a subset of languages:

`python get_data.py`

You will also want to download the relevant tokenizers from the BPEmb package (again edit the list if you only want a subset), to do so run:

`python get_tok.py`

Finally, to run the retrieval eval for English, we need to get the relevant datasets from the BEIR package:

`python get_beir.py`

### Checkpoints

In this repo, we also released the Checkpoints for en, af and te. Other languages are coming soon.
[For the Checkpoints]: https://huggingface.co/mxuax/Banyan_model_weight_data

## Intrinsic Evaluations 

Lexical and STS/STR evaluations will be run automatically during training. In our mindspore implementation, we decompose it from the training code. To get the result, just run:

`GLOG_v=3 python eval.py ../checkpoints/te_ms.pt --lang te`

`GLOG_v=3` is used to omit the warning and the `--lang te` should also be included since the default language is en.

## 📈 Extrinsic Evaluation (English)

However, if you want to run our extra classification and retrieval evals for English you'll need a pre-trained model you saved somewhere (e.g., in the `checkpoints` folder).

For retrieval eval, navigate to the `src` directory and use `retrieval.py`. The script expects you to specify a path to your saved MindSpore model. So for example if you had saved it under `en_ms.ckpt` in the checkpoints directory, you would run:

`GLOG_v=3 python retrieval.py ../checkpoints/en_ms.ckpt`

Similarly, for classification eval you can run:

`GLOG_v=3 python classify.py ../checkpoints/en_ms.ckpt`

## Citation

<pre>
@article{opper2024banyan,
  title={Banyan: Improved Representation Learning with Explicit Structure},
  author={Opper, Mattia and Siddharth, N},
  journal={arXiv preprint arXiv:2407.17771},
  year={2024}
}
</pre>

## Acknowledgements

We would like to thank the authors of the original [Banyan] paper for their excellent work and for open-sourcing the code. We also thank the MindSpore team and community for their support in developing and maintaining the framework that made this migration possible.

## License

Banyan is available under Apache 2.0.

[Banyan: Improved Representation Learning with Explicit Structure]: https://arxiv.org/abs/2407.17771
[official instructions]: https://www.mindspore.cn/install/
[official documentation]: https://www.mindspore.cn/docs/en/r2.5.0/model_train/parallel/msrun_launcher.html
