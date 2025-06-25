# QSEA: Quantum Self-supervised Learning with Entanglement Augmentation
- **Thanks for the support provided by MindSpore Community**
<img src=https://github.com/Lingxiao-Li61/models/blob/master/research/arxiv_papers/QSEA/Fig/1.jpg width=83% />


## 1. Conda Environment

- python 3.8
- mindspore 2.2
- mindformers 1.1.0
- `pip install -r requirements.txt`

## 2. Datasets

The datasets used in this paper are: 
- MNIST (https://www.kaggle.com/datasets/hojjatk/mnist-dataset)
- Fashion-MNIST (https://github.com/fendy07/FMNIST-DeepLearning)
- CIFAR_10 (https://www.cs.toronto.edu/~kriz/cifar.html).

## 3. Model:
The specific code of the model is as follows:
```
class HQSSL(nn.Cell):
    def __init__(self):
        super().__init__()
        self.sma_circ = build_sma_circuit()
        self.proj_circ = build_proj_circuit()

        self.theta_sma = ms.Parameter(ms.Tensor(np.random.rand(), ms.float32), name='theta_sma')
        self.sma_params = ms.ParameterTuple([self.theta_sma])

        self.proj_params = ms.ParameterTuple([
            ms.Parameter(ms.Tensor(np.random.rand(), ms.float32), name=name)
            for name in self.proj_circ.params_name
        ])
        self.proj_params_dict = {name: param for name, param in zip(self.proj_circ.params_name, self.proj_params)}

        self.encoder = ClassicEncoder()
        self.noise_strength = ms.Parameter(ms.Tensor(0.01, ms.float32))

    def construct(self, x):
        raw_amp = self.encoder(x)
        batch_size = int(raw_amp.shape[0])
        feature_dim = int(raw_amp.shape[1])
        n_raw = int(np.log2(feature_dim))
        n_rand = n_raw
        n_total = n_raw + n_rand

        rand_circ = Circuit()
        for i in range(n_rand):
            rand_circ += H.on(i)
            rand_circ += RX(np.random.uniform(0, np.pi)).on(i)

        enhance_circ = Circuit()
        for i in range(n_raw):
            control = i
            target = i + n_rand
            enhance_circ += X.on(target, control)
            enhance_circ += RY(f'ry_{i}').on(target)
            enhance_circ += X.on(target, control)

        if not hasattr(self, 'enhance_params'):
            self.enhance_params = ms.ParameterTuple([
                ms.Parameter(ms.Tensor(np.random.rand(), ms.float32), name=f'ry_{i}')
                for i in range(n_raw)
            ])
            self.enhance_param_dict = {f'ry_{i}': p for i, p in zip(range(n_raw), self.enhance_params)}

        outputs = []
        for i in range(batch_size):
            psi_raw = raw_amp[i].asnumpy().astype(np.complex64)
            psi_raw = psi_raw / np.linalg.norm(psi_raw)

            sim_rand = Simulator('mqvector', n_rand)
            zero_state = np.zeros(2 ** n_rand, dtype=np.complex64)
            zero_state[0] = 1.0
            sim_rand.set_qs(zero_state)
            sim_rand.apply_circuit(rand_circ)
            psi_rand = sim_rand.get_qs()

            psi_joint = np.kron(psi_raw, psi_rand)
            sim = Simulator('mqvector', n_total)
            sim.set_qs(psi_joint)
            noisy_enhance_circ = apply_complex_noise(enhance_circ, n_total, noise_prob_dict)
            sim.apply_circuit(apply_complex_noise(enhance_circ, n_total, noise_prob_dict), self.enhance_param_dict)

            psi_enhanced = sim.get_qs()

            psi_reduced = extract_front8_marginal(psi_enhanced)
            outputs.append(psi_reduced)

        proj_pos = ms.Tensor(np.stack(outputs), dtype=ms.float32)

        rand_amp = ops.standard_normal((batch_size, feature_dim)).astype(ms.float32)
        rand_amp_norm = ops.sqrt(ops.ReduceSum(keep_dims=True)(ops.pow(rand_amp, 2), 1))
        rand_amp = rand_amp / (rand_amp_norm + 1e-8)

        proj_neg_amp = self.apply_circuit_complex(rand_amp, self.proj_circ, self.proj_params_dict)
        proj_neg_np = proj_neg_amp.asnumpy()
        proj_neg_prob = np.abs(proj_neg_np) ** 2
        proj_neg = ms.Tensor(proj_neg_prob, dtype=ms.float32)

        return proj_pos, proj_pos, proj_neg, self.noise_strength  
    def apply_circuit_complex(self, state, circuit, params):
        state_np = state.asnumpy().astype(np.complex64)
        if len(state_np.shape) == 1:
            state_np = state_np.reshape(1, -1)

        n_qubits = int(np.log2(state_np.shape[1]))
        outputs = []

        for i in range(state_np.shape[0]):
            sim = Simulator('mqvector', n_qubits)
            sim.set_qs(state_np[i])

            noisy_circ = apply_complex_noise(circuit, n_qubits, noise_prob_dict)
            sim.apply_circuit(noisy_circ, params)

            outputs.append(sim.get_qs())

        return ms.Tensor(np.stack(outputs), dtype=ms.complex64)
```

## 4. Training and testing:
The model will go through 100 epochs and be tested after every 5 steps to get the performance.
```
python qsea.py
```

## 5. Cite our work:
```
@article{li2025QSEA,
  title     = {QSEA: Quantum Self-supervised Learning with Entanglement Augmentation},
  author    = {LingXiao Li, XiaoHui Ni, Jing Li, SuJuan Qin, and Fei Gao},
  journal={arXiv preprint arXiv:2506.10306},
  year={2025}
```
