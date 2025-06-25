import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.dataset import vision, MnistDataset
from mindquantum import Circuit, RY, H, Simulator, QubitOperator, RX, X
from mindquantum.core.circuit import Circuit
from mindspore import grad
from mindquantum.core.parameterresolver import ParameterResolver


N_QUBITS = 8
BATCH_SIZE = 12
INPUT_DIM = 784
HIDDEN_DIM = 256
TRAIN_SET_SIZE = 50
SEED = 42

USE_NOISE = True 
#USE_NOISE = False
ms.set_seed(SEED)
np.random.seed(SEED)

import itertools

def partial_trace(rho, keep, dims):

    # total = len(dims)
    total = int(np.log2(rho.shape[0]))
    traced = [i for i in range(total) if i not in keep]

    reshaped_rho = rho.reshape([2] * total * 2)  # 将 2^n x 2^n reshape 为 [2]*n + [2]*n
    idx_keep = keep
    idx_traced = traced

 
    for i, k in enumerate(sorted(idx_traced)):
        reshaped_rho = np.trace(reshaped_rho, axis1=k - i, axis2=k - i + total - i)

    final_shape = 2 ** len(keep)
    return reshaped_rho.reshape((final_shape, final_shape))

def apply_complex_noise(circ, n_qubits, prob_dict):
    from mindquantum.core.gates import X, Y, Z, RX, RY
    import random

    noisy_circ = Circuit()

    for gate in circ: 
        noisy_circ += gate

        if USE_NOISE:  
            rand_prob = random.random()

            # Depolarizing Noise
            if rand_prob < prob_dict['depolarizing']:
                
                noisy_circ += random.choice([X, Y, Z]).on(random.choice(range(n_qubits)))

            # Bit Flip Noise
            elif rand_prob < prob_dict['bit_flip']:
                noisy_circ += X.on(random.choice(range(n_qubits)))  

            # Phase Flip Noise
            elif rand_prob < prob_dict['phase_flip']:
                noisy_circ += Z.on(random.choice(range(n_qubits))) 

            # Amplitude Damping Noise
            elif rand_prob < prob_dict['amplitude_damping']:
              
                noisy_circ += RX(np.random.uniform(0, np.pi)).on(random.choice(range(n_qubits)))

            # Phase Damping Noise
            elif rand_prob < prob_dict['phase_damping']:
                
                noisy_circ += RY(np.random.uniform(0, np.pi)).on(random.choice(range(n_qubits)))

            # Custom Noise
            elif rand_prob < prob_dict['custom']:
                noisy_circ += RX(np.random.uniform(0, 2 * np.pi)).on(random.choice(range(n_qubits)))

    return noisy_circ


# -----------------------------------------------------------------------------------
def prepare_datasets(data_dir):
    def _trans(data):
        data = vision.Resize((28, 28))(data)
        data = vision.HWC2CHW()(data)
        data = vision.Rescale(1 / 255., 0)(data)
        data = data.astype(np.float32).flatten()
        return data

    train = MnistDataset(data_dir, 'train').map(_trans, 'image')
    test = MnistDataset(data_dir, 'test').map(_trans, 'image')

    train = train.shuffle(10000).take(TRAIN_SET_SIZE).batch(BATCH_SIZE)
    test = test.batch(BATCH_SIZE)

    return train, test


# -----------------------------------------------------------------------------------

class ClassicEncoder(nn.Cell):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Dense(INPUT_DIM, HIDDEN_DIM)
        self.fc2 = nn.Dense(HIDDEN_DIM, 2 ** N_QUBITS)  
        self.pow = ops.Pow()
        self.sum = ops.ReduceSum(keep_dims=True)

    def construct(self, x):
        x = self.flatten(x)
        x = ops.ReLU()(self.fc1(x))
        raw_features = self.fc2(x)
        norm = ops.sqrt(self.sum(self.pow(raw_features, 2), 1))
        return raw_features / norm


# -----------------------------------------------------------------------------------
def build_sma_circuit(n_qubits=N_QUBITS):
    circ = Circuit()
    theta = ParameterResolver('theta_sma') 
    for i in range(n_qubits):
        circ += RY(theta).on(i)  
    return circ


# -----------------------------------------------------------------------------------

def build_proj_circuit(n_qubits=N_QUBITS):
    circ = Circuit()
    for i in range(n_qubits):
        circ += RX(f'rx_{i}').on(i)  
        circ += RY(f'ry_{i}').on(i)
    for i in range(0, n_qubits - 1, 2): 
        circ += X.on(i + 1, i)
    return circ

noise_prob_dict = {
    'depolarizing': 0.1,    
    'bit_flip': 0.2,        
    'phase_flip': 0.2,     
    'amplitude_damping': 0.1,  
    'phase_damping': 0.1,   
    'custom': 0.3           
}
# -----------------------------------------------------------------------------------

class QuantumLayer(nn.Cell):
    def __init__(self, circuit):
        super().__init__()
        self.circ = circuit
        self.n_qubits = circuit.n_qubits
        self.sim = Simulator('mqmatrix', self.n_qubits)

    def construct(self, params):
        state = self.sim.get_state(params)
        return state


# -----------------------------------------------------------------------------------

class EntropyLoss(nn.Cell):
    def __init__(self, subsystems=[0, 1, 2, 3]):
        super().__init__()
        self.subsystems = subsystems

    def construct(self, state):
        batch_size = state.shape[0]
        entropies = []

        for s in state:
            s_np = s.asnumpy()
            dim = s_np.shape[0]
            n_qubits = int(np.log2(dim))
            dims = [2] * n_qubits

            rho = np.outer(s_np, s_np.conj())

            eigenvals = np.linalg.eigvalsh(rho_A)
            eigenvals = eigenvals[eigenvals > 1e-12] 
            entropy = -np.sum(eigenvals * np.log2(eigenvals))
            entropies.append(entropy)

        return ops.ReduceMean()(ms.Tensor(entropies, ms.float32))

def extract_front8_marginal(psi_enhanced):
    prob = np.abs(psi_enhanced) ** 2
    front8_prob = np.zeros(2 ** 8, dtype=np.float32)
    for idx in range(len(prob)):
        front8_index = idx >> 8  
        front8_prob[front8_index] += prob[idx]
    return front8_prob

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


class HQSSL_Loss(nn.Cell):
    def __init__(self):
        super().__init__()
        self.entropy_loss = EntropyLoss()

    def construct(self, raw, proj_pos, proj_neg, sigma):
        sim = ops.ReduceSum()(proj_pos * proj_neg, axis=1)
        contrastive_loss = -ops.ReduceMean()(ops.log(sim + 1e-8))

        entropy_pos = self.entropy_loss(proj_pos)
        entropy_neg = self.entropy_loss(proj_neg)
        qe_loss = entropy_pos - entropy_neg

        ada_loss = contrastive_loss * sigma

        return contrastive_loss + 0.1 * qe_loss + 0.05 * ada_loss

def train_hqssl(data_dir, epochs=50, train_labels=None):
    train_set, test_set = prepare_datasets(data_dir)

    if train_labels is None:
        train_labels = []
        for _, labels in train_set.create_tuple_iterator():
            train_labels.append(labels.asnumpy())
        train_labels = np.concatenate(train_labels)[:TRAIN_SET_SIZE]

    model = HQSSL()
    loss_fn = HQSSL_Loss()

    all_params = list(model.encoder.trainable_params()) + \
                list(model.sma_params) + \
                list(model.proj_params) + \
                [model.noise_strength]
    optimizer = nn.Adam(all_params, learning_rate=1e-4)

    def forward_fn(images):
        raw, proj_p, proj_n, sigma = model(images)
        loss = loss_fn(raw, proj_p, proj_n, sigma)
        return loss

    grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters)



    for epoch in range(epochs):
        print(" --------- Epoch:", epoch)
        model.set_train()
        total_loss = 0
        for images, _ in train_set.create_tuple_iterator():
            loss, grads = grad_fn(images)
            optimizer(grads)
            total_loss += loss.asnumpy()

        if (epoch + 1) % 5 == 0:
            train_feats = []
            for images, _ in train_set.create_tuple_iterator():
                feats = model.encoder(images).asnumpy()
                train_feats.append(feats)
            train_feats = np.concatenate(train_feats)

            from sklearn.svm import LinearSVC
            classifier = LinearSVC(max_iter=2000)
            classifier.fit(train_feats, train_labels)

            test_feats = []
            test_labels = []
            for images, labels in test_set.create_tuple_iterator():
                feats = model.encoder(images).asnumpy()
                test_feats.append(feats)
                test_labels.append(labels.asnumpy())

            test_feats = np.concatenate(test_feats)
            test_labels = np.concatenate(test_labels)
            test_acc = classifier.score(test_feats, test_labels)

            print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}, "
                  f"Noise: {model.noise_strength.asnumpy():.4f}, "
                  f"Test Acc: {test_acc * 100:.2f}%")


if __name__ == "__main__":
    data_dir = "./MNIST"

    test_model = HQSSL()
    dummy_input = ms.Tensor(np.random.randn(32, 784), ms.float32)
    out = test_model(dummy_input)
    print(f"raw_amplitude: {out[0].shape}")
    print(f"proj_pos: {out[1].shape}")
    print(f"proj_neg: {out[2].shape}")
    print(f"noise_strength: {out[3]}")

    train_hqssl(data_dir, epochs=100)
