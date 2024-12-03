import mindspore as ms
import mindspore.ops as ops
from can import build_loss

ms.set_context(mode=ms.PYNATIVE_MODE, pynative_synchronize=True)


def test_can_loss():
    """
        This test case assumes some network output tensors to test 
        whether the model loss function can perform properly.
    """

    # Simulate some data, Suppose batch_size = 1, seq_length = 10, vocab_size = 111
    batch_size = 1
    seq_length = 10
    vocab_size = 111

    # Simulate preds
    word_probs = ops.randn((batch_size, seq_length, vocab_size))
    counting_preds = ops.randn((batch_size, vocab_size))
    counting_preds1 = ops.randn((batch_size, vocab_size))
    counting_preds2 = ops.rand((batch_size, vocab_size))

    # Simulates labels and labels_mask
    labels = ops.randint(low=0, high=vocab_size, size=(batch_size, seq_length), dtype=ms.int32)
    labels_mask = ops.randn((batch_size, seq_length)) > 0.5
    labels_mask = labels_mask.astype('float32')

    # Construct batch data
    batch = [labels, labels_mask]

    # Create a CANLoss instance
    loss_fn = build_loss("CANLoss")

    preds=dict()
    preds["word_probs"]=word_probs
    preds["counting_preds"]=counting_preds
    preds["counting_preds1"]=counting_preds1
    preds["counting_preds2"]=counting_preds2

    # Call the forward method
    loss_dict = loss_fn(preds, *batch)

    assert isinstance(loss_dict, ms.Tensor), f"loss_dict is incorrect."
    assert isinstance(loss_dict.item(), (int, float)), f"The variable {loss_dict} is not a number."


if __name__ == "__main__":
    test_can_loss()
