import argparse

import cv2
import mindspore_lite as mslite
import numpy as np
from matplotlib import pyplot as plt

from segment_anything.dataset.transform import ImageResizeAndPad, ImageNorm, TransformPipeline
from segment_anything.utils.utils import Timer
from use_sam_with_promts import show_box, show_mask


def set_context(device='Ascend', device_id=0):
    context = mslite.Context()
    context.target = [device.lower()]
    if device.lower() == 'ascend':
        context.ascend.device_id = device_id
        context.ascend.precision_mode = "preferred_fp32"  # this line is important for keeping precision
    elif device.lower() == 'gpu':
        context.gpu.device_id = device_id
    else:
        raise NotImplementedError
    return context


def build_model(lite_mindir_path, context):
    print(f'build model from: {lite_mindir_path}')
    model = mslite.Model()
    model.build_from_file(lite_mindir_path, mslite.ModelType.MINDIR, context)
    return model


def infer(args):
    # Step0: prepare model
    context = set_context(device=args.device, device_id=args.device_id)
    with Timer('build model'):
        model = build_model(args.model_path, context)

    # Step1: data preparation
    with Timer('preprocess'):
        transform_list = [
            ImageResizeAndPad(target_size=1024, apply_mask=False),
            ImageNorm(),
        ]
        transform_pipeline = TransformPipeline(transform_list)

        image_path = args.image_path
        image_np = cv2.imread(image_path)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        boxes_np = np.array([[425, 600, 700, 875]])

        transformed = transform_pipeline(dict(image=image_np, boxes=boxes_np))
        image, boxes, origin_hw = transformed['image'], transformed['boxes'], transformed['origin_hw']
        # batch_size for speed test
        # image = ms.Tensor(np.expand_dims(image, 0).repeat(8, axis=0))  # b, 3, 1023
        # boxes = ms.Tensor(np.expand_dims(boxes, 0).repeat(8, axis=0))  # b, n, 4
        image = np.expand_dims(image, 0)  # b, 3, 1023
        boxes = np.expand_dims(boxes, 0)  # b, n, 4

        inputs = model.get_inputs()
        inputs[0].set_data_from_numpy(image.astype(np.float32))
        inputs[1].set_data_from_numpy(boxes.astype(np.float32))


    # Step2: inference
    with Timer('model inference'):
        mask_logits = model.predict(inputs)[0]   # (1, 1, 1024, 1024)

    with Timer('Second time inference'):
        mask_logits = model.predict(inputs)[0]   # (1, 1, 1024, 1024)

    # Step3: post-process
    with Timer('post-process'):
        mask_logits = mask_logits.get_data_to_numpy()[0, 0] > 0.0  # (1024, 1024)
        mask_logits = mask_logits.astype(np.uint8)
        final_mask = cv2.resize(mask_logits[:origin_hw[2], :origin_hw[3]], tuple((origin_hw[1], origin_hw[0])),
                                interpolation=cv2.INTER_CUBIC)

    # Step4: visualize
    plt.imshow(image_np)
    show_box(boxes_np[0], plt.gca())
    show_mask(final_mask, plt.gca())
    plt.savefig(args.image_path + '_lite_infer.jpg')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=("Runs inference on one image"))
    parser.add_argument("--image_path", type=str, default='./images/truck.jpg', help="Path to an input image.")
    parser.add_argument("--model-path", type=str, default='./models/sam_vit_b_lite.mindir',  help="mindir model path for lite inference")
    parser.add_argument("--device", type=str, default="Ascend", help="The device to run generation on.")
    parser.add_argument("--device_id", type=int, default=0, help="The device to run inference on.")

    args = parser.parse_args()
    print(args)
    infer(args)
