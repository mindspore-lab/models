from importlib import import_module

from munch import DefaultMunch

from utils.tiler import Tiler

BACKEND = {"ascend": "AscendBackend", "onnx": "OnnxBackend", "bolt": "BoltBackend", "ms": "MsBackend"}

DATA_IO = {
    "basicvsr_mindspore": "BasicVSRDataIO",
    "ipt_sr_mindspore": "IPTSRDataIO",
    "mimo_unet_mindspore": "MIMOUnetDataIO",
    "rgb_to_bgr": "RGB2BGRDataIO",
    "bgr_to_rgb": "BGR2RGBDataIO",
}


class Task:
    # TODO
    def __init__(self, cfg):
        if isinstance(cfg, dict):
            cfg = DefaultMunch.fromDict(cfg)

        self.task_name = cfg.task_name
        self.once_process_frames = cfg.once_process_frames
        self.frame_overlap = cfg.get("frame_overlap", 0)
        self.patch_overlap = cfg.get("patch_overlap", 0)
        self.dtype = cfg.get("dtype", "float32")
        self.up_scale = cfg.get("up_scale", 1)
        self.need_tiling = cfg.get("need_tiling", True)
        self.tiler = None

        if cfg.backend and cfg.model_file:
            backend_name = cfg.backend.lower()
            backend_module = import_module(f"utils.{backend_name}.{backend_name}_backend")
            # backend_device_id = cfg.device_id if cfg.device_id is not None else -1
            self.backend = getattr(backend_module, BACKEND[backend_name])(cfg.model_file, **cfg.__dict__)
        else:
            self.backend = None

        data_io_name = cfg.data_io.lower()
        data_io_module = import_module(f"utils.{data_io_name}_io")
        self.data_io = getattr(data_io_module, DATA_IO[data_io_name])()
        self.data_io.set_scale(self.up_scale)
        if self.backend is not None:
            input_shape = self.backend.get_input_shape()
            if len(input_shape) == 4:
                tlen, height, width = input_shape[0], input_shape[-2], input_shape[-1]
            elif len(input_shape) == 5:
                _, tlen, _, height, width = self.backend.get_input_shape()
            if self.need_tiling:
                self.tiler = Tiler(
                    self.backend,
                    frame_window_size=tlen,
                    frame_overlap=self.frame_overlap,
                    patch_size=(height, width),
                    patch_overlap=self.patch_overlap,
                    sf=self.up_scale,
                    dtype=self.dtype,
                )

    # @func_time('one task total')
    def run(self, input_data=None, **kwargs):
        input = self.data_io.preprocess(input_data=input_data)
        if self.tiler is not None:
            output = self.tiler.process_video(input, **kwargs)
        elif self.backend is not None:
            output = self.backend.run(input_list=input, **kwargs)
        else:
            print("Warning. Please check if backend is set correctly.")
            # Can be case where task is RGB2BGR or similar.
            output = input
        result = self.data_io.postprocess(output)
        return result
