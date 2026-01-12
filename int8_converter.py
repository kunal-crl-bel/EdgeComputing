import os
import argparse
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import glob
import cv2

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class ImageCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, calibration_images_dir, input_shape=(3, 640, 640), cache_file="calib.cache"):
        super(ImageCalibrator, self).__init__()
        self.cache_file = cache_file
        self.input_shape = input_shape
        self.batch_size = 1
        self.image_paths = glob.glob(
            os.path.join(calibration_images_dir, "*.jpg"))
        self.current_index = 0

        self.device_input = cuda.mem_alloc(
            trt.volume(self.input_shape) * np.float32().nbytes)

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.current_index >= len(self.image_paths):
            return None

        img_path = self.image_paths[self.current_index]
        img = cv2.imread(img_path)
        img = cv2.resize(img, (self.input_shape[2], self.input_shape[1]))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC to CHW
        img = np.expand_dims(img, axis=0)

        cuda.memcpy_htod(self.device_input, img)
        self.current_index += 1
        return [self.device_input]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)


def build_engine(onnx_path, engine_path, calib_dir):
    builder = trt.Builder(TRT_LOGGER)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            return None

    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30
    config.set_flag(trt.BuilderFlag.INT8)

    input_shape = (3, 640, 640)
    calibrator = ImageCalibrator(calib_dir, input_shape=input_shape)
    config.int8_calibrator = calibrator

    engine = builder.build_engine(network, config)
    with open(engine_path, 'wb') as f:
        f.write(engine.serialize())

    print(f"[INT8] Engine saved to {engine_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx', type=str, required=True,
                        help='Path to YOLOv8 ONNX model')
    parser.add_argument(
        '--engine', type=str, default='model_int8.engine', help='Output INT8 engine file')
    parser.add_argument('--calib_dir', type=str, required=True,
                        help='Path to directory with calibration images')
    args = parser.parse_args()

    build_engine(args.onnx, args.engine, args.calib_dir)


if __name__ == '__main__':
    main()


# Run : python3 int8_converter.py --onnx yolov8l.onnx --engine yolov8l_int8.engine --calib_dir calib_images/
