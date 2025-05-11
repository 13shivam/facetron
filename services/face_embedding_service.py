import onnxruntime


class FaceModel:
    def __init__(self, model_path: str):
        self.session = onnxruntime.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def get_embedding(self, input_tensor):
        result = self.session.run(None, {self.input_name: input_tensor})
        return result[0][0]