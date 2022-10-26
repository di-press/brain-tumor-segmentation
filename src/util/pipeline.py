from src.evaluation.iou import iou_similarity_score

class Pipeline:
    def __init__(self, parameters: dict, parameters_args: dict=None):
        self._steps = []
        self._args = []
        for step in ("equalization", "filtering", "segmentation", "post_processing"):
            function = parameters.get(step, None)
            if function.value is None:
                continue
            self._steps.append(function.value)
            self._args.append(parameters_args.get(function.name.lower(), None))

    def apply(self, input_images, ground_truth_images):
        images = input_images
        for function, args in zip(self._steps, self._args):
            new_images = []
            for image in images:
                if args is not None:
                    result = function(image, **args)
                else:
                    result = function(image)

                if isinstance(result, tuple):
                    new_images.append(result[0])
                else:
                    new_images.append(result)
            images = new_images
        
        predictions = [prediction for prediction in images]
        iou_score = [iou_similarity_score(ground_truth, prediction) for ground_truth, prediction in zip(ground_truth_images, predictions)]
        return predictions, iou_score