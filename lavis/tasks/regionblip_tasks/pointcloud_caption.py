import json
import os

from lavis.common.dist_utils import main_process
from lavis.common.registry import registry
from lavis.tasks.base_task import BaseTask


@registry.register_task("pointcloud_captioning_objaverse")
class PointcloudCaptionObjaverseTask(BaseTask):
    def __init__(self, num_beams, max_len, min_len, evaluate, report_metric=True):
        super().__init__()

        self.num_beams = num_beams
        self.max_len = max_len
        self.min_len = min_len
        self.evaluate = evaluate

        self.report_metric = report_metric

    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg

        num_beams = run_cfg.num_beams       # 5
        max_len = run_cfg.max_len           # 30
        min_len = run_cfg.min_len           # 8
        evaluate = run_cfg.evaluate         # True

        report_metric = run_cfg.get("report_metric", True)

        return cls(
            num_beams=num_beams,
            max_len=max_len,
            min_len=min_len,
            evaluate=evaluate,
            report_metric=report_metric,
        )

    def valid_step(self, model, samples):
        results = []
        num_captions = 1 # Number of captions to be generated for each image.

        captions = model.generate_pointcloud_caption(
            samples,
            use_nucleus_sampling=False,
            num_beams=self.num_beams,
            max_length=self.max_len,
            min_length=self.min_len,
            num_captions=num_captions)

        img_ids = samples["image_id"]                                      
        for caption, img_id in zip(captions, img_ids):
            results.append({"caption": caption, "image_id": img_id})

        return results

    def after_evaluation(self, val_result, split_name, epoch, **kwargs):
        # val_result: [ {'caption': 'a person riding a motorcycle on a dirt road', 'image_id': 391895}, 
        #               {'caption': 'a little girl blowing out a candle on a cupcake', 'image_id': 60623},
        #               ...]
        
        eval_result_file = self.save_result(
            result=val_result,
            result_dir=registry.get_path("result_dir"),
            filename="{}_epoch{}".format(split_name, epoch),
            remove_duplicate="image_id",
        )

        if self.report_metric:
            metrics = self._report_metrics(
                eval_result_file=eval_result_file, split_name=split_name
            )
        else:
            metrics = {"agg_metrics": 0.0}

        return metrics

    @main_process
    def _report_metrics(self, eval_result_file, split_name):

        coco_val = coco_caption_eval(eval_result_file)

        agg_metrics = coco_val.eval["CIDEr"] + coco_val.eval["Bleu_4"]
        log_stats = {split_name: {k: v for k, v in coco_val.eval.items()}}

        with open(
            os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
        ) as f:
            f.write(json.dumps(log_stats) + "\n")

        coco_res = {k: v for k, v in coco_val.eval.items()}
        coco_res["agg_metrics"] = agg_metrics

        return coco_res


# TODO better structure for this.
from pycocoevalcap.eval import COCOEvalCap
from pycocotools.coco import COCO


def coco_caption_eval(results_file):
    annotation_file = registry.get_path('caption_evaluate_objaverse_test')
    assert os.path.isfile(annotation_file), "Please register the path of caption_evaluate_refcoco_test in lavis.__init__.py"

    # create coco object and coco_result object
    coco = COCO(annotation_file)
    coco_result = coco.loadRes(results_file)

    # create coco_eval object by taking coco and coco_result
    coco_eval = COCOEvalCap(coco, coco_result)

    # evaluate on a subset of images by setting
    # coco_eval.params['image_id'] = coco_result.getImgIds()
    # please remove this line when evaluating the full validation set
    # coco_eval.params['image_id'] = coco_result.getImgIds()

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    coco_eval.evaluate()

    # print output evaluation scores
    for metric, score in coco_eval.eval.items():
        print(f"{metric}: {score:.3f}")

    return coco_eval
