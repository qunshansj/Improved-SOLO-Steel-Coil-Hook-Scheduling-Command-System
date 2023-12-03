python

class ObjectDetector:
    def __init__(self, config, checkpoint, device='cuda:0', palette='coco', score_thr=0.3):
        self.model = init_detector(config, checkpoint, device=device)
        self.palette = palette
        self.score_thr = score_thr

    def detect(self, img, out_file):
        result = inference_detector(self.model, img)
        show_result_pyplot(
            self.model,
            img,
            result,
            palette=self.palette,
            score_thr=self.score_thr,
            out_file=out_file
        )

    async def async_detect(self, img, out_file):
        tasks = asyncio.create_task(async_inference_detector(self.model, img))
        result = await asyncio.gather(tasks)
        show_result_pyplot(
            self.model,
            img,
            result[0],
            palette=self.palette,
            score_thr=self.score_thr,
            out_file=out_file
        )


