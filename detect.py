python
from mmdet.apis import init_detector, inference_detector
import mmcv
import cv2

class ObjectDetector:
    def __init__(self, config_file, checkpoint_file):
        self.model = init_detector(config_file, checkpoint_file, device='cuda:0')

    def detect_image(self, img_path, score_thr=0.2):
        img = mmcv.imread(img_path)
        result = inference_detector(self.model, img)
        outimg = self.model.show_result(img, result, score_thr=score_thr, bbox_color='red', text_color='red')
        cv2.namedWindow('image', 0)
        mmcv.imshow(outimg, 'image', 0)

    def detect_video(self, video_path, output_path):
        video = mmcv.VideoReader(video_path)
        video_writer = None
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, video.fps, (video.width, video.height))
        count = 0
        for frame in mmcv.track_iter_progress(video):
            count += 1
            result = inference_detector(self.model, frame)
            outframe = self.model.show_result(frame, result, score_thr=0.8)
            cv2.namedWindow('video', 1)
            mmcv.imshow(outframe, 'video', 1)
            video_writer.write(outframe)
        video_writer.release()
        cv2.destroyAllWindows()
