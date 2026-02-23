import numpy as np
import pandas as pd
from glob import glob
from pathlib import Path
import iris
from iris.nodes.matcher.hamming_distance_matcher import HammingDistanceMatcher


EMPTY_ROW = {
    "image_id": np.nan,
    "frame_no": np.nan,
    "eye_side": np.nan,
    "image_width": np.nan,
    "image_height": np.nan,

    "iris_center_x": np.nan,
    "iris_center_y": np.nan,
    "pupil_center_x": np.nan,
    "pupil_center_y": np.nan,

    "pupil_iris_diameter_ratio": np.nan,
    "pupil_iris_center_dist_ratio": np.nan,

    "offgaze_score": np.nan,
    "eye_orientation": np.nan,
    "occlusion90": np.nan,
    "occlusion30": np.nan,
    "sharpness_score": np.nan,

    "iris_xmin": np.nan,
    "iris_ymin": np.nan,
    "iris_xmax": np.nan,
    "iris_ymax": np.nan,

    "score": np.nan,
    "rotation": np.nan,
}


def iris_result_to_row(d):
    return {
        "image_id": d["image_id"],
        "frame_no": d["frame_no"],
        "eye_side": d["eye_side"],
        "image_width": d["image_size"][0],
        "image_height": d["image_size"][1],

        # 瞳孔 & 虹膜中心（你前面正好在研究这个）
        "iris_center_x": d["eye_centers"]["iris_center"][0],
        "iris_center_y": d["eye_centers"]["iris_center"][1],
        "pupil_center_x": d["eye_centers"]["pupil_center"][0],
        "pupil_center_y": d["eye_centers"]["pupil_center"][1],

        # 比例信息
        "pupil_iris_diameter_ratio": d["pupil_to_iris_property"]["pupil_to_iris_diameter_ratio"],
        "pupil_iris_center_dist_ratio": float(d["pupil_to_iris_property"]["pupil_to_iris_center_dist_ratio"]),

        # 质量指标
        "offgaze_score": float(d["offgaze_score"]),
        "eye_orientation": float(d["eye_orientation"]),
        "occlusion90": float(d["occlusion90"]),
        "occlusion30": float(d["occlusion30"]),
        "sharpness_score": float(d["sharpness_score"]),

        # bbox
        "iris_xmin": d["iris_bbox"]["x_min"],
        "iris_ymin": d["iris_bbox"]["y_min"],
        "iris_xmax": d["iris_bbox"]["x_max"],
        "iris_ymax": d["iris_bbox"]["y_max"],

        # match socre
        "score": float(d["score"]),

        # rotation
        "rotation": float(d["rotation"]) * 360.0 / 256.0,
    }


def proces_one_frame(frame, id, frame_no=0, ref_code=None, eye_side="right"):
    output = iris_pipeline(iris.IRImage(img_data=frame, image_id=id, eye_side=eye_side))
    if output['error'] != '':
        metadata = EMPTY_ROW.copy()
        metadata["image_id"] = id
        metadata["eye_side"] = eye_side
    else:
        code = output['iris_template']
        if ref_code is None:
            score, rotation = 0.0, 0
        else:
            score, rotation = matcher.run_rotation(ref_code, code)

        metadata = output['metadata']
        metadata['score'] = score
        metadata['rotation'] = rotation

    metadata['frame_no'] = frame_no

    return iris_result_to_row(metadata)


def process_one_video(vid_file):
    cap = cv2.VideoCapture(vid_file)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {input_video_path}")

    # 获取视频信息
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # 先处理第一帧以获取图像尺寸
    ret, first_frame = cap.read()
    if not ret:
        raise ValueError("无法读取视频帧")
    first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    eye_side = 'right' if '右' in vid_file else 'left'
    ref_code = iris_pipeline(iris.IRImage(img_data=first_frame, image_id=vid_file, eye_side=eye_side))

    # 重置视频读取位置
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # 逐帧处理
    frame_idx = 0
    ret_rows = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret_one_frame = proces_one_frame(frame, id, frame_idx, ref_code=ref_code, eye_side=eye_side)
        ret_rows.append(ret_one_frame)

        frame_idx += 1
        if frame_idx % 200 == 0:
            print(f"已处理 {frame_idx}/{frame_count} 帧")

    df = pd.DataFrame(rows)

    # 释放资源
    cap.release()
    out.release()

    return df


if __name__ == "__main__":
    matcher = HammingDistanceMatcher(rotation_shift=20)
    iris_pipeline = iris.IRISPipeline()

    vid_files = glob("./data/data1/**/*.mp4")
    for i, vid_file in enumerate(vid_files):
        print(f"========== {i}, {vid_file} 开始处理!==========")
        df = process_one_video(vid_file)
        df.to_excel(f"./output/{Path(vid_file).stem}.xlsx", index=False)
        print(f"========== {i}, {vid_file} 处理完成!==========\n\n")


