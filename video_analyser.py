import argparse as argp
import multiprocessing.dummy as mp
from pathlib import Path
import time

import cv2
import pandas as pd
from tqdm import tqdm


def writer_write(writer, q):
    while True:
        frame = q.get()
        if type(frame) == str and frame == "STOP":
            return
        writer.write(frame)


def writer_create(name: str, fps, size, q):
    video_writer = cv2.VideoWriter(name, cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    writer_write(video_writer, q)
    video_writer.release()
    return


def main(args):
    cap = cv2.VideoCapture(args.INPUT)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    reported_fps = int(cap.get(cv2.CAP_PROP_FPS))
    reported_bitrate = int(cap.get(cv2.CAP_PROP_BITRATE))

    print("Total frames: {0}".format(total_frames))
    print("Reported FPS: {0}".format(reported_fps))
    print("Reported Bitrate: {0}kbps".format(reported_bitrate))

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = (width, height)

    original_path = None
    if args.OUTPUT:
        original_path = Path(args.OUTPUT)
    else:
        original_path = Path(args.INPUT)
    original_name = original_path.stem
    original_parent = original_path.parent

    # Video with duplicated frames removed
    result_queue = None
    result_proc = None
    if args.SAVE in (1, 3):
        result_name = Path(original_parent).joinpath(original_name + "_result.avi")
        result_name.unlink(missing_ok=True)
        result_queue = mp.Queue()
        result_proc = mp.Process(target=writer_create, args=(str(result_name), reported_fps, size, result_queue,))
        result_proc.start()

    # Video with difference blend mode between original and result video
    diff_queue = None
    diff_proc = None
    if args.SAVE in (2, 3):
        diff_name = Path(original_parent).joinpath(original_name + "_diff.avi")
        diff_name.unlink(missing_ok=True)
        diff_queue = mp.Queue()
        diff_proc = mp.Process(target=writer_create, args=(str(diff_name), reported_fps, size, diff_queue,))
        diff_proc.start()

    frames = []
    frame_number = -1
    prev_frame = None
    # with tqdm(total=total_frames, unit="frames") as prog_bar:
    time_start = time.time()
    prog_bar = tqdm(total=total_frames, unit="frames", leave=True)
    while(cap.isOpened()):
        frame_number += 1
        prog_bar.set_description("Processing frame number {}".format(frame_number))
        prog_bar.update(1)

        ret, frame = cap.read()

        if frame_number == 0:
            prev_frame = frame.copy()
            continue

        try:
            # frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            # frame_diff = cv2.absdiff(frame_gray, prev_frame_gray)
            frame_diff = cv2.absdiff(frame, prev_frame)
            if diff_queue is not None:
                # diff_writer.write(frame_diff)
                diff_queue.put(frame_diff)
            mean = frame_diff.mean()

            if mean > args.THRESHOLD:
                frames.append(True)
                if result_queue is not None:
                    # result_writer.write(frame)
                    result_queue.put(frame)
            else:
                frames.append(False)

            prev_frame = frame.copy()
        except KeyboardInterrupt:
            exit(1)
        except Exception as e:
            # print("\r\n{0}".format(e))
            if frame_number > total_frames:
                break
            else:
                continue

        if frame_number > total_frames:
            break

    time_stop = time.time()
    n = prog_bar.n
    prog_bar.close()
    msg = "Calculations "

    cap.release()
    if result_queue is not None:
        print("Finishing writing to result video file.")
        result_queue.put("STOP")
        result_proc.join()
        msg += "and writing to the result file "
    if diff_queue is not None:
        print("Finishing writing to difference video file.")
        diff_queue.put("STOP")
        diff_proc.join()
        msg += "and writing to the difference file "
    cv2.destroyAllWindows()

    time_total = time_stop - time_start
    app_fps = n / time_total
    msg += "took {0} seconds to complete.\n"
    msg += "Average frames calculated per second is {1}."
    print(msg.format(time_total, app_fps))

    res = [i for i in frames if i]

    times = dict()

    base = float((1 / reported_fps) * 1000)
    for i in range(len(res)):
        if i == 0:
            times[i] = base
        else:
            times[i] = base + (base * (res[i] - res[i - 1]))

    data = dict()
    data["frame number"] = []
    data["frametime"] = []
    data["framerate"] = []
    for k, v in times.items():
        data["frame number"].append(k)
        data["frametime"].append(v)
        if v == 0:
            data["framerate"].append("INF")
        else:
            data["framerate"].append(1 / (v / 1000))

    df = pd.DataFrame.from_dict(data)

    frametime_stats = pd.DataFrame(df, columns=["frametime"])
    framerate_stats = pd.DataFrame(df, columns=["framerate"])

    stats_basic = dict()
    stats_frametime_dict = dict()
    stats_framerate_dict = dict()

    stats_basic["Number of Unique Frames"] = [int(sum(frames))]
    stats_basic["Number of Duplicated Frames"] = [int(len(frames) - sum(frames))]
    if len(frames) == 0:
        stats_basic["Percentage of Unique Frames"] = ["0 %"]
        stats_basic["Percentage of Duplicated Frames"] = ["0 %"]
    else:
        stats_basic["Percentage of Unique Frames"] = ["{} %".format(sum(frames) / len(frames) * 100)]
        stats_basic["Percentage of Duplicated Frames"] = ["{} %".format(stats_basic["Number of Duplicated Frames"][0] / len(frames) * 100)]

    stats_frametime_dict["Lowest"] = dict(frametime_stats.min(axis=0))
    stats_frametime_dict["Highest"] = dict(frametime_stats.max(axis=0))
    stats_frametime_dict["Mean"] = dict(frametime_stats.mean(axis=0))
    stats_frametime_dict["Median"] = dict(frametime_stats.median(axis=0))
    stats_frametime_dict["0.1 Percent Lows"] = dict(frametime_stats.quantile(q=0.001, axis=0))
    stats_frametime_dict["1 Percent Lows"] = dict(frametime_stats.quantile(q=0.01, axis=0))
    stats_frametime_dict["99 Percent Lows"] = dict(frametime_stats.quantile(q=0.99, axis=0))
    stats_frametime_dict["99.9 Percent Lows"] = dict(frametime_stats.quantile(q=0.999, axis=0))

    stats_framerate_dict["Lowest"] = dict(framerate_stats.min(axis=0))
    stats_framerate_dict["Highest"] = dict(framerate_stats.max(axis=0))
    stats_framerate_dict["Mean"] = dict(framerate_stats.mean(axis=0))
    stats_framerate_dict["Median"] = dict(framerate_stats.median(axis=0))
    stats_framerate_dict["0.1 Percent Lows"] = dict(framerate_stats.quantile(q=0.001, axis=0))
    stats_framerate_dict["1 Percent Lows"] = dict(framerate_stats.quantile(q=0.01, axis=0))
    stats_framerate_dict["99 Percent Lows"] = dict(framerate_stats.quantile(q=0.99, axis=0))
    stats_framerate_dict["99.9 Percent Lows"] = dict(framerate_stats.quantile(q=0.999, axis=0))

    stats_basic_df = pd.DataFrame.from_dict(stats_basic)
    stats_frametime_df = pd.DataFrame.from_dict(stats_frametime_dict)
    stats_framerate_df = pd.DataFrame.from_dict(stats_framerate_dict)

    stats_joined = pd.concat([stats_frametime_df, stats_framerate_df], axis=0)

    print("\nStatistics")
    print(stats_basic_df.transpose().to_string(header=False))
    print("\n", stats_joined.transpose().to_string())

    csv_name = Path(original_parent).joinpath(original_name + "_report.csv")
    csv_name.unlink(csv_name)
    df.to_csv(csv_name, index=False)
    # stats_joined.to_csv(csv_name)
    # with open(csv_name, "w", newline="\n") as csv_file:
    #     df.to_csv(csv_file, index=False)


def parse_arguments():
    main_help = "Analyze framerate, frame drops, and frame tears of a video file.\n"
    parser = argp.ArgumentParser(description=main_help, formatter_class=argp.RawTextHelpFormatter)
    parser.add_argument("INPUT", type=str, help="Video File")

    output_help = "Output filename (Default will be named after input file)."
    parser.add_argument("-o", "--output", dest="OUTPUT", type=str, help=output_help)

    threshold_help = "Pixel difference threshold to count as duplicate frames, must be an integer between 0 and 255.\n"
    threshold_help += "A value of 0 will count all all frames as unique, while 255 will only count\n"
    threshold_help += "frames that are 100 percent different (Default: 5)."
    parser.add_argument("-t", "--threshold", dest="THRESHOLD", type=int, default=5, help=threshold_help)

    save_help = "Save the video frames of the video with duplicated frames removed and/or the video showing the difference between the original and deduplicated frame video.\n"
    save_help += "A value of 0 will not save any video files.\n"
    save_help += "A value of 1 will only save the version with duplicated frames removed.\n"
    save_help += "A value of 2 will only save the version that shows the difference between the original and the deduplicated video.\n"
    save_help += "A value of 3 will save both of the videos from options 1 and 2.\n"
    save_help += "Note that saving the video file(s) can drastically increase the program's runtime. (Default: 0)"
    parser.add_argument("-s", "--save", dest="SAVE", action="store", type=int, default=0, choices=[0, 1, 2, 3], help=save_help)

    args = parser.parse_args()

    if args.THRESHOLD < 0 or args.THRESHOLD > 255:
        parser.error("Value {0} for \"threshold\" argument was not within the range of 0 to 255".format(args.THRESHOLD))
        exit(1)

    return(args)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
