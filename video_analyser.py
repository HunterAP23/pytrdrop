import argparse as argp
import os
import sys
import time

import cv2
import numpy as np
import pandas as pd

def progress(cur_frame, total_frames, msg):
    percent = "{0:.2f}".format(cur_frame * 100 / total_frames).zfill(5)
    sys.stdout.write("\r{0} {1} out of {2} : {3}%".format(msg, cur_frame, total_frames, percent))
    sys.stdout.flush()

def main(args):
    cap = cv2.VideoCapture(args.INPUT)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    reported_fps = int(cap.get(cv2.CAP_PROP_FPS))
    reported_bitrate = int(cap.get(cv2.CAP_PROP_BITRATE))

    print("Total frames: {0}".format(total_frames))
    print("Reported FPS: {0}".format(reported_fps))
    print("Reported Bitrate: {0}kbps".format(reported_bitrate))

    frame_diffs = []

    frame_number = -1

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = (width, height)

    result = None
    if args.SAVE == 1 or args.SAVE == 3:
        result = cv2.VideoWriter("original.avi", cv2.VideoWriter_fourcc(*"MJPG"), reported_fps, size)

    prev_frame = None
    while(cap.isOpened()):
        frame_number += 1
        progress(frame_number, total_frames, "Calculating frame")

        ret, frame = cap.read()

        if result is not None:
            result.write(frame)

        if frame_number == 0:
            prev_frame = frame.copy()
            continue

        try:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            frame_diff = cv2.absdiff(frame_gray, prev_frame_gray)
            frame_diffs.append(frame_diff)

            prev_frame = frame.copy()
        except KeyboardInterrupt:
            exit(1)
        except:
            break

        if frame_number > total_frames:
            break

    cap.release()
    if result is not None:
        result.release()
    cv2.destroyAllWindows()
    progress(total_frames, total_frames, "Calculating frame")

    print("\nCalculating frame differences...")
    means = []
    for i in range(len(frame_diffs)):
        progress(i, len(frame_diffs), "Calculating mean for frame")
        means.append(frame_diffs[i].mean())
    progress(len(frame_diffs), len(frame_diffs), "Calculating mean for frame")

    frames = []

    print("\nGetting average difference per frame...")
    for i in range(len(means)):
        progress(i, len(means), "Calculating mean difference per frame for frame")
        if means[i] > args.THRESHOLD:
            frames.append(True)
        else:
            frames.append(False)
    progress(len(means), len(means), "Calculating mean difference per frame for frame")

    print("\nThere were a total of {0} unique frames found with the threshold of {1}".format(sum(frames), args.THRESHOLD))

    res = [i for i, val in enumerate(frames) if val]

    times = dict()

    base = float((1 / reported_fps) * 1000)
    for i in range(len(res)):
        if i == 0:
            times[i] = base
            continue
        times[i] = base * (res[i] - res[i - 1])

    data = dict()
    data["frame number"] = []
    data["frametime"] = []
    data["framerate"] = []
    for k, v in times.items():
        data["frame number"].append(k)
        data["frametime"].append(v)
        data["framerate"].append(1 / (v / 1000))

    output_name = ""
    if args.OUTPUT:
        temp_name = args.OUTPUT.split(".")
        if temp_name[-1] == "csv":
            temp_name = ".".join(temp_name)
            output_name = os.path.join(os.getcwd(), temp_name)
        else:
            temp_name = ".".join(temp_name) + ".csv"
            output_name = os.path.join(os.getcwd(), temp_name)
    else:
        temp_name = args.INPUT.split(".")
        temp_name = "".join(temp_name[0:-1])
        output_name = "{0}.csv".format(temp_name)

    df = pd.DataFrame.from_dict(data)
    with open(output_name, "w", newline="\n") as csv_file:
        df.to_csv(csv_file, index=False)

    frametime_stats = pd.DataFrame(df, columns=["frametime"])
    framerate_stats = pd.DataFrame(df, columns=["framerate"])

    stats_basic = dict()
    stats_frametime_dict = dict()
    stats_framerate_dict = dict()

    stats_basic["Number of Unique Frames"] = [int(sum(frames))]
    stats_basic["Number of Duplicated Frames"] = [int(len(frames) - sum(frames))]
    stats_basic["Percentage of Unique Frames"] = [sum(frames) / len(frames) * 100]
    stats_basic["Percentage of Duplicated Frames"] = [stats_basic["Number of Duplicated Frames"][0] / len(frames) * 100]

    stats_frametime_dict["Lowest"] = dict(frametime_stats.min(axis=0))
    stats_frametime_dict["Highest"] = dict(frametime_stats.max(axis=0))
    stats_frametime_dict["Mean"] = dict(frametime_stats.mean(axis=0))
    stats_frametime_dict["Median"] = dict(frametime_stats.median(axis=0))
    stats_frametime_dict["0.1 Percent Lows"] = dict(frametime_stats.quantile(q=0.001, axis=0))
    stats_frametime_dict["1 Percent Lows"] = dict(frametime_stats.quantile(q=0.01, axis=0))
    stats_frametime_dict["99 Percent Lows"] = dict(frametime_stats.quantile(q=0.99, axis=0))
    stats_frametime_dict["99.9 Percent Lows"] = dict(frametime_stats.quantile(q=0.999, axis=0))

    stats_framerate_dict["Lowest"] = dict(framerate_stats.min(axis=0))
    stats_framerate_dict["Highest"]= dict(framerate_stats.max(axis=0))
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

    if args.SAVE > 1:
        print("Saving video containing only unique frames.")
        cur_frame = 0
        cap = cv2.VideoCapture(args.INPUT)

        result = None
        if args.SAVE > 1:
            result = cv2.VideoWriter("unique.avi", cv2.VideoWriter_fourcc(*"MJPG"), reported_fps, size)

        while(cap.isOpened()):
            cur_frame += 1
            progress(cur_frame, len(frames), "Saving frame")
            if cur_frame in res:
                ret, frame = cap.read()
                try:
                    if result is not None:
                        result.write(frame)
                except KeyboardInterrupt:
                    exit(1)
                except:
                    break

            if cur_frame > len(frames):
                break

    cap.release()
    if result is not None:
        result.release()
    cv2.destroyAllWindows()

    print("\nStatistics", stats_basic_df.transpose().to_string(header=False))
    print("\n", stats_joined.transpose().to_string())


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

    save_help = "Save the video frames of the original video as well as a version with duplicated frames removed.\n"
    save_help += "A value of 0 will not save any video files.\n"
    save_help += "A value of 1 will only save the original video file.\n"
    save_help += "A value of 2 will only save the version with duplicated frames removed.\n"
    save_help += "A value of 3 will save both the original and version with duplicated frames removed.\n"
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
