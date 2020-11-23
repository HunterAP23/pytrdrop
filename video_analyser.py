import argparse as argp
import os
import sys
import time

import cv2
from matplotlib import pyplot as plt
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

    prev_frame = None
    while(cap.isOpened()):
        frame_number += 1
        progress(frame_number, total_frames, "Calculating frame")

        ret, frame = cap.read()

        if frame_number == 0:
            prev_frame = frame.copy()
            continue

        try:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            frame_diff = cv2.absdiff(frame_gray, prev_frame_gray)
            frame_diffs.append(frame_diff)

            prev_frame = frame.copy()
        except:
            break

        if frame_number > total_frames:
            break

    cap.release()
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
        times[i] = base * (res[i] - res[i - 1] + 1)

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
        temp_name = "".join(temp_name[0:-1])
        output_name = "{0}.csv".format(temp_name)
    else:
        temp_name = args.INPUT.split(".")
        temp_name = "".join(temp_name[0:-1])
        output_name = "{0}.csv".format(temp_name)

    with open(output_name, "w", newline="\n") as csv_file:
        df = pd.DataFrame.from_dict(data)
        df.to_csv(csv_file, index=False)

    # if args.VIEW:
    #     print("Displaying video containing only unique frames.")
    #     cur_frame = 0
    #     cap = cv2.VideoCapture(args.INPUT)
    #     while(cap.isOpened()):
    #         cur_frame += 1
    #         if cur_frame <= len(res):
    #             if cur_frame in res:
    #                 ret, frame = cap.read()
    #                 try:
    #                     cv2.imshow("frame", frame)
    #                     if cv2.waitKey(1) & 0xFF == ord("q"):
    #                         break
    #                 except:
    #                     continue
    #         else:
    #             break


def parse_arguments():
    main_help = "Analyze framerate, frame drops, and frame tears of a video file.\n"
    parser = argp.ArgumentParser(description=main_help, formatter_class=argp.RawTextHelpFormatter)
    parser.add_argument("INPUT", type=str, help="Video File")

    output_help = "Output filename (Default will be named after input file)."
    parser.add_argument("-o", "--output", dest="OUTPUT", type=str, help=output_help)

    threshold_help = "Pixel difference threshold to count as duplicate frames, must be an integer between 0 and 255.\n"
    threshold_help += "A value of 0 will count all all frames as unique, while 255 will only count\n"
    threshold_help += "frames that are 100 percent different (Default is 5)."
    parser.add_argument("-t", "--threshold", dest="THRESHOLD", type=int, default=5, help=threshold_help)

    # view_help = "Whether or not to display the final video with all the duplicated frames removed (default: False)"
    # parser.add_argument("-v", "--view", dest="VIEW", action="store_true", default=False, help=view_help)

    args = parser.parse_args()

    # if args.dpi <= 1:
    #     parser.error("Value {0} for \"dpi\" argument was not a positive integer.".format(args.dpi))
    #     exit(1)

    return(args)

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
