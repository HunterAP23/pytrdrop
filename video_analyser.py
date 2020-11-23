import argparse as argp
import os
import sys
import time

import cv2
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

def progress(cur_frame, total_frames):
    percent = "{0:.2f}".format(cur_frame * 100 / total_frames).zfill(5)
    sys.stdout.write("\rCalculating frame " + str(cur_frame) + " out of " + str(total_frames) + " : " + percent + "%")
    sys.stdout.flush()

def main(args):
    cap = cv2.VideoCapture(args.INPUT)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    reported_fps = int(cap.get(cv2.CAP_PROP_FPS))
    reported_bitrate = int(cap.get(cv2.CAP_PROP_BITRATE))

    print("Total frames: {0}".format(total_frames))
    print("Reported FPS: {0}".format(reported_fps))
    print("Reported Bitrate: {0}kbps".format(reported_bitrate))

    # data_dict = dict()
    # data_dict["framerate"] = []
    # data_dict["frametime"] = []

    # times = dict()
    # times["start"] = []
    # times["end"] = []
    # times["processing"] = []
    frame_number = 0

    # new_frame_time = 0
    # prev_frame_time = time.time()
    # processing_time_start = 0
    # while(cap.isOpened()):
    #     progress(frame_number, total_frames)
    #     frame_number += 1
    #
    #     processing_time_total = time.time() - processing_time_start
    #
    #     ret, frame = cap.read()
    #
    #     if not ret:
    #         break
    #
    #     # width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    #     # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #
    #     new_frame_time = time.time()
    #     times["start"].append(prev_frame_time)
    #     times["end"].append(new_frame_time)
    #     times["processing"].append(processing_time_total)
    #
    #     prev_frame_time = new_frame_time
    #     processing_time_start = time.time()
    #     # if frame_number == 60:
    #     #     break
    #
    #     # cv2.putText(frame, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)
    #     cv2.imshow("frame", frame)
    #
    #     # press "Q" if you want to exit
    #     if cv2.waitKey(1) & 0xFF == ord("q"):
    #         break

    prev_frame = None
    while(cap.isOpened()):
        progress(frame_number, total_frames)
        frame_number += 1

        ret, frame = cap.read()

        if frame_number == 1:
            prev_frame = frame.copy()
            continue

        current_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        previous_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

        frame_diff = cv2.absdiff(current_frame_gray,previous_frame_gray)

        cv2.imshow("frame diff", frame_diff)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        prev_frame = frame.copy()

    cap.release()
    cv2.destroyAllWindows()
    progress(total_frames, total_frames)

    output_name = ""
    if args.OUTPUT:
        temp_name = args.OUTPUT.split(".")
        temp_name = "".join(temp_name[0:-1])
        output_name = "{0}.csv".format(temp_name)
    else:
        temp_name = args.INPUT.split(".")
        temp_name = "".join(temp_name[0:-1])
        output_name = "{0}.csv".format(temp_name)

    print("")
    for i in range(len(times["start"])):
        val = 1 / (times["end"][i] - times["start"][i])
        if val > 60:
            print("ERROR")
            print("{0}".format(val))
        else:
            print("{0}".format(val))

    # df = pd.DataFrame.from_dict(data_dict)
    # print("")
    # print(df)

    # with open(output_name, "w") as csv_file:



def parse_arguments():
    main_help = "Analyze framerate, frame drops, and frame tears of a video file.\n"
    parser = argp.ArgumentParser(description=main_help, formatter_class=argp.RawTextHelpFormatter)
    parser.add_argument("INPUT", type=str, help="Video File")

    output_help = "Output filename (Default will be named after input file)."
    parser.add_argument("-o", "--output", dest="OUTPUT", type=str, help=output_help)

    args = parser.parse_args()

    # if args.dpi <= 1:
    #     parser.error("Value {0} for \"dpi\" argument was not a positive integer.".format(args.dpi))
    #     exit(1)

    return(args)

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
