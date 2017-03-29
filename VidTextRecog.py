#!/usr/bin/env python
"""
    * REFERENCE

    * FEATURES

    * NEXT STEP

"""

########################################################################################################################

import sys
import os
import ImageTextRecognition
import argparse
import cv2
import numpy as np
from MyUtility import RED, BLUE, BLACK
from MyUtility import get_video_stream_info, my_imshow
import pickle

########################################################################################################################

FRAME_SKIP_SEC = 5
FUNCTION_NAME = "VideoTextRecognition"

DISP_TEXT_OVERLAY_IMG_1BY1 = False
DISP_ALL_TEXT_OVERLAY_IMG = False

########################################################################################################################


class VideoInfoDB:
    def __init__(self, width, height, fps, frame_num):
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_num = frame_num


class VideoTextDB:
    def __init__(self, frame_num, frame_sec, x, y, width, height, text):
        self.frame_num = frame_num
        self.frame_sec = frame_sec
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.text = text


class VideoTextRecognition:

    def __init__(self, fps, width, height, frame_skip_sec):

        self.fps = fps
        self.width = width
        self.height = height
        self.frame_skip_sec = frame_skip_sec
        self.vid_text_list = []
        self.vid_text_idx = 0
        self.ImgText = ImageTextRecognition.ImgTextRecognition()

    # ------------------------------------------------------------------------------------------------------------------
    class VidText:
        def __init__(self, img_text_list, frame_num, frame_sec):
            self.img_text_list = img_text_list
            self.frame_num = frame_num
            self.frame_sec = frame_sec

    # ------------------------------------------------------------------------------------------------------------------
    def extract_text_from_video(self, img, frame_num, frame_sec):

        self.ImgText.clear__img_text_list()
        self.ImgText.img_text_list = self.ImgText.extract_text_from_image(img, display=False)
        for img_text in self.ImgText.img_text_list:
            img_text.img = []
        self.vid_text_list.append(self.VidText(list(self.ImgText.img_text_list), frame_num, frame_sec))

    # ------------------------------------------------------------------------------------------------------------------
    def overlay_all_texts(self, img, frame_num, frame_sec):

        text_img = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255

        # Find the video text object whose time is less than, equal, or close to the current frame sec.
        idx = 0
        for idx in range(len(self.vid_text_list)):
            if self.vid_text_list[idx].frame_sec > frame_sec:
                break
        vid_text = self.vid_text_list[idx]

        cv2.putText(text_img, str(frame_num) + " : " + str(frame_sec), (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, BLACK, 1,
                    cv2.LINE_AA)
        for img_text in vid_text.img_text_list:
            cv2.rectangle(img, img_text.get_stt_pnt(), img_text.get_end_pnt(), RED, 2)
            cv2.rectangle(text_img, img_text.get_stt_pnt(), img_text.get_end_pnt(), BLUE, 1)
            cv2.putText(text_img, img_text.text[0].encode('utf-8'), (img_text.x + 4, img_text.y + img_text.height - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, img_text.height * 0.016, BLACK, 1, cv2.LINE_AA)

        out_img = np.concatenate((img, text_img), axis=1)
        if DISP_ALL_TEXT_OVERLAY_IMG:           # show all text overlayed image.
            my_imshow(out_img)

        return out_img

########################################################################################################################
#   LOCAL FUNCTIONS...


def convert_video_text_class_to_db(img_text, frame_num, frame_sec):
    return VideoTextDB(frame_num, frame_sec, img_text.x, img_text.y, img_text.width, img_text.height,
                       " || ".join(img_text.text)
                       )


# ----------------------------------------------------------------------------------------------------------------------
def analyze_dump(dump_file):
    [videoInfoDB, videoTextDB] = pickle.load(open(dump_file, "rb"))

    fid = open(".".join(dump_file.split(".")[:-1]) + ".db_txt", "w")
    fid.write("\n\n # Video Info DB")
    fid.write("\n   - width   : {:d}".format(videoInfoDB.width))
    fid.write("\n   - height  : {:d}".format(videoInfoDB.height))
    fid.write("\n   - FPS     : {:4.2f}".format(videoInfoDB.fps))
    fid.write("\n   - frame # : {:d}".format(videoInfoDB.frame_num))

    fid.write("\n\n # Video Text DB")
    idx = -1
    for k in range(len(videoTextDB)):
        dat = videoTextDB[k]
        if idx != dat.frame_num:
            fid.write("\n")
            idx = dat.frame_num
        fid.write("\n [{:d}] : ({:4d} x {:4d}) & ({:4d} x {:4d})) : {}".format(dat.frame_num, dat.x, dat.y, dat.width,
                                                                               dat.height, dat.text.encode('utf-8')))
    fid.close()
    pass


# ----------------------------------------------------------------------------------------------------------------------
def main_image(arg):

    if not os.path.exists(arg.in_vid_file):
        print(" @ Error: input video file not found {}".format(arg.in_vid_file))
        sys.exit()

    try:
        video_stream = cv2.VideoCapture(arg.in_vid_file)
    except Exception as e:
        print(e)
        sys.exit()

    video_stream.set(cv2.CAP_PROP_POS_FRAMES, arg.frame_idx)
    flag, org_img = video_stream.read()
    if not flag:
        print(" @ Error: There is no {:d}-th frame...".format(arg.frame_idx))
        sys.exit()

    img_text = ImageTextRecognition.ImgTextRecognition()
    img_text.extract_text_from_image(org_img, display=False)
    img_text.show_img_text(org_img)


# ----------------------------------------------------------------------------------------------------------------------
def main_video(arg):

    if not os.path.exists(arg.in_vid_file):
        print(" @ Error: input video file not found {}".format(arg.in_vid_file))
        sys.exit()

    try:
        video_stream = cv2.VideoCapture(arg.in_vid_file)
        fps, width, height, tot_frame_num = get_video_stream_info(video_stream)
        arg.out_vid_file = ".".join(arg.out_vid_file.split('.')[:-1]) + ".avi"
        video_writer = cv2.VideoWriter(arg.out_vid_file, cv2.VideoWriter_fourcc(*'XVID'), fps, (2*width, height), 1)
    except Exception as e:
            print e
            sys.exit()

    # Loop
    frame_num = 0

    videoText = VideoTextRecognition(fps, width, height, tot_frame_num)
    videoInfoDB = VideoInfoDB(width, height, fps, tot_frame_num)
    video_text_db_list = []

    while cv2.waitKey(1) != ord('q'):

        flag, org_img = video_stream.read()
        if not flag:
            break
        frame_sec = int(frame_num / videoText.fps)

        print(" # {:d} frame in {:d} sec".format(frame_num, frame_sec)),

        if frame_num % int(videoText.fps * FRAME_SKIP_SEC) == 0:
            print("- text extraction...")
            videoText.extract_text_from_video(org_img, frame_num, frame_sec)
            for k in range(len(videoText.vid_text_list[-1].img_text_list)):
                video_text_db_list.append(convert_video_text_class_to_db(videoText.vid_text_list[-1].img_text_list[k],
                                                                         frame_num, frame_sec))
        else:
            print("")

        overlay_img = videoText.overlay_all_texts(org_img, frame_num, frame_sec)

        video_writer.write(overlay_img)

        frame_num += 1

        if frame_num == tot_frame_num:
            print(" # End of video\n")
            break

        if True:
            zoom_fac = 1500. / overlay_img.shape[1]
            cv2.imshow('overlay', cv2.resize(overlay_img, (0,0), fx=zoom_fac, fy=zoom_fac))
            cv2.waitKey(1)
            pass

    video_writer.release()
    pickle.dump([videoInfoDB, video_text_db_list], open(arg.pkl_file + ".pkl", 'wb'), pickle.HIGHEST_PROTOCOL)
    # object_name = pickle.load(open(pkl_filename, 'rb'))


########################################################################################################################

if __name__ == "__main__":

    if len(sys.argv) == 1:
        # sys.argv.extend(["-i", "Video.lecture/lecture_1.mp4", "-o", "lecture_1_text.mp4", "-n", "470"])
        sys.argv.extend(["-i", "Video.lecture/lecture_1.mp4", "-o", "lecture_1_text.mp4"])
        # sys.argv.extend(["-d", "VideoTextRecognition.pkl"])
        # sys.argv.extend(["-i", "Video/cosmetics/The_Ride_Or_Die_Makeup_Tag.mp4", "-o", "output.mp4"])

    parser = argparse.ArgumentParser(description="Text recognition in video")
    parser.add_argument("-i", "--in_video", dest="in_vid_file", help="Input video file name")
    parser.add_argument("-o", "--out_video", dest="out_vid_file", help="Output video file name")
    parser.add_argument("-d", "--dump_file", dest="dump_file", default=None, help="Dump file name")
    parser.add_argument("-n", "--frame_num", dest="frame_idx", default=-1, type=int,
                        help="Frame number to be processed")
    args = parser.parse_args()
    args.pkl_file = '.'.join(args.out_vid_file.split('.')[:-1]) + '.pkl'

    if args.dump_file:
        analyze_dump(args.dump_file)
    elif args.frame_idx >= 0:
        main_image(args)
    else:
        main_video(args)
