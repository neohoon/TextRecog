#!/usr/bin/env python
"""
    * REFERENCE
        > http://stackoverflow.com/questions/24385714/detect-text-region-in-image-using-opencv
            - erosion and dilation based text area detection
            - implemented.
        > http://stackoverflow.com/questions/23506105/extracting-text-opencv
            - Might need to be read.

    * FEATURES
        - erosion and dilation -based text area detection
        - text recognition by using pyocr and tesseract.

    * NEXT STEP
        - Elaborate compare_and_merge function.
        - Handle the text box with non-zero angle.
        - Test inverted image.
        - misspelled-word correction.
        - Which one should be selected when two strings are similar to each other. (ISSUE#1)
"""

import os
import sys

import numpy as np
import pyocr
import pyocr.builders
import pyocr.tesseract
from PIL import Image as PIL_Image
import argparse
import re
import enchant
from HoonUtils import *
from difflib import SequenceMatcher

########################################################################################################################

LINE_BOX_MARGIN = (-6, -6, 10, 10)
AREA_RATIO_THRES = (0.8, 1.2)

TEXT_MATCH_RATIO_THRES = 0.75   # ratio unit.
TEXT_STT_POS_MATCH_THRES = 4        # pixel unit.

DEBUG__TEXT_BOX_FILTERING = False

########################################################################################################################


class ImgTextRecognition:

    def __init__(self, sel_lang='eng'):

        tools = pyocr.get_available_tools()
        if len(tools) == 0:
            print(" @ NO OCR tool found.\n")
            exit(1)

        # The tools are returned in the recommended order of usage.
        self.tool = tools[0]
        print(" > Will use tool '{}'".format(self.tool.get_name()))  # Ex: Will use tool 'libtesseract'

        langs = self.tool.get_available_languages()
        print(" > Available languages: '{}'".format(", ".join(langs)))
        self.lang = langs[langs.index(sel_lang)]
        print(" > Will use lang '{}'".format(self.lang))     # Ex: Will use lang 'fra'
        # Note that languages are NOT sorted in any way.
        # Please refer to the system locale settings for the default language to use.
        self.img_text_list = []

    @staticmethod
    def get_area_key(img_text):
        return img_text.img_area

    @staticmethod
    def get_len_key(img_text):
        return len(''.join(img_text.text).strip())

    def clear__img_text_list(self):
        self.img_text_list = []

    # ------------------------------------------------------------------------------------------------------------------
    class ImgText:
        def __init__(self, x=0, y=0, w=0, h=0):
            self.x = x
            self.y = y
            self.width = w
            self.height = h
            self.img_area = self.width * self.height
            self.img = []
            self.type = []
            self.text = []

        def get_width(self):
            return self.width

        def get_height(self):
            return self.height

        def get_stt_pnt(self):
            return tuple([self.x, self.y])

        def get_end_pnt(self):
            return tuple([self.x + self.width - 1, self.y + self.height - 1])

        def init_by_linebox(self, stt_pnt, end_pnt):
            self.x = stt_pnt[0]
            self.y = stt_pnt[1]
            self.width = end_pnt[0] - stt_pnt[0] + 1
            self.height = end_pnt[0] - stt_pnt[0] + 1

        def get_text_img(self, img):
            self.img = img[self.y:self.y+self.height, self.x:self.x+self.width]

        def get_vertex_pts(self):
            pnts = (self.x, self.y, self.x + self.width, self.y + self.height)
            return pnts

    # ------------------------------------------------------------------------------------------------------------------
    def extract_text_from_image(self, img_file, display=False):

        org_img = cv2.imread(img_file) if type(img_file) is str else img_file
        img_text_list = [self.extract_text_box(org_img, display=display), []]

        for img_text in img_text_list[-2]:
            img = PIL_Image.fromarray(img_text.img, 'RGB')
            # text = self.tool.image_to_string(img, lang=self.lang, builder=pyocr.builders.TextBuilder())
            # word_boxes = self.tool.image_to_string(img, lang=self.lang, builder=pyocr.builders.WordBoxBuilder())
            # digits = self.tool.image_to_string(img, lang=self.lang, builder=pyocr.tesseract.DigitBuilder())
            line_boxes = self.tool.image_to_string(img, lang=self.lang, builder=pyocr.builders.LineBoxBuilder())
            for line_box in line_boxes:
                x, y = line_box.position[0]
                w = line_box.position[1][0] - line_box.position[0][0] + 1
                h = line_box.position[1][1] - line_box.position[0][1] + 1
                (x, y, w, h) = tuple(map(sum, zip((x,y,w,h), LINE_BOX_MARGIN)))
                if w < 10 or h < 10 or line_box.content == '':
                    continue
                img_text_rst = self.ImgText(x + img_text.x, y + img_text.y, w, h)
                img_text_rst.get_text_img(org_img)
                img_text_rst.type = img_text.type
                img_text_rst.text.append(line_box.content)
                img_text_list[-1].append(img_text_rst)

                if False:
                    test_img = cv2.rectangle(org_img.copy(), img_text_rst.get_stt_pnt(), img_text_rst.get_end_pnt(),
                                             RED, 2)
                    print(img_text_rst.text)
                    # cv2.putText(test_img, img_text_rst.text, (img_text_rst.x, img_text_rst.y+img_text_rst.height+16),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, BLUE, 1, cv2.LINE_AA)
                    my_imshow(test_img)

        img_text_list.append(self.select_text_box_by_word(img_text_list[-1]))
        # img_text_list.append(self.select_text_box_by_same_area(img_text_list[-1]))
        # img_text_list.append(self.select_text_box_by_sentence(img_text_list[-1]))
        # img_text_list.append(self.select_text_box_by_common_area(img_text_list[-1], display=False, img=org_img))

        self.img_text_list = img_text_list[-1]
        return img_text_list[-1]

    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def print_img_text_list(img_text_list, display=False, desc=None):
        if display:
            print(" # img text list"),
            if desc:
                print("{}".format(desc))
            else:
                print("")
            for k in range(len(img_text_list)):
                img_text = img_text_list[k]
                print(" {:>2d} : ({:>4d}, {:<4d}) - {:>4d} x {:<4d} = {:>6d} :".
                      format(k, img_text.x, img_text.y, img_text.width, img_text.height, img_text.img_area)),
                for kk in range(len(img_text_list[k].text)):
                    print("{} ||".format(img_text.text[kk].encode('utf-8'))),
                print("")
            my_pause()

    # ------------------------------------------------------------------------------------------------------------------
    #   Select just one text box from many text boxes, if any, which have very similar starting point and area based
    #   on AREA_RATIO_THRES.
    #   It is noted that all the text information is appended to text box.
    #
    def select_text_box_by_same_area(self, img_text_list, display=False, img=None):
        sorted_img_text_list = sorted(img_text_list, key=self.get_area_key)
        self.print_img_text_list(sorted_img_text_list, display=DEBUG__TEXT_BOX_FILTERING, desc="sorted by area")
        idx = 0
        if not img:
            display = False
        while idx < len(sorted_img_text_list) - 1:
            if display:
                test_img = cv2.rectangle(img.copy(), sorted_img_text_list[idx].get_stt_pnt(),
                                         sorted_img_text_list[idx].get_end_pnt(), RED, 4)
                test_img = cv2.rectangle(test_img, sorted_img_text_list[idx+1].get_stt_pnt(),
                                         sorted_img_text_list[idx+1].get_end_pnt(), BLUE, 2)
                my_imshow(test_img)

            img_text = self.merge_img_text(sorted_img_text_list[idx], sorted_img_text_list[idx+1])
            if img_text is not None:
                del sorted_img_text_list[idx+1]
                del sorted_img_text_list[idx]
                sorted_img_text_list.insert(idx, img_text)
            else:
                idx += 1

        self.print_img_text_list(sorted_img_text_list, display=DEBUG__TEXT_BOX_FILTERING, desc="selected by area")
        return sorted_img_text_list

    # ------------------------------------------------------------------------------------------------------------------
    #   Select just one text box from many text boxes, if any, which have very similar starting point and area based
    #   on AREA_RATIO_THRES.
    #   It is noted that all the text information is appended to text box.
    #
    def select_text_box_by_common_area(self, img_text_list, display=False, img=None):

        img_text_list_rst = []
        idx1 = 0
        while idx1 < len(img_text_list) - 1:
            idx2 = idx1 + 1
            while idx2 < len(img_text_list):
                vertex_pts, ratio1, ratio2 = find_common_area(img_text_list[idx1].get_vertex_pts(),
                                                              img_text_list[idx2].get_vertex_pts())
                if ratio1 != 0:
                    if display and img is not None:
                        test_img = cv2.rectangle(img.copy(), img_text_list[idx1].get_stt_pnt(),
                                                 img_text_list[idx1].get_end_pnt(), RED, 2)
                        test_img = cv2.rectangle(test_img, img_text_list[idx2].get_stt_pnt(),
                                                 img_text_list[idx2].get_end_pnt(), BLUE, 2)
                        test_img = cv2.rectangle(test_img, vertex_pts[0:2], vertex_pts[2:], BLACK, -1)
                        my_imshow(test_img)
                    if ''.join(img_text_list[idx1].text) in ''.join(img_text_list[idx2].text):
                        img_text_list[idx1] = img_text_list[idx2]
                        del img_text_list[idx2]
                    elif ''.join(img_text_list[idx2].text) in ''.join(img_text_list[idx1].text):
                        del img_text_list[idx2]
                    else:
                        idx2 += 1
                else:
                    idx2 += 1
            img_text_list_rst.append(img_text_list[idx1])
            idx1 += 1

        self.print_img_text_list(img_text_list, display=DEBUG__TEXT_BOX_FILTERING, desc="selected by common area")
        return img_text_list_rst

    # ------------------------------------------------------------------------------------------------------------------
    #   Select the text box which has words which are in dictionary, or which has suggested words by enchant python
    #   package.
    #
    def select_text_box_by_word(self, img_text_list):

        en_dict = enchant.Dict("en_US")
        img_text_list_rst = []

        for img_text in img_text_list:
            try:
                word_list = filter(None, re.sub('[^a-zA-Z]+', ' ', ' '.join(img_text.text).encode('utf-8')).split(' '))
                for word in word_list:
                    if word != '':
                        if en_dict.check(word):
                            raise LocalBreak()
            except LocalBreak:
                img_text_list_rst.append(img_text)
                pass

        self.print_img_text_list(img_text_list_rst, display=DEBUG__TEXT_BOX_FILTERING, desc="selected by word")

        return img_text_list_rst

    # ------------------------------------------------------------------------------------------------------------------
    #   Select just one text box from many text boxes, if any, which have very similar sentence and very close starting
    #   point.
    #   It is noted that all the text information is appended to text box.
    #
    def select_text_box_by_sentence(self, img_text_list):

        sorted_img_text_list = sorted(img_text_list, key=self.get_len_key)
        self.print_img_text_list(sorted_img_text_list, display=DEBUG__TEXT_BOX_FILTERING, desc="sorted by text length")

        idx = 0
        while idx < len(sorted_img_text_list) - 1:
            match_ratio = SequenceMatcher(None, ''.join(sorted_img_text_list[idx].text).strip(),
                                          ''.join(sorted_img_text_list[idx+1].text).strip()).ratio()
            if match_ratio > TEXT_MATCH_RATIO_THRES \
                    and abs(sorted_img_text_list[idx].x - sorted_img_text_list[idx+1].x) < TEXT_STT_POS_MATCH_THRES \
                    and abs(sorted_img_text_list[idx].y - sorted_img_text_list[idx+1].y) < TEXT_STT_POS_MATCH_THRES:
                del sorted_img_text_list[idx+1]
            else:
                idx += 1

        self.print_img_text_list(sorted_img_text_list, display=DEBUG__TEXT_BOX_FILTERING, desc="selected by sentence")
        return sorted_img_text_list

    # ------------------------------------------------------------------------------------------------------------------
    def show_img_text(self, img_file):

        org_img = cv2.imread(img_file) if type(img_file) is str else img_file
        print(" # Total {:d} line texts detected".format(len(self.img_text_list)))
        idx = 1
        for img_text in self.img_text_list:
            print("   - {:2d}-th line box: {}".format(idx, " || ".join(img_text.text).encode('utf-8')))
            test_img = cv2.rectangle(org_img.copy(), img_text.get_stt_pnt(), img_text.get_end_pnt(), RED, 2)
            cv2.putText(test_img, img_text.text[0].encode('utf-8'), (10, org_img.shape[0]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, BLUE, 2, cv2.LINE_AA)
            my_imshow(test_img)
            idx += 1

    # ------------------------------------------------------------------------------------------------------------------
    def overlay_all_texts(self, img):

        overlay_img = img.copy()
        for img_text in self.img_text_list:
            test_img = cv2.rectangle(overlay_img, img_text.get_stt_pnt(), img_text.get_end_pnt(), RED, 2)
            cv2.putText(test_img, img_text.text.encode('utf-8'), (test_img.x + 4, test_img.y + test_img.height - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, BLUE, 2, cv2.LINE_AA)

        return overlay_img

    # ------------------------------------------------------------------------------------------------------------------
    def extract_text_box(self, img_file, display=False):

        tar_img, dilated_img = [None]*2, [None]*2
        contours, hierarchy = [None]*2, [None]*2
        img_text_list = []

        org_img = cv2.imread(img_file) if type(img_file) is str else img_file
        overlay_img = org_img.copy()
        gray_img = cv2.cvtColor(org_img, cv2.COLOR_BGR2GRAY)
        ret, mask_img = cv2.threshold(gray_img, 180, 255, cv2.THRESH_BINARY)
        src_img = cv2.bitwise_and(gray_img, gray_img, mask=mask_img)
        ret, tar_img[0] = cv2.threshold(src_img, 180, 255, cv2.THRESH_BINARY)         # for black text...
        ret, tar_img[1] = cv2.threshold(src_img, 180, 255, cv2.THRESH_BINARY_INV)     # for white text...

        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        for k in range(1,2):        # ISSUE: black text image excluded.
            # Remove noisy portion...
            # to manipulate the orientation of dilution ,
            # large x means horizontally dilating  more, large y means vertically dilating more
            if k is 0: dilated_img[k] = cv2.erode( tar_img[k], kernel, iterations=9)
            else:      dilated_img[k] = cv2.dilate(tar_img[k], kernel, iterations=9)
            ret, contours[k], hierarchy[k] = cv2.findContours(np.array(dilated_img[k]).copy(), cv2.RETR_EXTERNAL,
                                                              cv2.CHAIN_APPROX_NONE)

            for contour in list(contours[k]):

                # get rectangle bounding contour
                [x, y, w, h] = cv2.boundingRect(contour)

                # Don't plot small false positives that aren't text
                if w < 10 or h < 10:
                    continue

                img_text = self.ImgText(x, y, w, h)

                img_text.img = org_img[y:y + h, x:x + w]
                img_text.type = k

                img_text_list.append(img_text)

                # draw rectangle around contour on original image
                if False:
                    color = RED if k is 0 else BLUE
                    cv2.rectangle(overlay_img, (x, y), (x + w, y + h), color, 2)
                    my_imshow(overlay_img)

        if False:
            size = gray_img.shape
            img_merge = np.zeros((size[0]*7, size[1]), dtype=np.uint8)
            img_merge[size[0]*0:size[0]*1,:] = gray_img
            img_merge[size[0]*1:size[0]*2,:] = mask_img
            img_merge[size[0]*2:size[0]*3,:] = src_img
            img_merge[size[0]*3:size[0]*4,:] = tar_img[0]
            img_merge[size[0]*4:size[0]*5,:] = tar_img[1]
            img_merge[size[0]*5:size[0]*6,:] = dilated_img[0]
            img_merge[size[0]*6:size[0]*7,:] = dilated_img[1]
            cv2.imwrite('img_merge.jpg', img_merge)

        # write original image with added contours to disk
        if display:
            my_imshow(overlay_img)

        return img_text_list

    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def merge_img_text(img_text_1, img_text_2):

        x1, y1 = max(img_text_1.x, img_text_2.x), max(img_text_1.y, img_text_2.y)
        x2, y2 = min(img_text_1.x + img_text_1.width - 1,  img_text_2.x + img_text_2.width - 1), \
                 min(img_text_1.y + img_text_1.height - 1, img_text_2.y + img_text_2.height - 1)

        if x2 < x1 or y2 < y1:
            return None

        img_area = (x2 - x1 + 1) * (y2 - y1 + 1)
        area_ratio = img_area / float(img_text_1.img_area)
        if area_ratio < AREA_RATIO_THRES[0]:
            return None
        area_ratio = img_area / float(img_text_2.img_area)
        if area_ratio < AREA_RATIO_THRES[0]:
            return None

        area_ratio = img_text_1.img_area / float(img_text_2.img_area)
        if area_ratio < AREA_RATIO_THRES[0] or AREA_RATIO_THRES[1] < area_ratio:
            return None

        if "".join(img_text_1.text) != "".join(img_text_2.text):
            for k in range(len(img_text_2.text)):
                img_text_1.text.append(img_text_2.text[k])

        return img_text_1

########################################################################################################################


def main(arg):

    if os.path.exists(arg.img_file):

        img_text = ImgTextRecognition()
        img_text.extract_text_from_image(arg.img_file, display=False)
        img_text.show_img_text(arg.img_file)

    else:
        print(" @ Error: file not found {}".format(arg.img_file))

########################################################################################################################

if __name__ == "__main__":

    if len(sys.argv) == 1:
        sys.argv.extend(["-i", "Img/road_sign_2.jpg"])

    parser = argparse.ArgumentParser(description="Text recognition in image")
    parser.add_argument("-i", "--image", dest="img_file", help="Image file name")
    args = parser.parse_args()

    main(args)

########################################################################################################################
########################################################################################################################
########################################################################################################################
"""




















"""

