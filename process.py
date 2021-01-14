import warnings
warnings.filterwarnings('ignore')

import pytesseract
import cv2
import numpy as np
from collections import defaultdict 
# from nltk.tokenize import LineTokenizer 
# import spacy
# import re
# # import en_core_web_lg
# # import en_core_web_md
# # nlp = en_core_web_md.load()
# nlp = spacy.load("en_core_web_sm")
# # fetching education dates
# from spacy.matcher import Matcher
# matcher = Matcher(nlp.vocab)


from pdf2image import convert_from_path,convert_from_bytes

def pdftoimage(pdffile):
	pages = convert_from_path(pdffile)
	pages[0].save("img.jpg", "JPEG")

def preprocess_img(img):

  # Make HSV and extract S, i.e. Saturation 
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	s=hsv[:,:,1]

	# Make greyscale version and inverted, thresholded greyscale version
	gr = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	_,grinv = cv2.threshold(gr,127,255,cv2.THRESH_BINARY_INV)

	# Find row numbers of rows with colour in them
	meanSatByRow=np.mean(s,axis=1)
	rows = np.where(meanSatByRow>50)
 
	# Replace selected rows with those from the inverted, thresholded image
	gr[rows]=grinv[rows]

	return gr

def extract_text(gr_img):

	custom_config = r'-c --psm 6'
	text = pytesseract.image_to_string(gr_img, lang="eng")

	return text

def process_pdf(pdffile):
	pdftoimage(pdffile)
	image = cv2.imread("img.jpg")
	#converting image into gray scale image
	gray_image = preprocess_img(image)
	text=extract_text(gray_image)
	return text


def extract_dates(resume_text):
    nlp_text = nlp(resume_text)
    
    # First name and Last name are always Proper Nouns
    
    pattern1 = [{'POS': 'NUM'},{'POS': 'SYM'}, {'POS': 'NUM'}]
    pattern2 = [{'POS': 'NUM'},{'POS': 'PUNCT'}, {'POS': 'PROPN'}]
    pattern3 = [{'POS': 'PROPN'},{'TAG': 'HYPH'}, {'POS': 'PROPN'}]
    matcher.add('NAME', None, pattern1,pattern2,pattern3)
    
    matches = matcher(nlp_text)
    a=[]
    for match_id, start, end in matches:
        span = nlp_text[start:end]
        a.append(span.text)
    return a

def get_data(pdffile):
	text=process_pdf(pdffile)
	
	return text

	
# print(process_pdf("resume_1.pdf"))