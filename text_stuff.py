
from PIL import Image, ImageDraw
import io, math
import numpy as np
import onnxruntime
import cv2

def is_n_times_div_by_2(value, n=4):
	for _ in range(n):
		if value % 2 != 0:
			return False
		value /= 2
	return True

def nearest_n_times_div_by_2(value, n=4):
	while True:
		if is_n_times_div_by_2(value, n):
			return value
		value += 1

def pad_image(image):
	c, h, w = image.shape
	new_h = nearest_n_times_div_by_2(h)
	new_w = nearest_n_times_div_by_2(w)
	new_image = np.zeros([c, new_h, new_w], dtype=np.float32)
	new_image[:, :h, :w] = image
	return new_image

class Alphabet:
	def __init__(self):
		self.char_groups = [
			"0OoОо",
			"1",
			"2",
			"3ЗзЭэ",
			"4",
			"5",
			"6б",
			"7",
			"8",
			"9",
			"!",
			"?",
			"#",
			"$",
			"%",
			"&",
			"@",
			"([{",
			"<",
			")]}",
			">",
			"+",
			"-_",
			"*",
			"/",
			"\\",
			".,",
			":;",
			"\"'",
			"^",
			"~",
			"=",
			"|lI",
			"AА",
			"aа",
			"BВв",
			"bЬьЪъ",
			"CcСс",
			"D",
			"d",
			"EЕЁ",
			"eеё",
			"F",
			"f",
			"G",
			"g",
			"HНн",
			"h",
			"i",
			"J",
			"j",
			"KКк",
			"k",
			"L",
			"MМм",
			"m",
			"N",
			"n",
			"PpРр",
			"R",
			"r",
			"Q",
			"q",
			"Ss",
			"TТт",
			"t",
			"U",
			"u",
			"Vv",
			"Ww",
			"XxХх",
			"Y",
			"yУу",
			"Zz",
			"Б",
			"Гг",
			"Дд",
			"Жж",
			"ИиЙй",
			"Лл",
			"Пп",
			"Фф",
			"Цц",
			"Чч",
			"ШшЩщ",
			"Ыы",
			"Юю",
			"Яя"
		]

		self.dict = {}
		for code, char_group in enumerate(self.char_groups):
			for char in char_group:
				self.dict[char] = code

alphabet = Alphabet()

class Prediction:
	def __init__(self, item):
		self.maybe_blank = False
		self.codepoints = ''

		item = item.squeeze()
		indexes = range(item.shape[0])
		indexes = sorted(indexes, key=lambda i: -item[i])
		for index in indexes:
			if item[index] < -10:
				break;
			if index:
				self.codepoints += alphabet.char_groups[index-1]
			else:
				self.maybe_blank = True

def match_text(predictions, x, query):
	y = 0;
	while x < len(predictions) and (y < len(query)):
		if query[y] in predictions[x].codepoints:
			y += 1
			x += 1
			continue
		if (y > 0) and (query[y-1] in predictions[x].codepoints):
			x += 1
			continue
		if predictions[x].maybe_blank:
			x += 1;
			continue
		break
	return y == len(query)

def image_contains_text(original_image, text):
	original_image = Image.open(io.BytesIO(original_image))
	image = np.array(original_image)
	image = image[:,:,:3]
	h, w, c = image.shape
	x = image / 255
	x = (x - (0.485, 0.456, 0.406)) / (0.229, 0.224, 0.225)
	x = x.transpose([2, 0, 1])    # [h, w, c] to [c, h, w]
	x = pad_image(x)
	x = x[np.newaxis]             # [c, h, w] to [b, c, h, w]
	ort_session = onnxruntime.InferenceSession("TextDetector.onnx")
	ort_inputs = {"input": x}
	ort_outs = ort_session.run(None, ort_inputs)
	y = ort_outs[0][0]
	up_heatmap = y[0,:h,:w]
	down_heatmap = y[1,:h,:w]
	up_heatmap = (np.clip(up_heatmap, 0, 1) * 255).astype(np.uint8)
	down_heatmap = (np.clip(down_heatmap, 0, 1) * 255).astype(np.uint8)
	up_heatmap = cv2.threshold(up_heatmap, 191, 255, cv2.THRESH_BINARY)[1]
	down_heatmap = cv2.threshold(down_heatmap, 191, 255, cv2.THRESH_BINARY)[1]
	_, up_labels, up_stats, _ = cv2.connectedComponentsWithStats(up_heatmap)
	_, down_labels, down_stats, _ = cv2.connectedComponentsWithStats(down_heatmap)

	textlines = []

	for up_stat in up_stats[1:]:
		up_x, up_y, up_w, up_h, _ = up_stat
		x = up_x + up_w // 2
		y_begin = up_y + up_h
		y_end = up_y + up_h * 2
		for y in range(y_begin, y_end):
			label = down_labels[y, x]
			if label:
				down_stat = down_stats[label]
				down_x, down_y, down_w, down_h, _ = down_stat
				left = min(up_x, down_x)
				top = up_y
				right = max(up_x + up_w, down_x + down_w)
				bottom = down_y + down_h
				textlines.append(original_image.crop((left, top, right, bottom)))
				break

	ort_session = onnxruntime.InferenceSession("TextRecognizer.onnx")

	query = ''
	for char in text:
		if char != ' ':
			query += char

	for image in textlines:
		ratio = math.ceil(image.width / image.height)
		image = image.resize([ratio * 32 * 2, 32])
		image = np.array(image)
		image = image[:,:,:3]
		h, w, c = image.shape

		x = image / 255
		x = (x - (0.485, 0.456, 0.406)) / (0.229, 0.224, 0.225)
		x = x.transpose([2, 0, 1])       # [h, w, c] to [c, h, w]
		x = x[np.newaxis]                # [c, h, w] to [b, c, h, w]
		x = x.astype(np.float32)

		ort_inputs = {"input": x}
		ort_outs = ort_session.run(None, ort_inputs)
		y = ort_outs[0]

		predictions = [Prediction(item) for item in y]
		for offset in range(0, len(predictions) - len(query)):
			if match_text(predictions, offset, query):
				return True

	return False

if __name__ == '__main__':
	with open("input.png", 'rb') as f:
		image_contains_text(f.read(), "Яндекс")
